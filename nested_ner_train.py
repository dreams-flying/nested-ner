import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=1

import json
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.layers import PositionEmbedding
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from tqdm import tqdm

maxlen = 128
batch_size = 32
epoch = 20

#数据路径
train_data_path = 'data/cluener/raw_data/train.json'
dev_data_path = 'data/cluener/raw_data/dev.json'
entity_label_path = 'data/cluener/entity_label.json'

#模型保存路径
if not os.path.exists('save'):
    print('mkdir {}'.format('save'))
    os.makedirs('save')

Flag = True    #True:中文；False:英文

config_path = 'pretrained_model/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'pretrained_model/chinese_L-12_H-768_A-12/vocab.txt'

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            tmp = []
            for label_entity in line['label']:
                for k, v in line['label'][label_entity].items():
                    tmp.append((label_entity, k, v[0][0], v[0][1]))

            D.append(
                {
                    "text": line['text'],
                    "entity": tmp
                }
            )

    return D

# 加载数据集
train_data = load_data(train_data_path)
print(len(train_data))

valid_data = load_data(dev_data_path)
print(len(valid_data))

label2id = {}
id2label = {}
with open(entity_label_path, encoding='utf-8') as f:
    for line in f:
        line = json.loads(line)
        for k, v in line.items():
            if k not in label2id:
                id2label[len(label2id)] = k
                label2id[k] = len(label2id)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_subject_labels, batch_subject_ids, batch_entity_matrix = [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(d['text'], maxlen=maxlen)
            # 整理 [(entity, label)]
            entity_labels = []
            for label, entity, start, end in d['entity']:
                label = label2id[label]
                entity_id = tokenizer.encode(entity)[0][1:-1]
                entity_idx = search(entity_id, token_ids)
                if entity_idx != -1:
                    entity_label = (entity_idx, entity_idx + len(entity_id) - 1, label)
                    entity_labels.append(entity_label)

            if entity_labels:
                # 对应的entity_label标签
                entity_matrix = np.zeros((len(token_ids), len(label2id), 2))

                for e in entity_labels:
                    entity_matrix[e[0], e[2], 0] = 1
                    entity_matrix[e[1], e[2], 1] = 1

                # 构建batch
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_entity_matrix.append(entity_matrix)

                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_entity_matrix = sequence_padding(batch_entity_matrix)

                    yield [batch_token_ids, batch_segment_ids, batch_entity_matrix], None
                    batch_token_ids, batch_segment_ids, batch_entity_matrix = [], [], []

# 补充输入
entity_matrix = Input(shape=(None, len(label2id), 2), name='Entity-Matrix')

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    # model = "albert",   #预训练模型选择albert时开启
    return_keras_model=False,
)

output = PositionEmbedding(768, 768)(bert.model.output) #768为bert模型输出的维度

output = Dense(
    units=len(label2id) * 2,
    activation='sigmoid',
    kernel_initializer=bert.initializer)(output)

output = Lambda(lambda x: x**4)(output)
entity_preds = Reshape((-1, len(label2id), 2))(output)

object_model = Model(bert.model.inputs, entity_preds)

# 训练模型
train_model = Model(bert.model.inputs + [entity_matrix], entity_preds)
# train_model.summary()

mask = bert.model.get_layer('Embedding-Token').output_mask
mask = K.cast(mask, K.floatx())

object_loss = K.binary_crossentropy(entity_matrix, entity_preds)
object_loss = K.sum(K.mean(object_loss, 3), 2)


object_loss = K.sum(object_loss * mask) / K.sum(mask)

train_model.add_loss(object_loss)

AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = Adam(lr=1e-5)
train_model.compile(optimizer=optimizer)

def extract_entitys(text, Flag=True):
    """抽取输入text所包含的实体
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)

    entity_labels = []
    entity_preds = object_model.predict([[token_ids], [segment_ids]])
    for entity_pred in entity_preds:
        start = np.where(entity_pred[:, :, 0] > 0.6)
        end = np.where(entity_pred[:, :, 1] > 0.5)
        for _start, predicate1 in zip(*start):
            for _end, predicate2 in zip(*end):
                if _start <= _end and predicate1 == predicate2:
                    entity_labels.append((predicate1, (_start, _end)))
                    break

    output_result = []
    if Flag:#中文预测
        for p, o in entity_labels:
            entity = tokenizer.decode(token_ids[o[0]:o[1] + 1], tokens[o[0]:o[1] + 1])
            start = search(entity, text)
            if start != -1:
                output_result.append((id2label[p], entity, start, start+len(entity)-1))
    else:
        text = text.strip()
        raw_tokens = text.split(' ')
        for p, o in entity_labels:
            entity = tokenizer.decode(token_ids[o[0]:o[1] + 1], tokens[o[0]:o[1] + 1])
            entity_list = entity.split(' ')
            start_list = [i for i, x in enumerate(raw_tokens) if x == entity_list[0]]
            end_list = [i for i, x in enumerate(raw_tokens) if x == entity_list[-1]]
            if len(start_list) > 0 and len(end_list) > 0:
                output_result.append((id2label[p], entity, start_list[0], end_list[0]+1))

    return output_result

def evaluate(data, Flag):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    pbar = tqdm()
    for d in data:
        R = set([entity_label for entity_label in extract_entitys(d['text'], Flag=Flag)])
        T = set([entity_label for entity_label in d['entity']])

        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' %
                             (f1, precision, recall))
    pbar.close()
    return f1, precision, recall

class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = -0.1
    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data, Flag)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            train_model.save_weights('./save/best_model.weights')
        print('f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1))


if __name__ == '__main__':

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epoch,
        callbacks=[evaluator]
    )