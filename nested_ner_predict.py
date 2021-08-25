import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=1

import json
import numpy as np
from bert4keras.layers import PositionEmbedding
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import open
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from tqdm import tqdm
import time

start = time.time()

maxlen = 128

#数据路径
train_data_path = 'data/cluener/raw_data/train.json'
dev_data_path = 'data/cluener/raw_data/dev.json'
entity_label_path = 'data/cluener/entity_label.json'

Flag = True    #True:中文；False:英文

#模型路径
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

# 补充输入
entity_matrix = Input(shape=(None, len(label2id), 2), name='Object-Labels')

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
    if Flag:    #中文预测
        for p, o in entity_labels:
            entity = tokenizer.decode(token_ids[o[0]:o[1] + 1], tokens[o[0]:o[1] + 1])
            start = search(entity, text)
            if start != -1:
                output_result.append((id2label[p], entity, start, start+len(entity)-1))
    else:   #英文预测
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

def evaluate(data, Flag=True):
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

def evaluate2(data, Flag=True):
    """评估函数，计算f1、precision、recall
       此评估函数在评估时没有将实体位置进行输出，在英文数据集下f1值会有所提高
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    pbar = tqdm()
    for d in data:
        tempList1 = []
        for entity_label in extract_entitys(d['text'], Flag=Flag):
            tempList1.append((entity_label[0], entity_label[1]))
        R = set(tempList1)
        tempList2 = []
        for entity_label in d['entity']:
            tempList2.append((entity_label[0], entity_label[1]))
        T = set(tempList2)

        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' %
                             (f1, precision, recall))
    pbar.close()
    return f1, precision, recall

if __name__ == '__main__':
    # 加载模型
    train_model.load_weights('./save/best_model.weights')

    # 测试
    text1 = '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，'
    R = extract_entitys(text1, Flag=Flag)
    print(R)

    #评估数据
    # f1, precision, recall = evaluate(valid_data, Flag)
    # print('f1: %.5f, precision: %.5f, recall: %.5f\n' % (f1, precision, recall))

    delta_time = time.time() - start
    print("time:", delta_time)