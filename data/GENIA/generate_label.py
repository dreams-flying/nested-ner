import json

train_data_path = 'data/genia_train.json'
dev_data_path = 'data/genia_dev.json'
test_data_path = 'data/genia_test.json'

fw = open('entity_label.json', 'w', encoding='utf-8')

label2id = {}
id2label = {}
with open(train_data_path, encoding='utf-8') as f1, open(dev_data_path, encoding='utf-8') as f2, open(test_data_path, encoding='utf-8') as f3:
    for line in f1:
        line = json.loads(line)
        tmp = []
        for label_entity in line['label']:
            if label_entity not in label2id:
                id2label[len(label2id)] = label_entity
                label2id[label_entity] = len(label2id)
    for line in f2:
        line = json.loads(line)
        tmp = []
        for label_entity in line['label']:
            if label_entity not in label2id:
                id2label[len(label2id)] = label_entity
                label2id[label_entity] = len(label2id)
    for line in f3:
        line = json.loads(line)
        tmp = []
        for label_entity in line['label']:
            if label_entity not in label2id:
                id2label[len(label2id)] = label_entity
                label2id[label_entity] = len(label2id)

l = json.dumps(label2id, ensure_ascii=False)
fw.write(l)