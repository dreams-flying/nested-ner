import json

train_data_path = 'raw_data/train.json'
dev_data_path = 'raw_data/dev.json'

fw = open('entity_label.json', 'w', encoding='utf-8')

label2id = {}
with open(train_data_path, encoding='utf-8') as f1, open(dev_data_path, encoding='utf-8') as f2:
    for line in f1:
        line = json.loads(line)
        tmp = []
        for label_entity in line['label']:
            if label_entity not in label2id:
                label2id[label_entity] = len(label2id)
    for line in f2:
        line = json.loads(line)
        tmp = []
        for label_entity in line['label']:
            if label_entity not in label2id:
                label2id[label_entity] = len(label2id)
l1 = json.dumps(label2id, ensure_ascii=False)
fw.write(l1)