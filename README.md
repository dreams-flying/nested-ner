# 嵌套命名实体识别（nested-ner）
&emsp;&emsp;本项目为笔者在nested-ner的一次尝试，在cluener和GENIA两个数据集上进行了测试，取得了不错的效果。</br>
&emsp;&emsp;不同于传统的ner研究，嵌套命名实体识别（nested named entity recognition,nested ner）在识别的实体中存在嵌套的情况。比如北京大学不仅是一个组织，同时北京也是一个地点。
# 所需环境
Python==3.6</br>
tensorflow==1.14.0</br>
keras==2.3.1</br>
bert4keras==0.10.6</br>
笔者使用了开源的bert4keras，一个keras版的transformer模型库。bert4keras的更多介绍参见[这里](https://github.com/bojone/bert4keras)。
# 项目目录
├── bert4keras</br>
├── data    存放数据</br>
├── pretrained_model    存放预训练模型</br>
├── save    存放已训练好的模型</br>
├── nested_ner_train.py    训练代码</br>
├── nested_ner_predict.py    评估和测试代码
# 数据集
[CLUENER](https://www.cluebenchmarks.com/introduce.html)</br>
[GENIA](http://www.geniaproject.org/genia-corpus/pos-annotation)
# 使用说明
1.[下载预训练语言模型](https://github.com/google-research/bert#pre-trained-models)</br>
&emsp;&emsp;中文数据集可采用BERT-Base, Chinese等模型</br>
&emsp;&emsp;英文数据集可采用BERT-Base, Cased等模型</br>
&emsp;&emsp;更多的预训练语言模型可参见[bert4keras](https://github.com/bojone/bert4keras)给出的权重。</br>
2.构建数据集</br>
&emsp;&emsp;中文：将下载的[cluener数据集](https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip)放到data/cluener/raw_data文件夹下</br>
&emsp;&emsp;&emsp;&emsp;&emsp;运行generate_label.py生成entity_label.json</br>
&emsp;&emsp;英文：将下载的genia数据集放到data/GENIA/raw_data/genia文件夹下，</br>
&emsp;&emsp;&emsp;&emsp;&emsp;a.运行```parse_genia.py```，生成的数据将会放在data/GENIA/raw_data/processed_genia文件夹下</br>
&emsp;&emsp;&emsp;&emsp;&emsp;b.运行```gen_data_for_genia.py```，</br>
&emsp;&emsp;&emsp;&emsp;&emsp;生成的genia_train.json、genia_dev.json、genia_test.json数据会放在data/GENIA/data文件夹下</br>
&emsp;&emsp;&emsp;&emsp;&emsp;c.运行```generate_label.py```，生成entity_label.json</br>
&emsp;&emsp;英文数据的处理参考[nested-ner-tacl2020-transformers](https://github.com/yahshibu/nested-ner-tacl2020-transformers)项目的数据处理。</br>
3.训练模型</br>
```
python nested_ner_train.py
```
4.评估和测试</br>
```
python nested_ner_predict.py
```
# 结果
这里只对f1值进行了统计。
| 数据集 | train | dev | test |
| :------:| :------: | :------: | :------: |
| CLUENER | 90.721 | 80.412 | 79.340 |
| GENIA |  |  |  |
