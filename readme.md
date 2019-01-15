## Rumor Detection on Social Media: A Multi-View Model using Self-Attention Mechanism

This repository is the source code of paper: Rumor Detection on Social Media: A Multi-View Model using Self-Attention Mechanism

### Dependencies
Python 3.6
PyTorch 0.4.1
Numpy 1.15.4


#### Preparation

1. Download Weibo dataset at http://alt.qcri.org/~wgao/data/rumdect.zip

2. Download Tencent AI Lab Embedding Corpus at https://ai.tencent.com/ailab/nlp/embedding.html and place it at path 'embedding_data/tencent/Tencent_AILab_ChineseEmbedding.txt'

3. Download Baidu Senta Corpus at https://github.com/baidu/Senta

4. Download Bert pre-trained model at https://github.com/google-research/bert

5. Fine-tuning Bert pre-trained model with Baidu Senta Corpus. You can refer Bert readme file to finish this work.

6. Extract source post and replies of Weibo dataset(save the source post and all replies of each event in a file), get sentiment view input embbedding using the fine-tuned Bert model and pack the result file into 'embedding_data/bert/weibo_sentiment.tar'

7. Use Jieba to process the content and replies of each event file(for each post in each event file, cut 'text' into a word list through Jieba) and save the corpus in directory 'text_data/weibo/dataset', place the label file at path 'text_data/weibo/label/weibo.txt'

#### Running

Run the command to train content view model and get prediction in test dataset:

```bash
python train.py -corpus weibo -content
```

Run the command to train reply view model and get prediction in test dataset:

```bash
python train.py -corpus weibo -reply
```

Run the command to train sentiment view model and get prediction in test dataset:

```bash
python train.py -corpus weibo -bert_vector -bert_vector_type sentiment
```

The results are saved at directory 'text_data/weibo/result'



After getting three views' predictions of test dataset(the train and test dataset are the same for three views), please deploy a vote program to get final prediction.




