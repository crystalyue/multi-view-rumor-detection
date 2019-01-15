import json
import os
import numpy as np
import tarfile


class Event:
    def __init__(self, args, file, label=0, embedding_matrix=None, vector_file=None):
        self.args = args
        self.filename = os.path.split(file)[1]
        self.label = label
        self.embedding_matrix = embedding_matrix
        self.vector_file = vector_file

        # self.all_content = list(Content(json.load(line) for line in open(file)))
        self.content = json.load(open(file))[0]['text'][:self.args.content_length_limit]
        self.content_word_embedding_array = self.get_word_embedding_array(self.content)

        self.reply = list(dic['text'] for dic in json.loads(open(file).read())[
                                                   0: self.args.reply_number_limit * self.args.reply_sample_frequency
                                                   : self.args.reply_sample_frequency])
        self.reply_no_same = []
        for reply in self.reply:
            if reply not in self.reply_no_same:
                self.reply_no_same.append(reply)
        self.reply_no_same = self.reply_no_same
        self.reply_sentence_embedding_array = self.get_sentence_embedding_array(self.reply_no_same)

        # self.data = vocab.sen2id(self.original_content.tokens)+ vocab.sen2id(self.reply.tokens)
        self.len_reply = len(self.reply)
        self.sentence = [self.content] + self.reply

        # open bert vector file and extract it
        if self.args.bert_vector:
            with tarfile.open(args.bert_vector_path) as fp_tar:
                with fp_tar.extractfile(self.vector_file) as fp:
                    vector_json = json.loads(fp.readlines()[0].decode('utf-8'))
                    self.bert_vector = []
                    for i in range(0, len(vector_json))[
                                  0: self.args.reply_number_limit * self.args.reply_sample_frequency
                                  : self.args.reply_sample_frequency]:
                        if [vector_json[str(i)]] not in self.bert_vector:
                            self.bert_vector.append([vector_json[str(i)]])

    def get_word_embedding_array(self, content):
        word_embedding_array = []
        for word in content:
            word_embedding = self.embedding_matrix[word]
            word_embedding_array.append(word_embedding)
        word_embedding_array = np.array(word_embedding_array)
        return word_embedding_array

    def get_sentence_embedding_array(self, sentences):
        sentence_embedding_array = []
        for sentence in sentences:
            sentences_word_embedding = self.get_word_embedding_array(sentence)
            if sentences_word_embedding.shape[0] != 0:
                sentences_word_embedding = sentences_word_embedding.sum(axis=0)/sentences_word_embedding.shape[0]
                sentence_embedding_array.append(sentences_word_embedding)
        sentence_embedding_array = np.array(sentence_embedding_array)
        return sentence_embedding_array

if __name__ == '__main__':
    pass





