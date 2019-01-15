import glob
import random
import tarfile
import os

from config import opt
from event import Event


class Dataset:
    def __init__(self, args, mode, embedding_matrix):
        self.args = args
        self.mode = mode
        self.embedding_matrix = embedding_matrix

    def events(self):
        file_list = glob.glob(os.path.join(self.args.corpus_path, "*.json"))
        random.seed(self.args.random_seed)
        random.shuffle(file_list)
        # divide dataset to train and test
        if self.mode == "train":
            data_set = file_list[:int(0.9*0.75*len(file_list))]
        elif self.mode == 'test':
            data_set = file_list[int(0.9*0.75*len(file_list)):int(0.9*1*len(file_list))]
        elif self.mode == "val":
            data_set = file_list[int(0.9*len(file_list)):]

        # open label file
        if self.args.corpus == 'twitter':
            with open(opt.label_twitter) as fp_label:
                dict_label = {}
                for line in fp_label:
                    splits = line.split("	")
                    idx = splits[0].split(":")[1]
                    label = splits[1].split(':')[1]
                    dict_label[idx] = label

        elif self.args.corpus == 'weibo':
            with open(opt.label_weibo) as fp_label:
                dict_label = {}
                for line in fp_label:
                    splits = line.split("	")
                    idx = splits[0].split(":")[1]
                    label = splits[1].split(':')[1]
                    dict_label[idx] = label

        # open bert_vector
        if self.args.bert_vector:
            with tarfile.open(self.args.bert_vector_path) as bert_vector_tar:
                tar_dict = {}
                for tarinfo in bert_vector_tar:
                    filename = os.path.split(tarinfo.name)[1]
                    tar_dict[filename] = tarinfo

        Events = []
        for i in range(len(data_set)):
            file = data_set[i]
            filename = os.path.split(file)[1]
            print('Reading',self.mode, 'datasets', 'Read file num:', i, 'Reading file:', filename)
            label = dict_label[filename.split('.')[0]]
            if self.args.bert_vector:
                bert_vector_file = tar_dict[filename]
                Events.append(Event(self.args, file, int(label), self.embedding_matrix, bert_vector_file))
            else:
                Events.append(Event(self.args, file, int(label), self.embedding_matrix))

        return Events


if __name__ == '__main__':
    pass

