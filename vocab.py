import glob
import os
import pickle
import json


def build_vocab(args):
    vocab_file_name = args.corpus_path + '{0}_vocab.dat'.format(args.corpus)
    if os.path.exists(vocab_file_name):
        print('loading vocabulary:', vocab_file_name)
        vocabulary = pickle.load(open(vocab_file_name, 'rb'))
    else:
        print('loading vocabulary...')
        vocabulary = set()
        file_list = glob.glob(os.path.join(args.corpus_path, "*.json"))
        for datafile in file_list:
            with open(datafile) as fp:
                for tweet in json.load(fp):
                    tokens = tweet['text']
                    for token in tokens:
                        vocabulary.add(token)
        vocabulary = list(vocabulary)
        print('saving vocabulary...')
        pickle.dump(vocabulary, open(vocab_file_name, 'wb'))
    return vocabulary

