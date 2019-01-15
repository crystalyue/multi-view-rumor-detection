import os
import numpy as np
import pickle


def load_word_vec(args, path, vocab):

    vocab = set(vocab)
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as fp:
        word2vec = {}
        filelines = fp.readlines()[1:]
        for i, line in enumerate(filelines):
            tokens = line.rstrip().split()
            if vocab is None or ' '.join(tokens[:-args.embedding_dim]) in vocab:
                word2vec[tokens[0]] = np.asarray(tokens[-args.embedding_dim:], dtype='float64')
    return word2vec

def build_embedding_matrix(args, vocab):
    embedding_matrix_file_name = args.corpus_path + '{0}_embedding_matrix.dat'.format(args.corpus)

    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = {}
        fname = args.embedding_path
        word2vec = load_word_vec(args, fname, vocab)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word in vocab:
            vec = word2vec.get(word)
            #print(word, vec)
            if vec is not None and (len(vec)) == args.embedding_dim: # words not found in embedding index will be all-zeros.
                embedding_matrix[word] = vec
            else:
                embedding_matrix[word] = np.zeros([args.embedding_dim])
        print('saving embedding_matrix')
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


if __name__ == '__main__':
    pass
