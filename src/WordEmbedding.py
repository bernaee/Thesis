import gzip
import numpy as np
import logging


def load_all_word_embeding_dictionary(path):
    logging.info('Reading word vectors...')
    with gzip.open(path, 'rt') as fin:
        n, d = map(int, fin.readline().split())
        labels = []
        vectors = []
        for idx, line in enumerate(fin):
            tokens = line.rstrip().split(' ')
            labels.append(tokens[0])
            vectors.append(np.array(tokens[1:]))
    return labels, vectors


def get_word_embeding_dictionary(path):
    logging.info('Reading word vectors...')
    with gzip.open(path, 'rt') as fin:
        labels = []
        for idx, line in enumerate(fin):
            tokens = line.rstrip().split(' ')
            labels.append(tokens[0])
    return labels


def find_word_vector(path, word_indexes):
    word_vectors = dict()
    with gzip.open(path, 'rt') as fin:
        for idx, line in enumerate(fin):
            if idx in word_indexes:
                tokens = line.rstrip().split(' ')
                word_vector = tokens[1:]
                word_vectors[idx] = word_vector
    return word_vectors


def find_word_index(we_words, word):
    try:
        w_idx = we_words.index(word)
        return w_idx
    except ValueError:
        return False


def read_fastText_word_embeddings(we_path, words, n_words, embedding_size=300):
    we_dict = get_word_embeding_dictionary(we_path)
    word_vector_indexes = dict()
    unk_word_vector = [round(np.random.uniform(-0.1, 0.1), 5) for i in range(300)]
    unk_word_vector_indexes = dict()
    unk_word_vector_indexes[1] = unk_word_vector

    for vocab_idx, word in enumerate(words):
        if word == '<UNK>':
            continue
        w_idx = find_word_index(we_dict, word)
        ### next step is to add lemma search
        if not w_idx:
            w_idx = find_word_index(we_dict, word.lower())
            if not w_idx:
                first_part = word.split("'")[0]
                w_idx = find_word_index(we_dict, first_part)
                if not w_idx:
                    unk_word_vector_indexes[vocab_idx] = unk_word_vector
                else:
                    word_vector_indexes[vocab_idx] = w_idx
            else:
                word_vector_indexes[vocab_idx] = w_idx
        else:
            word_vector_indexes[vocab_idx] = w_idx

    word_vectors = find_word_vector(we_path, sorted(set(word_vector_indexes.values())))
    embedding_matrix = np.zeros((n_words, embedding_size))
    for v_idx, wv_idx in word_vector_indexes.items():
        embedding_vector = word_vectors[wv_idx]
        if embedding_vector is not None:
            embedding_matrix[v_idx] = np.array(embedding_vector)
    for v_idx, embedding_vector in unk_word_vector_indexes.items():
        if embedding_vector is not None:
            embedding_matrix[v_idx] = np.array(embedding_vector)
    n_unk_words = len(unk_word_vector_indexes)
    logging.info('Number of found tokens: %s' % n_words - n_unk_words)
    logging.info('Number of unknown tokens: %s' % n_unk_words)
    return embedding_matrix, unk_word_vector
