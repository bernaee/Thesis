import numpy as np
import logging
import fastText as fasttext


def read_fastText_word_embeddings(we_path, words, n_words, embedding_size=300):
    model = fasttext.load_model(we_path)
    model_words = model.get_words()
    unk_word_vector = [round(np.random.uniform(-0.1, 0.1), 5) for i in range(300)]
    sub_word_indexes = []
    unk_word_indexes = []
    embedding_matrix = np.zeros((n_words, embedding_size))
    for vocab_idx, word in enumerate(words):
        if word == '<UNK>':
            embedding_matrix[vocab_idx] = np.array(unk_word_vector)
            continue
        if word in model_words:
            embedding_vector = model.get_word_vector(word)
            embedding_matrix[vocab_idx] = np.array(embedding_vector)
        else:
            try:
                embedding_vector = model.get_word_vector(word)
                embedding_matrix[vocab_idx] = np.array(embedding_vector)
                sub_word_indexes.append(vocab_idx)
            except:
                embedding_matrix[vocab_idx] = np.array(unk_word_vector)
                unk_word_indexes.append(vocab_idx)
    n_sub_words = len(sub_word_indexes)
    n_unk_words = len(unk_word_indexes)
    logging.info('Number of tokens: %s' % n_words)
    logging.info('Number of sub tokens: %s' % n_sub_words)
    logging.info('Number of unknown tokens: %s' % n_unk_words)
    return embedding_matrix, unk_word_vector
