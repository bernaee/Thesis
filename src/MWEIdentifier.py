from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras.layers import *
from keras_contrib.layers import CRF
import logging
import math


class MWEIdentifier:
    def __init__(self, language, mwe):
        logging.info('Initialize MWEIdentifier for %s' % language)
        self.language = language
        self.mwe = mwe

    def set_model_cfg(self, model_cfg):
        logging.info('Setting model configuration...')
        self.model_cfg = model_cfg

    def set_params(self, params):
        logging.info('Setting params...')
        self.n_units = params['n_units']
        self._dropout = params['dropout']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']

    def set_char_cnn_model_params(self, params):
        logging.info('Setting char cnn params...')
        self.char_emb_size = params['char_emb_size']
        self.char_window_size = params['char_window_size']
        self.char_filter_size = params['char_filter_size']
        self.char_cnn_mask_zero = params['char_cnn_mask_zero']

    def set_char_lstm_model_params(self, params):
        logging.info('Setting char lstm params...')
        self.char_lstm_n_units = params['char_lstm_n_units']
        self.char_lstm_mask_zero = params['char_lstm_mask_zero']

    def set_test(self):
        logging.info('Setting test environment...')

        self.X_training = {'word_input': self.mwe.X_tr_word}
        self.X_test = [self.mwe.X_te_word]

        if self.model_cfg['CHAR']:
            self.X_training['char_input'] = np.array(self.mwe.X_tr_char)
            self.X_test.append(np.array(self.mwe.X_te_char))

        if self.model_cfg['POS']:
            self.X_training['pos_input'] = self.mwe.X_tr_pos
            self.X_test.append(self.mwe.X_te_pos)

        if self.model_cfg['DEPREL']:
            self.X_training['deprel_input'] = self.mwe.X_tr_deprel
            self.X_test.append(self.mwe.X_te_deprel)

        self.y = self.mwe.y

    def build_model(self):
        self.build_model_with_pretrained_embedding()

    def build_model_with_pretrained_embedding(self):
        logging.info('Building model with pretrained embedding...')
        word_emb_input = Input(shape=(None,), name='word_input')
        word_emb_layer = Embedding(input_dim=self.mwe.word_embeddings.shape[0],
                                   output_dim=self.mwe.word_embeddings.shape[1],
                                   weights=[self.mwe.word_embeddings],
                                   trainable=False, mask_zero=True, input_length=self.mwe.max_sent,
                                   name='word_embeddings')(
            word_emb_input)

        inputs = [word_emb_input]
        layers = [word_emb_layer]

        if self.model_cfg['CHAR']:
            char_embedding = []
            for char in self.mwe.chars:
                limit = math.sqrt(3.0 / self.char_emb_size)
                char_emb_vector = np.random.uniform(-limit, limit, self.char_emb_size)
                char_embedding.append(char_emb_vector)

            char_embedding[0] = np.zeros(self.char_emb_size)  # Zero padding
            char_embedding = np.asarray(char_embedding)

            if self.model_cfg['CHAR'].lower() == 'cnn':
                char_emb_input = Input(shape=(self.mwe.max_sent, self.mwe.max_char_length), dtype='int32',
                                       name='char_input')
                char_emb_layer = TimeDistributed(
                    Embedding(input_dim=char_embedding.shape[0], output_dim=char_embedding.shape[1],
                              weights=[char_embedding],
                              trainable=True, mask_zero=self.char_cnn_mask_zero), name='char_embeddings')(
                    char_emb_input)
                chars_cnn_layer = TimeDistributed(
                    Conv1D(filters=self.char_filter_size, kernel_size=self.char_window_size, padding='same'),
                    name="char_cnn")(
                    char_emb_layer)
                char_layer = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling")(chars_cnn_layer)
                inputs.append(char_emb_input)
                layers.append(char_layer)

            elif self.model_cfg['CHAR'].lower() == 'lstm':
                char_emb_input = Input(shape=(self.mwe.max_sent, self.mwe.max_char_length), dtype='int32',
                                       name='char_input')
                char_emb_layer = TimeDistributed(
                    Embedding(input_dim=char_embedding.shape[0], output_dim=char_embedding.shape[1],
                              weights=[char_embedding],
                              trainable=True, mask_zero=self.char_lstm_mask_zero), name='char_embeddings')(
                    char_emb_input)
                char_layer = TimeDistributed(Bidirectional(LSTM(self.char_lstm_n_units, return_sequences=False)),
                                             name="char_lstm")(
                    char_emb_layer)
                inputs.append(char_emb_input)
                layers.append(char_layer)

        if self.model_cfg['POS']:
            pos_embedding = np.identity(len(self.mwe.pos2idx.keys()) + 1)
            pos_emb_input = Input(shape=(None,), name='pos_input')
            pos_emb_layer = Embedding(input_dim=pos_embedding.shape[0], output_dim=pos_embedding.shape[1],
                                      weights=[pos_embedding],
                                      trainable=False, mask_zero=True, input_length=self.mwe.max_sent,
                                      name='pos_embeddings')(
                pos_emb_input)
            inputs.append(pos_emb_input)
            layers.append(pos_emb_layer)

        if self.model_cfg['DEPREL']:
            deprel_embedding = np.identity(len(self.mwe.deprel2idx.keys()) + 1)
            deprel_emb_input = Input(shape=(None,), name='deprel_input')
            deprel_emb_layer = Embedding(input_dim=deprel_embedding.shape[0], output_dim=deprel_embedding.shape[1],
                                         weights=[deprel_embedding],
                                         trainable=False, mask_zero=True, input_length=self.mwe.max_sent,
                                         name='deprel_embeddings')(
                deprel_emb_input)
            inputs.append(deprel_emb_input)
            layers.append(deprel_emb_layer)

        if len(layers) >= 2:
            merged_input = concatenate(layers)
        else:
            merged_input = layers[0]

        shared_layer = merged_input
        shared_layer = Bidirectional(LSTM(self.n_units, return_sequences=True, dropout=self._dropout[0],
                                          recurrent_dropout=self._dropout[1]),
                                     name='shared_varLSTM')(shared_layer)

        output = shared_layer
        output = TimeDistributed(Dense(self.mwe.n_tags, activation=None))(output)
        crf = CRF(self.mwe.n_tags)  # CRF layer
        output = crf(output)  # output

        model = Model(inputs=inputs, outputs=[output])
        model.compile(optimizer="nadam", loss=crf.loss_function, metrics=[crf.accuracy])
        self.model = model
        self.model.summary()

    def fit_model(self):
        logging.info('Fitting model...')
        self.model.fit(
            self.X_training,
            np.array(self.y),
            batch_size=self.batch_size, epochs=self.epochs)  # , validation_split=0.2, verbose=1)

    def predict(self):
        logging.info('Predicting...')
        predicted_tags = []
        preds = self.model.predict(self.X_test)
        for i in range(self.mwe.X_te_word.shape[0]):
            p = preds[i]
            p = np.argmax(p, axis=-1)
            tp = []
            for w, pred in zip(self.mwe.X_te_word[i], p):
                if w != 0:
                    tp.append(self.mwe.tags[pred])
            predicted_tags.append(tp)
        self.predicted_tags = predicted_tags

    def add_tags_to_test(self):
        logging.info('Tagging...')
        tags = []
        for i in range(len(self.predicted_tags)):
            for j in range(len(self.predicted_tags[i])):
                tags.append(self.predicted_tags[i][j])
            tags.append('space')
        self.mwe._test_corpus['BIO'] = tags
        self.mwe.convert_tag()
