from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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
        self.dropout = params['dropout']
        self.var_dropout = params['var_dropout']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.patience = params['patience']
        self.verbose = params['verbose']

    def set_char_cnn_model_params(self, params):
        logging.info('Setting char cnn params...')
        self.char_emb_size = params['char_emb_size']
        self.char_window_size = params['char_window_size']
        self.char_filter_size = params['char_filter_size']

    def set_char_lstm_model_params(self, params):
        logging.info('Setting char lstm params...')
        self.char_emb_size = params['char_emb_size']
        self.char_lstm_n_units = params['char_lstm_n_units']

    def set_morpheme_cnn_model_params(self, params):
        logging.info('Setting morpheme cnn params...')
        self.morp_emb_size = params['morp_emb_size']
        self.morp_window_size = params['morp_window_size']
        self.morp_filter_size = params['morp_filter_size']

    def set_morpheme_lstm_model_params(self, params):
        logging.info('Setting morpheme lstm params...')
        self.morp_emb_size = params['morp_emb_size']
        self.morp_lstm_n_units = params['morp_lstm_n_units']

    def set_test(self):
        logging.info('Setting test environment...')
        self.mwe.set_model_word_embeddings(self)
        self.X_training = {'word_input': self.mwe.X_tr_word}
        self.X_test = [self.mwe.X_te_word]

        if self.model_cfg['SPELLING']:
            self.mwe.set_model_spelling_embeddings(self)
            self.X_training['spelling_input'] = self.mwe.X_tr_spelling
            self.X_test.append(self.mwe.X_te_spelling)

        if self.model_cfg['CHAR']:
            self.mwe.set_model_char_embeddings(self)
            self.X_training['char_input'] = self.mwe.X_tr_char
            self.X_test.append(self.mwe.X_te_char)

        if self.model_cfg['POS']:
            self.mwe.set_model_pos_embeddings(self)
            self.X_training['pos_input'] = self.mwe.X_tr_pos
            self.X_test.append(self.mwe.X_te_pos)

        if self.model_cfg['DEPREL']:
            self.mwe.set_model_deprel_embeddings(self)
            self.X_training['deprel_input'] = self.mwe.X_tr_deprel
            self.X_test.append(self.mwe.X_te_deprel)

        if self.model_cfg['MORPHEME']:
            self.mwe.set_model_morpheme_embeddings(self)
            self.X_training['morpheme_input'] = self.mwe.X_tr_morpheme
            self.X_test.append(self.mwe.X_te_morpheme)

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

        if self.model_cfg['SPELLING']:
            spelling_emb_input = Input(shape=(None,), name='spelling_input')
            spelling_emb_layer = Embedding(input_dim=self.mwe.spelling_embeddings.shape[0],
                                           output_dim=self.mwe.spelling_embeddings.shape[1],
                                           weights=[self.mwe.spelling_embeddings],
                                           trainable=False, mask_zero=True, input_length=self.mwe.max_sent,
                                           name='spelling_embeddings')(
                spelling_emb_input)
            inputs.append(spelling_emb_input)
            layers.append(spelling_emb_layer)

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
                              trainable=True, mask_zero=False), name='char_embeddings')(
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
                              trainable=True, mask_zero=True), name='char_embeddings')(
                    char_emb_input)
                char_layer = TimeDistributed(Bidirectional(LSTM(self.char_lstm_n_units, return_sequences=False)),
                                             name="char_lstm")(
                    char_emb_layer)
                inputs.append(char_emb_input)
                layers.append(char_layer)

        if self.model_cfg['MORPHEME']:
            morp_embedding = []
            for morp in self.mwe.morphemes:
                limit = math.sqrt(3.0 / self.morp_emb_size)
                morp_emb_vector = np.random.uniform(-limit, limit, self.morp_emb_size)
                morp_embedding.append(morp_emb_vector)

            morp_embedding[0] = np.zeros(self.morp_emb_size)  # Zero padding
            morp_embedding = np.asarray(morp_embedding)

            if self.model_cfg['MORPHEME'].lower() == 'cnn':
                morp_emb_input = Input(shape=(self.mwe.max_sent, self.mwe.max_morpheme_len), dtype='int32',
                                       name='morpheme_input')
                morp_emb_layer = TimeDistributed(
                    Embedding(input_dim=morp_embedding.shape[0], output_dim=morp_embedding.shape[1],
                              weights=[morp_embedding],
                              trainable=True, mask_zero=False), name='morp_embeddings')(
                    morp_emb_input)
                morp_cnn_layer = TimeDistributed(
                    Conv1D(filters=self.char_filter_size, kernel_size=self.morp_window_size, padding='same'),
                    name="morp_cnn")(
                    morp_emb_layer)
                morp_layer = TimeDistributed(GlobalMaxPooling1D(), name="morp_pooling")(morp_cnn_layer)
                inputs.append(morp_emb_input)
                layers.append(morp_layer)



            elif self.model_cfg['MORPHEME'].lower() == 'lstm':
                morp_emb_input = Input(shape=(self.mwe.max_sent, self.mwe.max_morpheme_len), dtype='int32',
                                       name='morpheme_input')
                morp_emb_layer = TimeDistributed(
                    Embedding(input_dim=morp_embedding.shape[0], output_dim=morp_embedding.shape[1],
                              weights=[morp_embedding],
                              trainable=True, mask_zero=True), name='morp_embeddings')(
                    morp_emb_input)
                morp_layer = TimeDistributed(Bidirectional(LSTM(self.morp_lstm_n_units, return_sequences=False)),
                                             name="morp_lstm")(
                    morp_emb_layer)
                inputs.append(morp_emb_input)
                layers.append(morp_layer)

        if self.model_cfg['POS']:
            pos_emb_input = Input(shape=(None,), name='pos_input')
            pos_emb_layer = Embedding(input_dim=self.mwe.pos_embeddings.shape[0],
                                      output_dim=self.mwe.pos_embeddings.shape[1],
                                      weights=[self.mwe.pos_embeddings],
                                      trainable=False, mask_zero=True, input_length=self.mwe.max_sent,
                                      name='pos_embeddings')(
                pos_emb_input)
            inputs.append(pos_emb_input)
            layers.append(pos_emb_layer)

        if self.model_cfg['DEPREL']:
            deprel_emb_input = Input(shape=(None,), name='deprel_input')
            deprel_emb_layer = Embedding(input_dim=self.mwe.deprel_embeddings.shape[0],
                                         output_dim=self.mwe.deprel_embeddings.shape[1],
                                         weights=[self.mwe.deprel_embeddings],
                                         trainable=False, mask_zero=True, input_length=self.mwe.max_sent,
                                         name='deprel_embeddings')(
                deprel_emb_input)
            inputs.append(deprel_emb_input)
            layers.append(deprel_emb_layer)

        if len(layers) >= 2:
            embedding_layer = concatenate(layers)
        else:
            embedding_layer = layers[0]

        if self.model_cfg['DROPOUT']:
            embedding_layer = Dropout(self.dropout)(embedding_layer)

        bilstm_layer = Bidirectional(
            LSTM(self.n_units, return_sequences=True, dropout=self.var_dropout[0],
                 recurrent_dropout=self.var_dropout[1]),
            name='shared_varLSTM')(embedding_layer)

        output = TimeDistributed(Dense(self.mwe.n_tags, activation=None))(bilstm_layer)
        crf = CRF(self.mwe.n_tags)  # CRF layer
        output = crf(output)  # output

        model = Model(inputs=inputs, outputs=[output])
        model.compile(optimizer="nadam", loss=crf.loss_function, metrics=[crf.accuracy])
        self.model = model
        self.model.summary()

    def fit_model(self):
        logging.info('Fitting model...')
        callbacks = [EarlyStopping(monitor='loss', patience=self.patience)
                     ]
        self.model.fit(
            self.X_training,
            np.array(self.y),
            batch_size=self.batch_size, epochs=self.epochs, callbacks=callbacks,
            verbose=self.verbose)  # , validation_split=0.2, verbose=1)

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
