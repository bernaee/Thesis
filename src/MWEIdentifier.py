from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import *
from keras_contrib.layers import CRF
import logging
import math

CI = {'ID': 0, 'FORM': 1, 'LEMMA': 2, 'UPOS': 3, 'XPOS': 4,
      'FEATS': 5, 'HEAD': 6, 'DEPREL': 7, 'DEPS': 8, 'MISC': 9,
      'BIO': -1}


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

    def set_embedding_params(self, params):
        self.set_char_cnn_model_params(params['char_trigram_cnn'])
        self.set_char_lstm_model_params(params['char_lstm'])
        self.set_morpheme_lstm_model_params(params['morpheme_lstm'])
        self.set_morpheme_char_lstm_model_params(params['morpheme_char_lstm'])

    def set_char_cnn_model_params(self, params):
        logging.info('Setting char cnn params...')
        self.char_emb_size = params['char_emb_size']
        self.char_window_size = params['char_window_size']
        self.char_filter_size = params['char_filter_size']

    def set_char_lstm_model_params(self, params):
        logging.info('Setting char lstm params...')
        self.char_emb_size = params['char_emb_size']
        self.char_lstm_n_units = params['char_lstm_n_units']

    def set_morpheme_lstm_model_params(self, params):
        logging.info('Setting morpheme lstm params...')
        self.mor_emb_size = params['mor_emb_size']
        self.mor_lstm_n_units = params['mor_lstm_n_units']

    def set_morpheme_char_lstm_model_params(self, params):
        logging.info('Setting morpheme char lstm params...')
        self.mor_char_emb_size = params['mor_char_emb_size']
        self.mor_char_lstm_n_units = params['mor_char_lstm_n_units']

    def set_model_word_embeddings(self):
        self.X_tr_word = [[self.mwe.word2idx[w[CI['FORM']]] for w in s] for s in self.mwe.train_sentences]
        self.X_tr_word = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_tr_word, padding="post", value=0)
        self.X_te_word = [[self.mwe.word2idx[w[CI['FORM']]] for w in s] for s in self.mwe.test_sentences]
        self.X_te_word = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_te_word, padding="post", value=0)

    def set_model_spelling_embeddings(self):
        self.X_tr_spelling = [[self.mwe.word2idx[w[CI['FORM']]] for w in s] for s in self.mwe.train_sentences]
        self.X_tr_spelling = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_tr_spelling, padding="post",
                                           value=0)
        self.X_te_spelling = [[self.mwe.word2idx[w[CI['FORM']]] for w in s] for s in self.mwe.test_sentences]
        self.X_te_spelling = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_te_spelling, padding="post",
                                           value=0)

    def set_model_char_embeddings(self):
        self.X_tr_char = self.mwe.create_char_matrix(self.mwe.train_sentences)
        self.X_te_char = self.mwe.create_char_matrix(self.mwe.test_sentences)

    def set_model_morpheme_embeddings(self):
        self.X_tr_morpheme = self.mwe.create_morpheme_matrix(self.mwe.train_sentences)
        self.X_te_morpheme = self.mwe.create_morpheme_matrix(self.mwe.test_sentences)

    def set_model_morpheme_wor_char_embeddings(self):
        self.X_tr_morpheme_wor_char = self.mwe.create_morphe_wor_char_matrix(self.mwe.train_sentences)
        self.X_te_morpheme_wor_char = self.mwe.create_morphe_wor_char_matrix(self.mwe.test_sentences)

    def set_model_morpheme_wr_char_embeddings(self):
        self.X_tr_morpheme_wr_char = self.mwe.create_morphe_wr_char_matrix(self.mwe.train_sentences)
        self.X_te_morpheme_wr_char = self.mwe.create_morphe_wr_char_matrix(self.mwe.test_sentences)

    def set_model_pos_embeddings(self):
        self.X_tr_pos = [[self.mwe.pos2idx[w[CI['UPOS']]] for w in s] for s in self.mwe.train_sentences]
        self.X_tr_pos = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_tr_pos, padding="post", value=0)
        self.X_te_pos = [[self.mwe.pos2idx[w[CI['UPOS']]] for w in s] for s in self.mwe.test_sentences]
        self.X_te_pos = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_te_pos, padding="post", value=0)

    def set_model_deprel_embeddings(self):
        self.X_tr_deprel = [[self.mwe.deprel2idx[w[CI['DEPREL']]] for w in s] for s in self.mwe.train_sentences]
        self.X_tr_deprel = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_tr_deprel, padding="post", value=0)
        self.X_te_deprel = [[self.mwe.deprel2idx[w[CI['DEPREL']]] for w in s] for s in self.mwe.test_sentences]
        self.X_te_deprel = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.X_te_deprel, padding="post", value=0)

    def set_crf_features(self):
        self.X_tr_crf = self.mwe.create_crf_matrix(self.mwe._train_corpus)
        self.X_te_crf = self.mwe.create_crf_matrix(self.mwe._test_corpus)

    def set_model_tags(self):
        self.y = [[self.mwe.tag2idx[w[CI['BIO']]] for w in s] for s in self.mwe.train_sentences]
        self.y = pad_sequences(maxlen=self.mwe.max_sent, sequences=self.y, padding="post", value=self.mwe.tag2idx["O"])
        self.y = [to_categorical(i, num_classes=self.mwe.n_tags) for i in self.y]

    def set_test(self):
        logging.info('Setting test environment...')
        self.set_model_word_embeddings()
        self.X_training = {'word_input': self.X_tr_word}
        self.X_test = [self.X_te_word]

        if self.model_cfg['SPELLING']:
            self.set_model_spelling_embeddings()
            self.X_training['spelling_input'] = self.X_tr_spelling
            self.X_test.append(self.X_te_spelling)

        if self.model_cfg['CHAR']:
            self.set_model_char_embeddings()
            self.X_training['char_input'] = self.X_tr_char
            self.X_test.append(self.X_te_char)

        if self.model_cfg['POS']:
            self.set_model_pos_embeddings()
            self.X_training['pos_input'] = self.X_tr_pos
            self.X_test.append(self.X_te_pos)

        if self.model_cfg['DEPREL']:
            self.set_model_deprel_embeddings()
            self.X_training['deprel_input'] = self.X_tr_deprel
            self.X_test.append(self.X_te_deprel)

        if self.model_cfg['MORPHEME']:
            self.set_model_morpheme_embeddings()
            self.X_training['morpheme_input'] = self.X_tr_morpheme
            self.X_test.append(self.X_te_morpheme)

        if self.model_cfg['MORPHEME-CHAR']:
            if self.model_cfg['MORPHEME-CHAR'] == 'wor':
                self.set_model_morpheme_wor_char_embeddings()
                self.X_training['morpheme_wor_char_input'] = self.X_tr_morpheme_wor_char
                self.X_test.append(self.X_te_morpheme_wor_char)
            elif self.model_cfg['MORPHEME-CHAR'] == 'wr':
                self.set_model_morpheme_wr_char_embeddings()
                self.X_training['morpheme_wr_char_input'] = self.X_tr_morpheme_wr_char
                self.X_test.append(self.X_te_morpheme_wr_char)

        # if self.model_cfg['CRF']:
        #     self.set_crf_features()
        #     self.X_training['crf_form'] = self.X_tr_crf['form']
        #     self.X_training['crf_prev_form'] = self.X_tr_crf['prev_form']
        #     self.X_training['crf_next_form'] = self.X_tr_crf['next_form']
        #     self.X_test.append(self.X_te_crf['form'])
        #     self.X_test.append(self.X_te_crf['prev_form'])
        #     self.X_test.append(self.X_te_crf['next_form'])

        self.set_model_tags()
        self.y = self.y

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
            mor_embedding = []
            for mor in self.mwe.morphemes:
                limit = math.sqrt(3.0 / self.mor_emb_size)
                mor_emb_vector = np.random.uniform(-limit, limit, self.mor_emb_size)
                mor_embedding.append(mor_emb_vector)

            mor_embedding[0] = np.zeros(self.mor_emb_size)  # Zero padding
            mor_embedding = np.asarray(mor_embedding)

            if self.model_cfg['MORPHEME'].lower() == 'lstm':
                mor_emb_input = Input(shape=(self.mwe.max_sent, self.mwe.max_morpheme_len), dtype='int32',
                                      name='morpheme_input')
                mor_emb_layer = TimeDistributed(
                    Embedding(input_dim=mor_embedding.shape[0], output_dim=mor_embedding.shape[1],
                              weights=[mor_embedding],
                              trainable=True, mask_zero=True), name='morp_embeddings')(
                    mor_emb_input)
                mor_layer = TimeDistributed(Bidirectional(LSTM(self.mor_lstm_n_units, return_sequences=False)),
                                            name="morp_lstm")(
                    mor_emb_layer)
                inputs.append(mor_emb_input)
                layers.append(mor_layer)

        if self.model_cfg['MORPHEME-CHAR']:
            if self.model_cfg['MORPHEME-CHAR'] == 'wor':
                mor_wor_char_embedding = []
                for mor_char in self.mwe.morpheme_wor_chars:
                    limit = math.sqrt(3.0 / self.mor_char_emb_size)
                    morpheme_wor_char_emb_vector = np.random.uniform(-limit, limit, self.mor_char_emb_size)
                    mor_wor_char_embedding.append(morpheme_wor_char_emb_vector)

                mor_wor_char_embedding[0] = np.zeros(self.mor_char_emb_size)  # Zero padding
                mor_wor_char_embedding = np.asarray(mor_wor_char_embedding)

                mor_wor_char_emb_input = Input(shape=(self.mwe.max_sent, self.mwe.max_morpheme_wor_char_length),
                                               dtype='int32',
                                               name='morpheme_wor_char_input')
                mor_wor_char_emb_layer = TimeDistributed(
                    Embedding(input_dim=mor_wor_char_embedding.shape[0], output_dim=mor_wor_char_embedding.shape[1],
                              weights=[mor_wor_char_embedding],
                              trainable=True, mask_zero=True), name='mor_wor_char_embeddings')(
                    mor_wor_char_emb_input)
                mor_wor_char_layer = TimeDistributed(
                    Bidirectional(LSTM(self.mor_char_lstm_n_units, return_sequences=False)),
                    name="mor_wor_char_lstm")(
                    mor_wor_char_emb_layer)
                inputs.append(mor_wor_char_emb_input)
                layers.append(mor_wor_char_layer)

            if self.model_cfg['MORPHEME-CHAR'] == 'wr':
                mor_wr_char_embedding = []
                for mor_char in self.mwe.morpheme_wr_chars:
                    limit = math.sqrt(3.0 / self.mor_char_emb_size)
                    morpheme_wr_char_emb_vector = np.random.uniform(-limit, limit, self.mor_char_emb_size)
                    mor_wr_char_embedding.append(morpheme_wr_char_emb_vector)

                mor_wr_char_embedding[0] = np.zeros(self.mor_char_emb_size)  # Zero padding
                mor_wr_char_embedding = np.asarray(mor_wr_char_embedding)

                mor_wr_char_emb_input = Input(shape=(self.mwe.max_sent, self.mwe.max_morpheme_wr_char_length),
                                              dtype='int32',
                                              name='morpheme_wr_char_input')
                mor_wr_char_emb_layer = TimeDistributed(
                    Embedding(input_dim=mor_wr_char_embedding.shape[0], output_dim=mor_wr_char_embedding.shape[1],
                              weights=[mor_wr_char_embedding],
                              trainable=True, mask_zero=True), name='mor_wr_char_embeddings')(
                    mor_wr_char_emb_input)
                mor_wr_char_layer = TimeDistributed(
                    Bidirectional(LSTM(self.mor_char_lstm_n_units, return_sequences=False)),
                    name="mor_wr_char_lstm")(
                    mor_wr_char_emb_layer)
                inputs.append(mor_wr_char_emb_input)
                layers.append(mor_wr_char_layer)

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

        bilstm_output = TimeDistributed(Dense(self.mwe.n_tags, activation=None), name='time_distributed')(bilstm_layer)

        # if self.model_cfg['CRF']:
        #     crf_form_input = Input(shape=(None,), name='crf_form')
        #     crf_form_layer = Embedding(input_dim=self.mwe.n_words,
        #                                output_dim=1,
        #                                trainable=True, mask_zero=True, input_length=self.mwe.max_sent,
        #                                name='crf_form_embeddings')(
        #         crf_form_input)
        #     crf_prev_form_input = Input(shape=(None,), name='crf_prev_form')
        #     crf_prev_form_layer = Embedding(input_dim=self.mwe.n_words,
        #                                     output_dim=1,
        #                                     trainable=True, mask_zero=True, input_length=self.mwe.max_sent,
        #                                     name='crf_prev_form_embeddings')(
        #         crf_prev_form_input)
        #     crf_next_form_input = Input(shape=(None,), name='crf_next_form')
        #     crf_next_form_layer = Embedding(input_dim=self.mwe.n_words,
        #                                     output_dim=1,
        #                                     trainable=True, mask_zero=True, input_length=self.mwe.max_sent,
        #                                     name='crf_next_form_embeddings')(
        #         crf_next_form_input)
        #     inputs.append(crf_form_input)
        #     inputs.append(crf_prev_form_input)
        #     inputs.append(crf_next_form_input)
        #     bilstm_output = concatenate([bilstm_output, crf_form_layer, crf_prev_form_layer, crf_next_form_layer])

        crf = CRF(self.mwe.n_tags)  # CRF layer
        output = crf(bilstm_output)  # output

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
        for i in range(self.X_te_word.shape[0]):
            p = preds[i]
            p = np.argmax(p, axis=-1)
            tp = []
            for w, pred in zip(self.X_te_word[i], p):
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
