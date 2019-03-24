from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import *
from keras_contrib.layers import CRF
import math
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from keras.layers import Input


def build_model_with_pretrained_embedding(mwe_identifier):
    word_emb_input = Input(shape=(None,), name='word_input')
    word_emb_layer = Embedding(input_dim=mwe_identifier.mwe.word_embeddings.shape[0],
                               output_dim=mwe_identifier.mwe.word_embeddings.shape[1],
                               weights=[mwe_identifier.mwe.word_embeddings],
                               trainable=False, mask_zero=True, input_length=mwe_identifier.mwe.max_sent,
                               name='word_embeddings')(
        word_emb_input)

    inputs = [word_emb_input]
    layers = [word_emb_layer]

    if mwe_identifier.model_cfg['SPELLING']:
        spelling_emb_input = Input(shape=(None,), name='spelling_input')
        spelling_emb_layer = Embedding(input_dim=mwe_identifier.mwe.spelling_embeddings.shape[0],
                                       output_dim=mwe_identifier.mwe.spelling_embeddings.shape[1],
                                       weights=[mwe_identifier.mwe.spelling_embeddings],
                                       trainable=False, mask_zero=True, input_length=mwe_identifier.mwe.max_sent,
                                       name='spelling_embeddings')(
            spelling_emb_input)
        inputs.append(spelling_emb_input)
        layers.append(spelling_emb_layer)

    if mwe_identifier.model_cfg['CHAR']:
        char_embedding = []
        for char in mwe_identifier.mwe.chars:
            limit = math.sqrt(3.0 / mwe_identifier.char_emb_size)
            char_emb_vector = np.random.uniform(-limit, limit, mwe_identifier.char_emb_size)
            char_embedding.append(char_emb_vector)

        char_embedding[0] = np.zeros(mwe_identifier.char_emb_size)  # Zero padding
        char_embedding = np.asarray(char_embedding)

        if mwe_identifier.model_cfg['CHAR'].lower() == 'cnn':
            char_emb_input = Input(shape=(mwe_identifier.mwe.max_sent, mwe_identifier.mwe.max_char_length),
                                   dtype='int32',
                                   name='char_input')
            char_emb_layer = TimeDistributed(
                Embedding(input_dim=char_embedding.shape[0], output_dim=char_embedding.shape[1],
                          weights=[char_embedding],
                          trainable=True, mask_zero=False), name='char_embeddings')(
                char_emb_input)
            chars_cnn_layer = TimeDistributed(
                Conv1D(filters=mwe_identifier.char_filter_size, kernel_size=mwe_identifier.char_window_size,
                       padding='same'),
                name="char_cnn")(
                char_emb_layer)
            char_layer = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling")(chars_cnn_layer)
            inputs.append(char_emb_input)
            layers.append(char_layer)



        elif mwe_identifier.model_cfg['CHAR'].lower() == 'lstm':
            char_emb_input = Input(shape=(mwe_identifier.mwe.max_sent, mwe_identifier.mwe.max_char_length),
                                   dtype='int32',
                                   name='char_input')
            char_emb_layer = TimeDistributed(
                Embedding(input_dim=char_embedding.shape[0], output_dim=char_embedding.shape[1],
                          weights=[char_embedding],
                          trainable=True, mask_zero=True), name='char_embeddings')(
                char_emb_input)
            char_layer = TimeDistributed(Bidirectional(LSTM(mwe_identifier.char_lstm_n_units, return_sequences=False)),
                                         name="char_lstm")(
                char_emb_layer)
            inputs.append(char_emb_input)
            layers.append(char_layer)

    if mwe_identifier.model_cfg['POS']:
        pos_emb_input = Input(shape=(None,), name='pos_input')
        pos_emb_layer = Embedding(input_dim=mwe_identifier.mwe.pos_embeddings.shape[0],
                                  output_dim=mwe_identifier.mwe.pos_embeddings.shape[1],
                                  weights=[mwe_identifier.mwe.pos_embeddings],
                                  trainable=False, mask_zero=True, input_length=mwe_identifier.mwe.max_sent,
                                  name='pos_embeddings')(
            pos_emb_input)
        inputs.append(pos_emb_input)
        layers.append(pos_emb_layer)

    if mwe_identifier.model_cfg['DEPREL']:
        deprel_emb_input = Input(shape=(None,), name='deprel_input')
        deprel_emb_layer = Embedding(input_dim=mwe_identifier.mwe.deprel_embeddings.shape[0],
                                     output_dim=mwe_identifier.mwe.deprel_embeddings.shape[1],
                                     weights=[mwe_identifier.mwe.deprel_embeddings],
                                     trainable=False, mask_zero=True, input_length=mwe_identifier.mwe.max_sent,
                                     name='deprel_embeddings')(
            deprel_emb_input)
        inputs.append(deprel_emb_input)
        layers.append(deprel_emb_layer)

    if len(layers) >= 2:
        embedding_layer = concatenate(layers)
    else:
        embedding_layer = layers[0]

    if mwe_identifier.dropout > 0.0:
        embedding_layer = Dropout(rate={{uniform(0, 1)}})(embedding_layer)

    bilstm_layer = Bidirectional(
        LSTM(mwe_identifier.n_units, return_sequences=True, dropout=mwe_identifier.var_dropout[0],
             recurrent_dropout=mwe_identifier.var_dropout[1]),
        name='shared_varLSTM')(embedding_layer)

    output = TimeDistributed(Dense(mwe_identifier.mwe.n_tags, activation=None))(bilstm_layer)
    crf = CRF(mwe_identifier.mwe.n_tags)  # CRF layer
    output = crf(output)  # output

    model = Model(inputs=inputs, outputs=[output])
    model.compile(optimizer="nadam", loss=crf.loss_function, metrics=[crf.accuracy])

    callbacks = [EarlyStopping(monitor='loss', patience=mwe_identifier.patience)
                 ]
    result = model.fit(
        mwe_identifier.X_training,
        np.array(mwe_identifier.y),
        batch_size=mwe_identifier.batch_size, epochs=mwe_identifier.epochs, callbacks=callbacks,
        verbose=mwe_identifier.verbose)  # , validation_split=0.2, verbose=1)

    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
