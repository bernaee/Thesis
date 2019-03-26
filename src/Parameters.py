model_params = {
    'BG': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 15, 'patience': 5,
           'verbose': 2},
    'DE': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 20, 'patience': 5,
           'verbose': 2},
    'EL': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 20, 'patience': 5,
           'verbose': 2},
    'EN': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 8, 'epochs': 20, 'patience': 5,
           'verbose': 2},
    'ES': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 20, 'patience': 5,
           'verbose': 2},
    'EU': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15, 'patience': 5,
           'verbose': 2},
    'FA': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15, 'patience': 5,
           'verbose': 2},
    'FR': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 15, 'patience': 5,
           'verbose': 2},
    'HE': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 15, 'patience': 5,
           'verbose': 2},
    'HI': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 20, 'patience': 5,
           'verbose': 2},
    'HR': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 8, 'epochs': 20, 'patience': 5,
           'verbose': 2},
    'HU': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 20, 'patience': 5,
           'verbose': 2},
    'IT': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15, 'patience': 5,
           'verbose': 2},
    'LT': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 15, 'patience': 5,
           'verbose': 2},
    'PL': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15, 'patience': 5,
           'verbose': 2},
    'PT': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 15, 'patience': 5,
           'verbose': 2},
    'RO': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 64, 'epochs': 15, 'patience': 5,
           'verbose': 2},
    'SL': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15, 'patience': 5,
           'verbose': 2},
    'TR': {'n_units': 20, 'dropout': 0.5, 'var_dropout': [0.1, 0.1], 'batch_size': 64, 'epochs': 15, 'patience': 5,
           'verbose': 2}
}
char_trigram_cnn_model_params = {'char_emb_size': 30, 'char_window_size': 3, 'char_filter_size': 30}
morpheme_trigram_cnn_model_params = {'mopr_emb_size': 30, 'morp_window_size': 3, 'morp_filter_size': 30}
char_lstm_model_params = {'char_emb_size': 30, 'char_lstm_n_units': 25}
morpheme_lstm_model_params = {'morp_emb_size': 30, 'morp_lstm_n_units': 25}

model_cfg = {
    '01': {'SPELLING': False, 'CHAR': False, 'POS': False, 'DEPREL': False, 'DROPOUT': False, 'MORPHEME': False},
    '02': {'SPELLING': True, 'CHAR': False, 'POS': False, 'DEPREL': False, 'DROPOUT': False, 'MORPHEME': False},
    '03': {'SPELLING': False, 'CHAR': 'cnn', 'POS': False, 'DEPREL': False, 'DROPOUT': False, 'MORPHEME': False},
    '04': {'SPELLING': False, 'CHAR': 'cnn', 'POS': False, 'DEPREL': False, 'DROPOUT': True, 'MORPHEME': False},
    '05': {'SPELLING': False, 'CHAR': 'lstm', 'POS': False, 'DEPREL': False, 'DROPOUT': False, 'MORPHEME': False},
    '06': {'SPELLING': False, 'CHAR': 'lstm', 'POS': False, 'DEPREL': False, 'DROPOUT': True, 'MORPHEME': False}, }
