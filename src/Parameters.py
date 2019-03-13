model_params = {'BG': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 12},
                'DE': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15},
                'EL': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15},
                'EN': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 256, 'epochs': 2},#'batch_size': 8, 'epochs': 15
                'ES': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15},
                'EU': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15},
                'FA': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 12},
                'FR': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 12},
                'HE': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 12},
                'HI': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15},
                'HR': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 8, 'epochs': 15},
                'HU': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 15},
                'IT': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 12},
                'LT': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 12},
                'PL': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 12},
                'PT': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 12},
                'RO': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 12},
                'SL': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 16, 'epochs': 12},
                'TR': {'n_units': 20, 'dropout': [0.1, 0.1], 'batch_size': 32, 'epochs': 12}
                }
char_cnn_model_params = {'char_emb_size': 30, 'char_window_size': 3, 'char_filter_size': 30,
                         'char_cnn_mask_zero': False}
char_lstm_model_params = {'char_lstm_n_units': 25, 'char_lstm_mask_zero': True}