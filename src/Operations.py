import pickle


def dump_pickle(data, pkl_path):
    print('Dumping to pickle %s ...' % pkl_path)
    with open(pkl_path, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)


def load_pickle(pkl_path):
    print('Loading %s to pickle...' % pkl_path)
    return pickle.load(open(pkl_path, 'rb'))
