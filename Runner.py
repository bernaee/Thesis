import os
import argparse

from src.Operations import load_pickle, dump_pickle
from src.Logger import get_logger
from src.Parameters import model_params, char_cnn_model_params, char_lstm_model_params
from src.MWEPreProcessor import MWEPreProcessor
from src.MWEIdentifier import MWEIdentifier

parser = argparse.ArgumentParser(prog='deep-bgt')
parser.add_argument('-l', '--lang')  # -l EN
parser.add_argument('-t', '--tag')  # -t IOB
parser.add_argument('-cp', '--corpus-path')
parser.add_argument('-ep', '--embeddings-path')
parser.add_argument('-op', '--output-path')
parser.add_argument('-model', '--model-name')
parser.add_argument('-exp', '--exp-no')
# -cp /home/berna/PycharmProjects/MS/Deep-BGT/data/corpora/sharedtask-data-master/1.1
# -ep /home/berna/PycharmProjects/MS/Deep-BGT/data/embeddings
# -rp /home/berna/PycharmProjects/MS/Deep-BGT/results
# -rp /home/berna/PycharmProjects/MS/Deep-BGT/results
# -model 01
# -exp 1

args = parser.parse_args()
lang = args.lang.upper()
tag = args.tag
corpus_path = args.corpus_path
embeddings_path = args.embeddings_path
output_path = args.output_path
model_name = args.model_name
exp_no = args.exp_no

input_path = os.path.join(corpus_path, lang)

word_emb = 'cc.%s.300.vec.gz' % lang.lower()
word_emb_path = os.path.join(embeddings_path, word_emb)

model_path = os.path.join(output_path, model_name)
train_output_path = os.path.join(model_path, lang)
test_output_path = os.path.join(train_output_path, exp_no)

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(train_output_path):
    os.makedirs(train_output_path)

if not os.path.exists(test_output_path):
    os.makedirs(test_output_path)

logger = get_logger(train_output_path)

# preprocessing
logger.info('Running %s %s#%s ...' % (lang, model_name, exp_no))

logger.info('Running MWE Preprocessor...')

mwe_train_path = os.path.join(train_output_path, 'train.pkl')
if not os.path.isfile(mwe_train_path):
    mwepp = MWEPreProcessor(lang, input_path, train_output_path, test_output_path)
    dump_pickle(mwepp, mwepp.train_pkl_path)
    mwepp.set_tagging(tag)
    mwepp.set_train_dev()
    mwepp.tag()
    mwepp.set_test_corpus()
    mwepp.update_test_corpus()
    mwepp.prepare_to_lstm()
    mwepp.set_fastText_word_embeddings(word_emb_path)
    dump_pickle(mwepp, mwepp.train_pkl_path)
else:
    logger.info('Loading MWE Preprocessor...')
    mwepp = load_pickle(mwe_train_path)

# mwepp.prepare_to_lstm()
# dump_pickle(mwepp, mwepp.train_pkl_path)
mwe_train_path = mwepp.train_pkl_path
mwe_test_path = mwepp.test_pkl_path
mwe_model_path = mwepp.model_pkl_path

# model
model_cfg = {'CHAR': 'lstm', 'POS': False, 'DEPREL': False}
logger.info('Running MWE Identifier...')
mwe_identifier = MWEIdentifier(lang, mwepp)
mwe_identifier.set_model_cfg(model_cfg)
mwe_identifier.set_params(model_params[lang])
mwe_identifier.set_char_cnn_model_params(char_cnn_model_params)
mwe_identifier.set_char_lstm_model_params(char_lstm_model_params)
mwe_identifier.set_test()
mwe_identifier.build_model()
mwe_identifier.fit_model()
mwe_identifier.predict()
mwe_identifier.add_tags_to_test()

logger.info('Saving model...')
mwe_identifier.model.save(mwe_model_path)
dump_pickle(mwe_identifier.mwe, mwe_test_path)

logger.info('Finish.')
