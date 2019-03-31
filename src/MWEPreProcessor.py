import re
import copy
import io
import os
import codecs
import logging
import pandas as pd
import numpy as np

from src.WordEmbedding import read_fastText_word_embeddings

CI = {'ID': 0, 'FORM': 1, 'LEMMA': 2, 'UPOS': 3, 'XPOS': 4,
      'FEATS': 5, 'HEAD': 6, 'DEPREL': 7, 'DEPS': 8, 'MISC': 9,
      'BIO': -1}


class MWEPreProcessor:
    def __init__(self, language, input_path, train_output_path, test_output_path):
        logging.info('Initialize MWEPreprocessor for %s' % language)
        self.language = language
        self.input_path = input_path
        self.train_output_path = train_output_path
        self.test_output_path = test_output_path
        self.set_paths()

    def set_tagging(self, tagging):
        self.tagging = tagging

    def set_paths(self):
        self.train_path = os.path.join(self.input_path, 'train.cupt')
        self.dev_path = os.path.join(self.input_path, 'dev.cupt')
        self.test_blind_path = os.path.join(self.input_path, 'test.blind.cupt')
        self.test_gold_path = os.path.join(self.test_output_path, 'test.cupt')
        self.train_pkl_path = os.path.join(self.train_output_path, 'train.pkl')
        self.train_tagged_path = os.path.join(self.train_output_path, 'train_tagged.cupt')
        self.test_pkl_path = os.path.join(self.test_output_path, 'test.pkl')
        self.model_pkl_path = os.path.join(self.test_output_path, 'model.pkl')
        self.test_tagged_path = os.path.join(self.test_output_path, 'test_tagged.cupt')

    def update_test_output_path(self, test_output_path):
        self.test_output_path = test_output_path
        self.test_pkl_path = os.path.join(self.test_output_path, 'test.pkl')
        self.model_pkl_path = os.path.join(self.test_output_path, 'model.pkl')
        self.test_tagged_path = os.path.join(self.test_output_path, 'test_tagged.cupt')

    def read_corpus(self, path):
        corpus_file = io.open(path, "r", encoding="utf-8")
        corpus = []
        for s in corpus_file:
            if not s.startswith('#'):
                corpus.append(s)
        corpus = [x.split('\t') for x in corpus]
        new_corpus = pd.DataFrame(corpus,
                                  columns=['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL',
                                           'DEPS', 'MISC',
                                           'PARSEME:MWE'])

        return new_corpus

    def remove_duplicate_rows(self, corpus):
        corpus['ID_tag'] = copy.deepcopy(corpus['ID'].apply(
            lambda x: 'extra' if bool(re.match("\d+-\d+", x)) else x))
        if self.language == 'TR':
            corpus.loc[corpus['FORM'] == '_', 'ID_tag'] = 'extra'
        additional_rows = corpus.loc[corpus['ID_tag'] == 'extra']
        corpus = corpus.loc[corpus['ID_tag'] != 'extra']
        return corpus, additional_rows

    def to_cupt(self, df, cupt_path):
        logging.info('Writing to %s...' % cupt_path)
        lines = ''
        for idx, row in df.iterrows():
            if row['PARSEME:MWE'] == 'space':
                line = '\n'
            else:
                line = str(row['ID']) + '\t' + row['FORM'] + '\t' + row['LEMMA'] + '\t' + row['UPOS'] + '\t' + row[
                    'XPOS'] + '\t' + row['FEATS'] + '\t' + row['HEAD'] + '\t' + row['DEPREL'] + '\t' + row[
                           'DEPS'] + '\t' + row['MISC'] + '\t' + row['BIO'] + '\n'
            lines += line

        f = codecs.open(cupt_path, "w", "utf-8")
        f.write(lines)
        f.close()

    def find_comments_in_cupt(self, path):
        cupt_file = io.open(path, "r", encoding="utf-8")
        comments = []
        isSpaceAdded = False
        for s in cupt_file:
            if s.startswith('#'):
                comments.append(s)
                isSpaceAdded = False
            elif not isSpaceAdded:
                comments.append(" ")
                isSpaceAdded = True
        return comments

    def to_cupt_with_comments(self):
        comments = self.find_comments_in_cupt(self.test_blind_path)
        logging.info('Writing to %s...' % self.test_tagged_path)
        lines = ''
        counterC = 0
        while not comments[counterC] == " ":
            line = comments[counterC]
            lines += line
            counterC = counterC + 1
        counterC = counterC + 1
        for idx, row in self._test_corpus.iterrows():
            if row['PARSEME:MWE'] == 'space':
                line = '\n'
                while counterC < len(comments) and not comments[counterC] == " ":
                    line = line + comments[counterC]
                    counterC = counterC + 1
                if comments[counterC] == " ":
                    if counterC + 1 < len(comments):
                        counterC = counterC + 1
            else:
                line = str(row['ID']) + '\t' + row['FORM'] + '\t' + row['LEMMA'] + '\t' + row['UPOS'] + '\t' + row[
                    'XPOS'] + '\t' + row['FEATS'] + '\t' + row['HEAD'] + '\t' + row['DEPREL'] + '\t' + row[
                           'DEPS'] + '\t' + row['MISC'] + '\t' + row['BIO'] + '\n'
            lines += line
        f = codecs.open(self.test_tagged_path, "w", "utf-8")
        f.write(lines)
        f.close()

    def read_sentences(self, corpus):
        sentence_indexes = [-1] + list(corpus.loc[corpus['BIO'] == 'space'].index)
        sentences = []
        for i, sentence_idx in enumerate(sentence_indexes[:-1]):
            sentence = corpus.loc[sentence_idx + 1:sentence_indexes[i + 1]].iloc[:-1]
            tr_sentence = []
            for j, row in sentence.iterrows():
                tr_sentence.append((row['ID'], row['FORM'], row['LEMMA'], row['UPOS'], row['XPOS'], row['FEATS'],
                                    row['HEAD'], row['DEPREL'], row['DEPS'], row['MISC'], row['BIO']))
            sentences.append(tr_sentence)
        return sentences

    def update_test_corpus(self):
        self._test_corpus['BIO'] = copy.deepcopy(self._test_corpus['PARSEME:MWE'])
        self._test_corpus[self._test_corpus['BIO'].isnull()] = 'space'
        self._test_corpus['BIO'] = copy.deepcopy(self._test_corpus['BIO'].apply(lambda x: x.strip()))
        return self._test_corpus

    def set_train_dev(self):
        train_corpus = self.read_corpus(self.train_path)
        if not os.path.isfile(self.dev_path):
            logging.info('Dev set is not found.')
            dev_corpus = pd.DataFrame()
        else:
            dev_corpus = self.read_corpus(self.dev_path)
        corpus = pd.concat([train_corpus, dev_corpus], ignore_index=True)
        self._train_corpus = corpus

    def set_test_corpus(self):
        self._test_corpus = self.read_corpus(self.test_blind_path)

    def preprocess_corpus(self):
        self._train_corpus, self._train_rows = self.remove_duplicate_rows(self._train_corpus)
        self._test_corpus, self._test_rows = self.remove_duplicate_rows(self._test_corpus)

    def set_word_embeddings(self, word_embeddings):
        self.word_embeddings = word_embeddings

    def set_fastText_word_embeddings(self, we_path):
        logging.info('Reading fastText word embeddings...')
        embedding_matrix, unk_word_vector = read_fastText_word_embeddings(we_path, self.words, self.n_words)
        self.set_word_embeddings(embedding_matrix)
        self.unk_word_vector = unk_word_vector

    def tag(self):
        if self.tagging == 'IOB':
            self.tag_IOB()
        elif self.tagging == 'gappy-1':
            self.tag_gappy_1_level()
        elif self.tagging == 'gappy-crossy':
            self.tag_bigappy_unicrossy()

    def convert_tag(self):
        if self.tagging == 'IOB':
            self.convert_IOB()
        elif self.tagging == 'gappy-1':
            self.convert_gappy_tag()
        elif self.tagging == 'gappy-crossy':
            self.convert_gappy_tag()

    def tag_bigappy_unicrossy(self):
        self._train_corpus['BIO'] = copy.deepcopy(self._train_corpus['PARSEME:MWE'])
        self._train_corpus[self._train_corpus['BIO'].isnull()] = 'space'

        # remove other tags after the first tag
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: x.split(';')[0] if bool(re.match("\d:\w+[.]*\w+;.", x)) else x)
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: x.split(';')[0] if bool(re.match("\d;.", x)) else x)
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(lambda x: x.strip())

        sentence_indexes = [-1] + list(self._train_corpus.loc[self._train_corpus['BIO'] == 'space'].index)

        # tag sentence by sentence
        for i, sentence_idx in enumerate(sentence_indexes[:-1]):
            sentence = self._train_corpus[sentence_idx + 1:sentence_indexes[i + 1]]
            o_indexes = []
            last_B_idx = 0
            last_b_idx = 0
            isB = False
            # j -> each token in the sentence
            # only 1 B(I) and 1 b(i) is taken into consideration simultaneously
            # allows crossings
            for j, row in sentence.iterrows():
                tag = sentence.loc[j, 'BIO']
                # if (there is a tag in the form of no:category and it is the first VMWE in the sentence)
                # or (there is a tag in the form of no:category
                # and it is the beginning of an VMWE after the last B(I) tagged VMWE ends)
                # it does not wait the end of the nested VMWE to begin a new B(I) tagged VMWE
                # B I b i I I B i I is possible
                # only 1 B is allowed - 1 level
                if (bool(re.match("\d:", tag)) and not isB) or (bool(re.match("\d:", tag)) and j > last_B_idx):
                    isB = True
                    no = tag.split(':')[0]
                    category = tag.split(':')[1]
                    category = category.strip()
                    self._train_corpus.loc[j, 'BIO'] = 'B:' + category
                    # stores I indexes
                    I_indexes = list(sentence.loc[self._train_corpus['BIO'] == no].index)
                    # if it is multi-token VMWE
                    if len(I_indexes) > 0:
                        for k in I_indexes:
                            self._train_corpus.loc[k, 'BIO'] = 'I:' + category
                        last_B_idx = I_indexes[-1]
                        # it is multi-token VMWE, so there is a possibility of gap
                        # add the beginning index and the end index of the VMWE
                        o_indexes.append([j, last_B_idx])
                    # if it is single-token VMWE
                    else:
                        last_B_idx = j

                # if (there is a tag in the form of no:category and
                # it is a nested VMWE since its in between the last BI tagged VMWE)
                # and it is the first nested VMWE in between the last BI tagged VMWE
                # only 1 b is allowed - 1 level
                # since the location of i is not checked, it allows crossing
                elif (bool(re.match("\d:", tag)) and j < last_B_idx) and j > last_b_idx:
                    no = tag.split(':')[0]
                    category = tag.split(':')[1]
                    category = category.strip()
                    self._train_corpus.loc[j, 'BIO'] = 'b:' + category
                    # stores i indexes
                    i_indexes = list(sentence.loc[self._train_corpus['BIO'] == no].index)
                    # if it is multi-token VMWE
                    if len(i_indexes) > 0:
                        for l in i_indexes:
                            self._train_corpus.loc[l, 'BIO'] = 'i:' + category
                        last_b_idx = i_indexes[-1]
                        # it is multi-token VMWE, so there is a possibility of gap
                        # add the beginning index and the end index of the VMWE
                        o_indexes.append([j, last_b_idx])
                    # if it is single-token VMWE
                    else:
                        last_b_idx = j

            for o_idx in o_indexes:
                for oo_idx in range(o_idx[0] + 1, o_idx[1]):
                    tag = sentence.loc[oo_idx, 'BIO']
                    # if there is no tag in between an multi-token MWE, tag with 'o'
                    if not (bool(re.match("I:", tag)) or bool(re.match("B:", tag)) or bool(re.match("i:", tag)) or bool(
                            re.match("b:", tag))):
                        self._train_corpus.loc[oo_idx, 'BIO'] = 'o'

        # tags with 'O'
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: 'O' if not (bool(re.match("i:", x)) or bool(re.match("b:", x)) or bool(re.match("I:", x)) or bool(
                re.match("B:", x)) or bool(re.match("o", x)) or bool(re.match("space", x))) else x)

        # self._train_corpus.to_csv(fileName)
        self.to_cupt(self._train_corpus, self.train_tagged_path)

    def tag_gappy_1_level(self):
        self._train_corpus['BIO'] = copy.deepcopy(self._train_corpus['PARSEME:MWE'])
        self._train_corpus[self._train_corpus['BIO'].isnull()] = 'space'

        # remove other tags after the first tag
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: x.split(';')[0] if bool(re.match("\d:\w+[.]*\w+;.", x)) else x)
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: x.split(';')[0] if bool(re.match("\d;.", x)) else x)
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(lambda x: x.strip())

        sentence_indexes = [-1] + list(self._train_corpus.loc[self._train_corpus['BIO'] == 'space'].index)

        # tag sentence by sentence
        for i, sentence_idx in enumerate(sentence_indexes[:-1]):
            sentence = self._train_corpus[sentence_idx + 1:sentence_indexes[i + 1]]
            o_indexes = []
            last_B_idx = 0
            last_b_idx = 0
            isB = False
            # j -> each token in the sentence
            # only 1 B(I) and 1 b(i) is taken into consideration simultaneously
            # does not allow crossings
            for j, row in sentence.iterrows():
                tag = sentence.loc[j, 'BIO']
                # if (there is a tag in the form of no:category and it is the first VMWE in the sentence)
                # or (there is a tag in the form of no:category
                # and it is the beginning of an VMWE after the last B(I) tagged VMWE ends)
                # only 1 B is allowed - 1 level
                if (bool(re.match("\d:", tag)) and not isB) or (bool(re.match("\d:", tag)) and j > last_B_idx):
                    isB = True
                    no = tag.split(':')[0]
                    category = tag.split(':')[1]
                    category = category.strip()
                    self._train_corpus.loc[j, 'BIO'] = 'B:' + category
                    # stores I indexes
                    I_indexes = list(sentence.loc[self._train_corpus['BIO'] == no].index)
                    # if it is multi-token VMWE
                    if len(I_indexes) > 0:
                        for k in I_indexes:
                            self._train_corpus.loc[k, 'BIO'] = 'I:' + category
                        last_B_idx = I_indexes[-1]
                        # it is multi-token VMWE, so there is a possibility of gap
                        # add the beginning index and the end index of the VMWE
                        o_indexes.append([j, last_B_idx])
                    # if it is single-token VMWE
                    else:
                        last_B_idx = j

                # if (there is a tag in the form of no:category and
                # it is a nested VMWE since its in between the last BI tagged VMWE)
                # and it is the first nested VMWE in between the last BI tagged VMWE
                # only 1 b is allowed - 1 level
                # since the location of i is checked, it does not allow crossing
                elif (bool(re.match("\d:", tag)) and j < last_B_idx) and j > last_b_idx:
                    prev_last_b_idx = last_b_idx
                    no = tag.split(':')[0]
                    category = tag.split(':')[1]
                    category = category.strip()
                    # stores i indexes
                    i_indexes = list(sentence.loc[self._train_corpus['BIO'] == no].index)
                    # if it is multi-token VMWE
                    if len(i_indexes) > 0:
                        valid_b = 0
                        last_b_idx = i_indexes[-1]
                        if last_b_idx < last_B_idx:
                            valid_b = 1
                            if not i_indexes[0] - j == 1:
                                valid_b = 0
                            for i_idx in range(1, len(i_indexes)):
                                if not i_indexes[i_idx] - i_indexes[i_idx - 1] == 1:
                                    valid_b = 0
                            if valid_b == 1:
                                for n_idx in range(j + 1, last_b_idx):
                                    tagg = sentence.loc[n_idx, 'BIO']
                                    # if there is no tag in between an nested MWE, tag with nested MWE
                                    # if there is a tag, invalid nested MWE
                                    if bool(re.match("I:", tagg)) or bool(re.match("B:", tagg)) or bool(
                                            re.match("i:", tagg)) or bool(re.match("b:", tagg)):
                                        valid_b = 0
                        if valid_b == 1:
                            self._train_corpus.loc[j, 'BIO'] = 'b:' + category
                            for l in i_indexes:
                                self._train_corpus.loc[l, 'BIO'] = 'i:' + category
                        if valid_b == 0:
                            last_b_idx = prev_last_b_idx
                    # if it is single-token VMWE
                    else:
                        self._train_corpus.loc[j, 'BIO'] = 'b:' + category
                        last_b_idx = j

            for o_idx in o_indexes:
                for oo_idx in range(o_idx[0] + 1, o_idx[1]):
                    tag = sentence.loc[oo_idx, 'BIO']
                    # if there is no tag in between an multi-token MWE, tag with 'o'
                    if not (bool(re.match("I:", tag)) or bool(re.match("B:", tag)) or bool(re.match("i:", tag)) or bool(
                            re.match("b:", tag))):
                        self._train_corpus.loc[oo_idx, 'BIO'] = 'o'

        # tags with 'O'
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: 'O' if not (bool(re.match("i:", x)) or bool(re.match("b:", x)) or bool(re.match("I:", x)) or bool(
                re.match("B:", x)) or bool(re.match("o", x)) or bool(re.match("space", x))) else x)

        # self._train_corpus.to_csv(fileName)
        self.to_cupt(self._train_corpus, self.train_tagged_path)

    def tag_IOB(self):
        self._train_corpus['BIO'] = copy.deepcopy(self._train_corpus['PARSEME:MWE'])
        self._train_corpus[self._train_corpus['BIO'].isnull()] = 'space'

        # remove other tags after the first tag
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: x.split(';')[0] if bool(re.match("\d:\w+[.]*\w+;.", x)) else x)
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: x.split(';')[0] if bool(re.match("\d;.", x)) else x)
        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(lambda x: x.strip())

        sentence_indexes = [-1] + list(self._train_corpus.loc[self._train_corpus['BIO'] == 'space'].index)

        # tag sentence by sentence
        for i, sentence_idx in enumerate(sentence_indexes[:-1]):
            sentence = self._train_corpus[sentence_idx + 1:sentence_indexes[i + 1]]
            last_B_idx = 0
            isB = False
            for j, row in sentence.iterrows():
                tag = sentence.loc[j, 'BIO']
                if (bool(re.match("\d:", tag)) and not isB) or (bool(re.match("\d:", tag)) and j > last_B_idx):
                    isB = True
                    no = tag.split(':')[0]
                    category = tag.split(':')[1]
                    category = category.strip()
                    self._train_corpus.loc[j, 'BIO'] = 'B:' + category
                    I_indexes = list(sentence.loc[self._train_corpus['BIO'] == no].index)
                    if len(I_indexes) > 0:
                        for k in I_indexes:
                            self._train_corpus.loc[k, 'BIO'] = 'I:' + category
                        last_B_idx = I_indexes[-1]
                    else:
                        last_B_idx = j

        self._train_corpus['BIO'] = self._train_corpus['BIO'].apply(
            lambda x: 'O' if not (
                    bool(re.match("I:", x)) or bool(re.match("B:", x)) or bool(re.match("space", x))) else x)

        # self._train_corpus.to_csv(fileName)
        self.to_cupt(self._train_corpus, self.train_tagged_path)

    def convert_gappy_tag(self):
        self._test_corpus = copy.deepcopy(pd.concat([self._test_corpus, self._test_rows]).sort_index())
        self._test_corpus.loc[self._test_corpus['ID_tag'] == 'extra', 'BIO'] = 'O'
        self._test_corpus['PARSEME:MWE'] = copy.deepcopy(self._test_corpus['BIO'])
        sentence_indexes = [-1] + list(self._test_corpus.loc[self._test_corpus['BIO'] == 'space'].index)

        # tag sentence by sentence
        for i, sentence_idx in enumerate(sentence_indexes[:-1]):
            sentence = self._test_corpus[sentence_idx + 1:sentence_indexes[i + 1]]
            counter = 0
            counterB = 0
            tagB = ""
            tagb = ""
            counterb = 0
            for j, row in sentence.iterrows():
                tag = sentence.loc[j, 'BIO']
                # tag = tag.strip() ###### ekle for FR and PL if necessary

                if tag == 'O':
                    self._test_corpus.loc[j, 'BIO'] = '*'

                elif tag == 'o':
                    self._test_corpus.loc[j, 'BIO'] = '*'

                elif not tag == 'space':
                    ib = tag.split(':')[0]
                    category = tag.split(':')[1]

                    if ib == "B":
                        counter = counter + 1
                        counterB = counter
                        tagB = category
                        self._test_corpus.loc[j, 'BIO'] = str(counterB) + ':' + tagB

                    if ib == "b":
                        counter = counter + 1
                        counterb = counter
                        tagb = category
                        self._test_corpus.loc[j, 'BIO'] = str(counterb) + ':' + tagb

                    if ib == "I" and tagB == category:
                        self._test_corpus.loc[j, 'BIO'] = str(counterB)

                    if ib == "i" and tagb == category:
                        self._test_corpus.loc[j, 'BIO'] = str(counterb)

                    if ib == "I" and (not (tagB == category)):
                        counter = counter + 1
                        counterB = counter
                        tagB = category
                        self._test_corpus.loc[j, 'BIO'] = str(counterB) + ':' + tagB

                    if ib == "i" and (not (tagb == category)):
                        counter = counter + 1
                        counterb = counter
                        tagb = category
                        self._test_corpus.loc[j, 'BIO'] = str(counterb) + ':' + tagb

        self.to_cupt_with_comments()

    def convert_IOB(self):
        self._test_corpus['PARSEME:MWE'] = copy.deepcopy(self._test_corpus['BIO'])
        sentence_indexes = [-1] + list(self._test_corpus.loc[self._test_corpus['BIO'] == 'space'].index)

        # tag sentence by sentence
        for i, sentence_idx in enumerate(sentence_indexes[:-1]):
            sentence = self._test_corpus[sentence_idx + 1:sentence_indexes[i + 1]]
            counter = 0
            counterB = 0
            tagB = ""
            for j, row in sentence.iterrows():
                tag = sentence.loc[j, 'BIO']
                # tag = tag.strip() ###### ekle for FR and PL if necessary

                if tag == 'O':
                    self._test_corpus.loc[j, 'BIO'] = '*'

                elif not tag == 'space':
                    ib = tag.split(':')[0]
                    category = tag.split(':')[1]

                    if ib == "B":
                        counter = counter + 1
                        counterB = counter
                        tagB = category
                        self._test_corpus.loc[j, 'BIO'] = str(counterB) + ':' + tagB

                    if ib == "I" and tagB == category:
                        self._test_corpus.loc[j, 'BIO'] = str(counterB)

                    if ib == "I" and (not (tagB == category)):
                        counter = counter + 1
                        counterB = counter
                        tagB = category
                        self._test_corpus.loc[j, 'BIO'] = str(counterB) + ':' + tagB

        self.to_cupt_with_comments()

    def create_char_matrix(self, sentences):
        char_matrix = []
        for sentence in sentences:
            sent_seq = []
            for i in range(self.max_sent):
                word_seq = []
                for j in range(self.max_char_length):
                    try:
                        word_seq.append(self.char2idx.get(sentence[i][CI['FORM']][j]))
                    except:
                        word_seq.append(self.char2idx.get("</s>"))
                sent_seq.append(word_seq)
            char_matrix.append(sent_seq)  # np.array(sent_seq)
        char_matrix = np.asarray(char_matrix)
        return char_matrix

    def create_spelling_embeddings(self):
        spelling_embeddings = []
        for word, idx in self.word2idx.items():
            spelling_feature_vector = self.get_spelling_feature_vector(word)
            spelling_embeddings.append(spelling_feature_vector)
        spelling_embeddings = np.asarray(spelling_embeddings)
        self.spelling_embeddings = spelling_embeddings

    def get_spelling_feature_vector(self, word):
        def include_digits(word):
            n_of_digits = len([char for char in word if char.isdigit()])
            res = n_of_digits > 0
            return res

        def include_punctuation(word):
            n_of_puncs = len([char for char in word if char.isdigit()])
            res = n_of_puncs > 0
            return res

        vector = np.zeros(8)

        if word[0].isupper():  # is initial upper
            vector[0] = 1
        if word.isupper():  # is all upper
            vector[1] = 1
        if word.islower():  # is all lower
            vector[2] = 1
        if word.isdigit():  # is it numeric
            vector[3] = 1
        if include_digits(word):  # does it include numeric
            vector[4] = 1
        if include_punctuation(word):  # does it include punc
            vector[5] = 1
        if '@' in word:  # is it email
            vector[6] = 1
        if 'http' in word:  # is it url
            vector[7] = 1

        return vector

    def create_morpheme_matrix(self, sentences):
        morpheme_matrix = []
        for sentence in sentences:
            sent_seq = []
            for i in range(self.max_sent):
                morpheme_seq = []
                for j in range(self.max_morpheme_len):
                    try:
                        feats = sentence[i][CI['FEATS']].split('|')
                        morpheme_seq.append(self.morpheme2idx.get(feats[j]))
                    except:
                        morpheme_seq.append(self.morpheme2idx.get("</s>"))
                sent_seq.append(morpheme_seq)
            morpheme_matrix.append(sent_seq)  # np.array(sent_seq)
        morpheme_matrix = np.asarray(morpheme_matrix)
        return morpheme_matrix

    def prepare_to_lstm(self):
        logging.info('Preparing to lstm..')
        self.train_sentences = self.read_sentences(self._train_corpus)
        self.test_sentences = self.read_sentences(self._test_corpus)

        self.words = []
        self.words.append("</s>")
        self.words.append("<UNK>")
        self.tr_words = self.words + list(set(self._train_corpus['FORM']))
        self.test_words = self.words + list(set(self._test_corpus['FORM']))
        self.words = self.words + list(set(self._train_corpus['FORM']) | set(
            self._test_corpus['FORM']))  # since cannot keep all word embeddings in memory

        self.tags = list(set(self._train_corpus['BIO']))
        self.tags.remove("space")

        self.feats = list(set(self._train_corpus['FEATS']) | set(self._test_corpus['FEATS']))
        self.morphemes = [feat.split('|') for feat in self.feats]
        self.max_morpheme_len = max([len(m) for m in self.morphemes])
        self.morphemes = list(set([j for i in self.morphemes for j in i]))
        self.morphemes.remove('space')
        self.morphemes = ["</s>"] + self.morphemes

        self.pos = []
        self.pos.append("</s>")
        self.pos = self.pos + list(set(self._train_corpus['UPOS']) | set(self._test_corpus['UPOS']))
        self.pos.remove("space")

        self.deprel = []
        self.deprel.append("</s>")
        self.deprel = self.deprel + list(set(self._train_corpus['DEPREL']) | set(self._test_corpus['DEPREL']))
        self.deprel.remove("space")

        self.chars = []
        self.chars.append("</s>")
        self.chars = self.chars + list(set([char for word in self.words for char in word]))

        self.n_words = len(self.words)
        self.n_chars = len(self.chars)
        self.n_tags = len(self.tags)
        self.n_morphemes = len(self.morphemes)

        self.word2idx = {w: i for i, w in enumerate(self.words)}
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.morpheme2idx = {t: i for i, t in enumerate(self.morphemes)}
        self.pos2idx = {t: i for i, t in enumerate(self.pos)}
        self.deprel2idx = {t: i for i, t in enumerate(self.deprel)}
        self.tag2idx = {t: i for i, t in enumerate(self.tags)}

        self.max_train_sent = max([len(sen) for sen in self.train_sentences])
        self.max_test_sent = max([len(sen) for sen in self.test_sentences])
        self.max_sent = max(self.max_train_sent, self.max_test_sent)

        word_lengths = [len(word) for word in self.words]
        word_length_mean = np.array(word_lengths).mean()
        word_length_std = np.array(word_lengths).std()
        max_char_length_bound = int(round(word_length_mean + 3 * word_length_std))
        max_char_length = max(word_lengths)
        if max_char_length > max_char_length_bound:
            self.max_char_length = max_char_length_bound
        else:
            self.max_char_length = max_char_length

        self.n_spelling_features = 8
        self.create_spelling_embeddings()

        self.pos_embeddings = np.identity(len(self.pos2idx.keys()) + 1)
        self.deprel_embeddings = np.identity(len(self.deprel2idx.keys()) + 1)

    def add_crf_features(self):
        self._train_corpus['FORM_RIGHT'] = self._train_corpus['FORM'].shift(-1)
        self._train_corpus['FORM_LEFT'] = self._train_corpus['FORM'].shift(1)
