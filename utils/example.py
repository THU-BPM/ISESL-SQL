#coding=utf8
import os, pickle, json
import torch, random
import numpy as np
from asdls.asdl import ASDLGrammar
from asdls.transition_system import TransitionSystem
from utils.constants import UNK, GRAMMAR_FILEPATH, SCHEMA_TYPES, RELATIONS
from utils.graph_example import GraphFactory
from utils.vocab import Vocab
from utils.word2vec import Word2vecUtils
from transformers import AutoTokenizer
from utils.evaluator import Evaluator
from itertools import chain

class Example():

    @classmethod
    def configuration(cls, plm=None, method='lgesql', table_path='data/new_tables.json', tables='data/tables.bin', db_dir='data/database'):
        cls.plm, cls.method = plm, method
        cls.grammar = ASDLGrammar.from_filepath(GRAMMAR_FILEPATH)
        cls.trans = TransitionSystem.get_class_by_lang('sql')(cls.grammar)
        cls.tables = pickle.load(open(tables, 'rb')) if type(tables) == str else tables
        cls.evaluator = Evaluator(cls.trans, table_path, db_dir)
        cls.tokenizer = AutoTokenizer.from_pretrained(plm)
        cls.word_vocab = cls.tokenizer.get_vocab()
        cls.relation_vocab = Vocab(padding=False, unk=False, boundary=False, iterable=RELATIONS, default=None)
        cls.graph_factory = GraphFactory(cls.method, cls.relation_vocab)

    @classmethod
    def load_dataset(cls, choice, debug=False):
        # assert choice in ['train', 'dev']
        print("loader ", choice)
        fp = os.path.join('data', choice + '.' + cls.method + '.bin')
        datasets = pickle.load(open(fp, 'rb'))
        # question_lens = [len(ex['processed_question_toks']) for ex in datasets]
        # print('Max/Min/Avg question length in %s dataset is: %d/%d/%.2f' % (choice, max(question_lens), min(question_lens), float(sum(question_lens))/len(question_lens)))
        # action_lens = [len(ex['actions']) for ex in datasets]
        # print('Max/Min/Avg action length in %s dataset is: %d/%d/%.2f' % (choice, max(action_lens), min(action_lens), float(sum(action_lens))/len(action_lens)))
        examples, outliers = [], 0
        for ex in datasets:
            if ex['db_id'] == 'new_concert_singer':
                ex['db_id'] = 'concert_singer'
            if choice == 'train' and len(cls.tables[ex['db_id']]['column_names']) > 100:
                outliers += 1
                continue
            examples.append(cls(ex, cls.tables[ex['db_id']]))
            if debug and len(examples) >= 100:
                return examples
        if choice == 'train':
            print("Skip %d extremely large samples in training dataset ..." % (outliers))
        return examples

    def __init__(self, ex: dict, db: dict):
        super(Example, self).__init__()
        self.ex = ex
        self.db = db

        """ Mapping word to corresponding index """
        t = Example.tokenizer
        self.question = [q.lower() for q in ex['raw_question_toks']]
        self.question_id = [t.cls_token_id] # map token to id
        self.question_mask_plm = [] # remove SEP token in our case
        self.question_subword_len = [] # subword len for each word, exclude SEP token
        self.question_total_len = 0
        for w in self.question:
            toks = t.convert_tokens_to_ids(t.tokenize(w))
            self.question_id.extend(toks)
            self.question_subword_len.append(len(toks))
            # self.question_total_len += len(w)
        self.question_total_len = len(self.question)
        self.question_mask_plm = [0] + [1] * (len(self.question_id) - 1) + [0]
        self.question_id.append(t.sep_token_id)

        self.table = [['table'] + t.lower().split() for t in db['table_names']]
        ### 增加 label 设置
        self.schema_labels = []
        self.table_labels = []
        self.table_begin_end = []
        for table_index in range(len(self.table)):
            label = 1 if table_index in ex['used_tables'] else 0
            self.schema_labels.append(label)
            begin = len(self.table_labels) + 1
            self.table_labels.extend([label for _ in range(len(self.table[table_index]))])
            end = len(self.table_labels) -1
            self.table_begin_end.append((begin, end))

        self.table_id, self.table_mask_plm, self.table_subword_len = [], [], []
        self.table_word_len = []
        self.table_total_len = 0
        for s in self.table:
            l = 0
            for w in s:
                toks = t.convert_tokens_to_ids(t.tokenize(w))
                self.table_id.extend(toks)
                self.table_subword_len.append(len(toks))
                l += len(toks)
            self.table_word_len.append(l)
            self.table_total_len += len(s)
        self.table_mask_plm = [1] * len(self.table_id)

        self.column = [[db['column_types'][idx].lower()] + c.lower().split() for idx, (_, c) in enumerate(db['column_names'])]
        ### 增加label设置
        self.column_labels = []
        self.column_begin_end = []
        for column_index in range(len(self.column)):
            if column_index in ex['used_columns']:
                label = 1
            else:
                label = 0
            self.schema_labels.append(label)
            begin = len(self.column_labels) + 1
            self.column_labels.extend([label for _ in range(len(self.column[column_index]))])
            end = len(self.column_labels) - 1
            self.column_begin_end.append((begin, end))

        self.column_id, self.column_mask_plm, self.column_subword_len = [], [], []
        self.column_word_len = []
        self.column_total_len = 0
        for s in self.column:
            l = 0
            for w in s:
                toks = t.convert_tokens_to_ids(t.tokenize(w))
                self.column_id.extend(toks)
                self.column_subword_len.append(len(toks))
                l += len(toks)
            self.column_word_len.append(l)
            self.column_total_len += len(s)
        self.column_mask_plm = [1] * len(self.column_id) + [0]
        self.column_id.append(t.sep_token_id)

        self.input_id = self.question_id + self.table_id + self.column_id
        self.segment_id = [0] * len(self.question_id) + [1] * (len(self.table_id) + len(self.column_id)) \
            if Example.plm != 'grappa_large_jnt' and not Example.plm.startswith('roberta') \
            else [0] * (len(self.question_id) + len(self.table_id) + len(self.column_id))

        self.question_mask_plm = self.question_mask_plm + [0] * (len(self.table_id) + len(self.column_id))
        self.table_mask_plm = [0] * len(self.question_id) + self.table_mask_plm + [0] * len(self.column_id)
        self.column_mask_plm = [0] * (len(self.question_id) + len(self.table_id)) + self.column_mask_plm

        self.graph = Example.graph_factory.graph_construction(ex, db)

        # outputs
        if 'schema_weight' in ex['graph'].__dict__.keys():
            self.schema_weight = ex['graph'].schema_weight
        self.query = ' '.join(ex['query'].split('\t'))
        self.ast = ex['ast']
        self.tgt_action = ex['actions']
        self.used_tables, self.used_columns = ex['used_tables'], ex['used_columns']
        self.column_dict, self.table_dict = {}, {}
        

def get_position_ids(ex, shuffle=True):
    # cluster columns with their corresponding table and randomly shuffle tables and columns
    # [CLS] q1 q2 ... [SEP] * t1 c1 c2 c3 t2 c4 c5 ... [SEP]
    db, table_word_len, column_word_len = ex.db, ex.table_word_len, ex.column_word_len
    table_num, column_num = len(db['table_names']), len(db['column_names'])
    question_position_id = list(range(len(ex.question_id)))
    start = len(question_position_id)
    table_position_id, column_position_id = [None] * table_num, [None] * column_num
    column_position_id[0] = list(range(start, start + column_word_len[0]))
    start += column_word_len[0] # special symbol * first
    table_idxs = list(range(table_num))
    if shuffle:
        random.shuffle(table_idxs)
    for idx in table_idxs:
        col_idxs = db['table2columns'][idx]
        table_position_id[idx] = list(range(start, start + table_word_len[idx]))
        start += table_word_len[idx]
        if shuffle:
            random.shuffle(col_idxs)
        for col_id in col_idxs:
            column_position_id[col_id] = list(range(start, start + column_word_len[col_id]))
            start += column_word_len[col_id]
    position_id = question_position_id + list(chain.from_iterable(table_position_id)) + \
        list(chain.from_iterable(column_position_id)) + [start]
    assert len(position_id) == len(ex.input_id)
    return position_id
