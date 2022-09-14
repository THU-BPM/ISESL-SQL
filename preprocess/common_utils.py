#coding=utf8
import os, sqlite3
import numpy as np
import stanza, torch
from nltk.corpus import stopwords
from itertools import product, combinations
from utils.constants import MAX_RELATIVE_DIST
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
# from stanza.pipeline.core import DownloadMethod
# import geoopt as gt


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    

def agg(input):
    return torch.sum(input,dim=1,keepdim=True)/input.size(1)


def quote_normalization(question):
    """ Normalize all usage of quotation marks into a separate \" """
    new_question, quotation_marks = [], ["'", '"', '`', '‘', '’', '“', '”', '``', "''", "‘‘", "’’"]
    for idx, tok in enumerate(question):
        if len(tok) > 2 and tok[0] in quotation_marks and tok[-1] in quotation_marks:
            new_question += ["\"", tok[1:-1], "\""]
        elif len(tok) > 2 and tok[0] in quotation_marks:
            new_question += ["\"", tok[1:]]
        elif len(tok) > 2 and tok[-1] in quotation_marks:
            new_question += [tok[:-1], "\"" ]
        elif tok in quotation_marks:
            new_question.append("\"")
        elif len(tok) == 2 and tok[0] in quotation_marks:
            # special case: the length of entity value is 1
            if idx + 1 < len(question) and question[idx + 1] in quotation_marks:
                new_question += ["\"", tok[1]]
            else:
                new_question.append(tok)
        else:
            new_question.append(tok)
    return new_question


def visualize(question, table, column, table_mat, column_mat):
    print(question)
    print("----table match----")
    for q_index in range(len(question)):
        for t_index in range(len(table)):
            if table_mat[q_index][t_index]:
                if question[q_index] not in '."?,':
                    print(question[q_index], table[t_index])
    print("----column match----")
    for q_index in range(len(question)):
        for c_index in range(len(column)):
            if column_mat[q_index][c_index]:
                if question[q_index] not in '."?,':
                    print(question[q_index], column[c_index])
    import ipdb; ipdb.set_trace()
    pass


class Preprocessor():

    def __init__(self, args, db_dir='data/database', db_content=True):
        super(Preprocessor, self).__init__()
        self.db_dir = db_dir
        self.db_content = db_content
        self.nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma')#, use_gpu=False)
        self.stopwords = stopwords.words("english")
        self.stop_symbol = '."?,'
        self.semantic_graph = args.semantic_graph
        if self.semantic_graph:
            self.semantic_graph_g = SemanticGraphGenerator(args)
            self.threshold = args.semantic_threshold

    def pipeline(self, entry: dict, db: dict, verbose: bool = False):
        """ db should be preprocessed """
        entry = self.preprocess_question(entry, db, verbose=verbose)
        entry = self.schema_linking(entry, db, verbose=verbose)
        entry = self.extract_subgraph(entry, db, verbose=verbose)
        return entry

    def preprocess_database(self, db: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase table and column names for each database """
        table_toks, table_names = [], []
        for tab in db['table_names']:
            doc = self.nlp(tab)
            tab = [w.lemma.lower() for s in doc.sentences for w in s.words]
            table_toks.append(tab)
            table_names.append(" ".join(tab))
        db['processed_table_toks'], db['processed_table_names'] = table_toks, table_names
        column_toks, column_names = [], []
        for _, c in db['column_names']:
            doc = self.nlp(c)
            c = [w.lemma.lower() for s in doc.sentences for w in s.words]
            column_toks.append(c)
            column_names.append(" ".join(c))
        db['processed_column_toks'], db['processed_column_names'] = column_toks, column_names
        column2table = list(map(lambda x: x[0], db['column_names'])) # from column id to table id
        table2columns = [[] for _ in range(len(table_names))] # from table id to column ids list
        for col_id, col in enumerate(db['column_names']):
            if col_id == 0: continue
            table2columns[col[0]].append(col_id)
        db['column2table'], db['table2columns'] = column2table, table2columns

        t_num, c_num, dtype = len(db['table_names']), len(db['column_names']), '<U100'

        # relations in tables, tab_num * tab_num
        tab_mat = np.array([['table-table-generic'] * t_num for _ in range(t_num)], dtype=dtype)
        table_fks = set(map(lambda pair: (column2table[pair[0]], column2table[pair[1]]), db['foreign_keys']))
        for (tab1, tab2) in table_fks:
            if (tab2, tab1) in table_fks:
                tab_mat[tab1, tab2], tab_mat[tab2, tab1] = 'table-table-fkb', 'table-table-fkb'
            else:
                tab_mat[tab1, tab2], tab_mat[tab2, tab1] = 'table-table-fk', 'table-table-fkr'
        tab_mat[list(range(t_num)), list(range(t_num))] = 'table-table-identity'

        # relations in columns, c_num * c_num
        col_mat = np.array([['column-column-generic'] * c_num for _ in range(c_num)], dtype=dtype)
        for i in range(t_num):
            col_ids = [idx for idx, t in enumerate(column2table) if t == i]
            col1, col2 = list(zip(*list(product(col_ids, col_ids))))
            col_mat[col1, col2] = 'column-column-sametable'
        col_mat[list(range(c_num)), list(range(c_num))] = 'column-column-identity'
        if len(db['foreign_keys']) > 0:
            col1, col2 = list(zip(*db['foreign_keys']))
            col_mat[col1, col2], col_mat[col2, col1] = 'column-column-fk', 'column-column-fkr'
        col_mat[0, list(range(c_num))] = '*-column-generic'
        col_mat[list(range(c_num)), 0] = 'column-*-generic'
        col_mat[0, 0] = '*-*-identity'

        # relations between tables and columns, t_num*c_num and c_num*t_num
        tab_col_mat = np.array([['table-column-generic'] * c_num for _ in range(t_num)], dtype=dtype)
        col_tab_mat = np.array([['column-table-generic'] * t_num for _ in range(c_num)], dtype=dtype)
        cols, tabs = list(zip(*list(map(lambda x: (x, column2table[x]), range(1, c_num))))) # ignore *
        col_tab_mat[cols, tabs], tab_col_mat[tabs, cols] = 'column-table-has', 'table-column-has'
        if len(db['primary_keys']) > 0:
            cols, tabs = list(zip(*list(map(lambda x: (x, column2table[x]), db['primary_keys']))))
            col_tab_mat[cols, tabs], tab_col_mat[tabs, cols] = 'column-table-pk', 'table-column-pk'
        col_tab_mat[0, list(range(t_num))] = '*-table-generic'
        tab_col_mat[list(range(t_num)), 0] = 'table-*-generic'

        relations = np.concatenate([
            np.concatenate([tab_mat, tab_col_mat], axis=1),
            np.concatenate([col_tab_mat, col_mat], axis=1)
        ], axis=0)
        db['relations'] = relations.tolist()

        if verbose:
            print('Tables:', ', '.join(db['table_names']))
            print('Lemmatized:', ', '.join(table_names))
            print('Columns:', ', '.join(list(map(lambda x: x[1], db['column_names']))))
            print('Lemmatized:', ', '.join(column_names), '\n')
        return db

    def preprocess_question(self, entry: dict, db: dict, verbose: bool = False):
        """ Tokenize, lemmatize, lowercase question"""
        # stanza tokenize, lemmatize and POS tag
        if 'question_toks' in entry:
            question = ' '.join(quote_normalization(entry['question_toks']))
        else:
            question = entry['question']
        doc = self.nlp(question)
        
        raw_toks = [w.text.lower() for s in doc.sentences for w in s.words]
        toks = [w.lemma.lower() if w.lemma is not None else w.text.lower() for s in doc.sentences for w in s.words]
        pos_tags = [w.xpos for s in doc.sentences for w in s.words]

        if 'align' in entry:
            misc = [w.misc for s in doc.sentences for w in s.words]
            begin_end = []
            ends = []
            for item in misc:
                begin = int(item.split('|')[0].split('=')[1])
                end = int(item.split('|')[1].split('=')[1])
                ends.append(end)
                begin_end.append((begin, end))
            assert begin_end[-1][1] == entry['align'][-1]['end_char']
            align_index = 0
            align_label = []
            for index in range(len(begin_end)):
                assert begin_end[index][1] <= entry['align'][align_index]['end_char']
                align_label.append(entry['align'][align_index]['data'])
                if begin_end[index][1] == entry['align'][align_index]['end_char']:
                    align_index += 1
            entry['align_label'] = align_label

            for align in entry['align']:
                assert align['end_char'] in ends

        entry['raw_question_toks'] = raw_toks
        entry['processed_question_toks'] = toks
        entry['pos_tags'] = pos_tags

        # relations in questions, q_num * q_num
        q_num, dtype = len(toks), '<U100'
        if q_num <= MAX_RELATIVE_DIST + 1:
            dist_vec = ['question-question-dist' + str(i) if i != 0 else 'question-question-identity'
                for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)]
            starting = MAX_RELATIVE_DIST
        else:
            dist_vec = ['question-question-generic'] * (q_num - MAX_RELATIVE_DIST - 1) + \
                ['question-question-dist' + str(i) if i != 0 else 'question-question-identity' \
                    for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)] + \
                    ['question-question-generic'] * (q_num - MAX_RELATIVE_DIST - 1)
            starting = q_num - 1
        q_mat = np.array([dist_vec[starting - i: starting - i + q_num] for i in range(q_num)], dtype=dtype)
        entry['relations'] = q_mat.tolist()
        if verbose:
            print('Question:', entry['question'])
            print('Tokenized:', ' '.join(entry['raw_question_toks']))
            print('Lemmatized:', ' '.join(entry['processed_question_toks']))
            print('Pos tags:', ' '.join(entry['pos_tags']), '\n')
        return entry

    def extract_subgraph(self, entry: dict, db: dict, verbose: bool = False):
        sql = entry['sql']
        used_schema = {'table': set(), 'column': set()}
        used_schema = self.extract_subgraph_from_sql(sql, used_schema)
        entry['used_tables'] = sorted(list(used_schema['table']))
        entry['used_columns'] = sorted(list(used_schema['column']))

        if verbose:
            print('Used tables:', entry['used_tables'])
            print('Used columns:', entry['used_columns'], '\n')
        return entry

    def extract_subgraph_from_sql(self, sql: dict, used_schema: dict):
        select_items = sql['select'][1]
        # select clause
        for _, val_unit in select_items:
            if val_unit[0] == 0:
                col_unit = val_unit[1]
                used_schema['column'].add(col_unit[1])
            else:
                col_unit1, col_unit2 = val_unit[1:]
                used_schema['column'].add(col_unit1[1])
                used_schema['column'].add(col_unit2[1])
        # from clause conds
        table_units = sql['from']['table_units']
        for _, t in table_units:
            if type(t) == dict:
                used_schema = self.extract_subgraph_from_sql(t, used_schema)
            else:
                used_schema['table'].add(t)
        # from, where and having conds

        # if len(sql['from']['conds']) > 0:
        #     import ipdb;
        #     ipdb.set_trace()
        # used_schema = self.extract_subgraph_from_conds(sql['from']['conds'], used_schema)
        used_schema = self.extract_subgraph_from_conds(sql['where'], used_schema)
        used_schema = self.extract_subgraph_from_conds(sql['having'], used_schema)
        # groupBy and orderBy clause
        groupBy = sql['groupBy']
        for col_unit in groupBy:
            used_schema['column'].add(col_unit[1])
        orderBy = sql['orderBy']
        if len(orderBy) > 0:
            orderBy = orderBy[1]
            for val_unit in orderBy:
                if val_unit[0] == 0:
                    col_unit = val_unit[1]
                    used_schema['column'].add(col_unit[1])
                else:
                    col_unit1, col_unit2 = val_unit[1:]
                    used_schema['column'].add(col_unit1[1])
                    used_schema['column'].add(col_unit2[1])
        # union, intersect and except clause
        if sql['intersect']:
            used_schema = self.extract_subgraph_from_sql(sql['intersect'], used_schema)
        if sql['union']:
            used_schema = self.extract_subgraph_from_sql(sql['union'], used_schema)
        if sql['except']:
            used_schema = self.extract_subgraph_from_sql(sql['except'], used_schema)
        return used_schema

    def extract_subgraph_from_conds(self, conds: list, used_schema: dict):
        if len(conds) == 0:
            return used_schema
        for cond in conds:
            if cond in ['and', 'or']:
                continue
            val_unit, val1, val2 = cond[2:]
            if val_unit[0] == 0:
                col_unit = val_unit[1]
                used_schema['column'].add(col_unit[1])
            else:
                col_unit1, col_unit2 = val_unit[1:]
                used_schema['column'].add(col_unit1[1])
                used_schema['column'].add(col_unit2[1])
            if type(val1) == list:
                used_schema['column'].add(val1[1])
            elif type(val1) == dict:
                used_schema = self.extract_subgraph_from_sql(val1, used_schema)
            if type(val2) == list:
                used_schema['column'].add(val1[1])
            elif type(val2) == dict:
                used_schema = self.extract_subgraph_from_sql(val2, used_schema)
        return used_schema

    
    def schema_linking(self, entry: dict, db: dict, verbose: bool = False):
        # verbose = True
        """ Perform schema linking: both question and database need to be preprocessed """
        raw_question_toks, question_toks = entry['raw_question_toks'], entry['processed_question_toks']
        table_toks, column_toks = db['processed_table_toks'], db['processed_column_toks']
        table_names, column_names = db['processed_table_names'], db['processed_column_names']
        q_num, t_num, c_num, dtype = len(question_toks), len(table_toks), len(column_toks), '<U100'
        
        if self.semantic_graph:
            semanitc_q_tab_mat, semantic_q_col_mat = self.semantic_graph_g.generate_semantic_schema_linking(raw_question_toks, question_toks, table_toks, column_toks, table_names, column_names)

        # relations between questions and tables, q_num*t_num and t_num*q_num
        table_matched_pairs = {'partial': [], 'exact': []}
        q_tab_mat = np.array([['question-table-nomatch'] * t_num for _ in range(q_num)], dtype=dtype)
        q_tab_weight = np.zeros((q_num, t_num), dtype=int)
        tab_q_mat = np.array([['table-question-nomatch'] * q_num for _ in range(t_num)], dtype=dtype)

        # not use gold linking data, use string match linking
        if 'align_label' not in entry:
            max_len = max([len(t) for t in table_toks])
            index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
            index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
            for i, j in index_pairs:
                phrase = ' '.join(question_toks[i: j])
                if phrase in self.stopwords: continue
                for idx, name in enumerate(table_names):
                    if phrase == name: # fully match will overwrite partial match due to sort
                        q_tab_mat[range(i, j), idx] = 'question-table-exactmatch'
                        tab_q_mat[idx, range(i, j)] = 'table-question-exactmatch'
                        if verbose:
                            table_matched_pairs['exact'].append(str((name, idx, phrase, i, j)))
                    elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                        q_tab_mat[range(i, j), idx] = 'question-table-partialmatch'
                        tab_q_mat[idx, range(i, j)] = 'table-question-partialmatch'
                        if verbose:
                            table_matched_pairs['partial'].append(str((name, idx, phrase, i, j)))

        if self.semantic_graph:
            assert semanitc_q_tab_mat.shape[0] == t_num and semanitc_q_tab_mat.shape[1] == q_num
            for x in range(t_num):
                for y in range(q_num):
                    if question_toks[y] in self.stopwords or question_toks[y] in self.stop_symbol:
                        continue
                    if semanitc_q_tab_mat[x, y] > self.threshold:
                        q_tab_weight[y, x] = 1

        # relations between questions and columns
        column_matched_pairs = {'partial': [], 'exact': [], 'value': []}
        q_col_mat = np.array([['question-column-nomatch'] * c_num for _ in range(q_num)], dtype=dtype)
        q_col_weight = np.zeros((q_num, c_num), dtype=int)
        col_q_mat = np.array([['column-question-nomatch'] * q_num for _ in range(c_num)], dtype=dtype)

        if 'align_label' in entry:
            assert len(entry['align_label']) == q_num
            for index in range(len(entry['align_label'])):
                if entry['align_label'][index] is not None:
                    if entry['align_label'][index]['type'] == 'col':
                        column_id = entry['align_label'][index]['id']
                        q_col_mat[index, column_id] = 'question-column-semanticmatch'
                        col_q_mat[column_id, index] = 'column-question-semanticmatch'
                        if verbose:
                            column_matched_pairs['exact'].append(str((column_names[column_id], column_id, raw_question_toks[index])))
                    elif entry['align_label'][index]['type'] == 'val':
                        column_id = entry['align_label'][index]['id']
                        q_col_mat[index, column_id] = 'question-column-semanticmatch'
                        col_q_mat[column_id, index] = 'column-question-semanticmatch'
                        if verbose:
                            column_matched_pairs['value'].append(str((column_names[column_id], column_id, raw_question_toks[index])))
                    elif entry['align_label'][index]['type'] == 'tbl':
                        table_id = entry['align_label'][index]['id']
                        q_tab_mat[index, table_id] = 'question-table-semanticmatch'
                        tab_q_mat[table_id, index] = 'table-question-semanticmatch'
                        if verbose:
                            table_matched_pairs['exact'].append(str((table_names[table_id], table_id, raw_question_toks[index])))
        else:
            max_len = max([len(c) for c in column_toks])
            index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
            index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
            for i, j in index_pairs:
                phrase = ' '.join(question_toks[i: j])
                if phrase in self.stopwords: continue
                for idx, name in enumerate(column_names):
                    if phrase == name: # fully match will overwrite partial match due to sort
                        q_col_mat[range(i, j), idx] = 'question-column-exactmatch'
                        col_q_mat[idx, range(i, j)] = 'column-question-exactmatch'
                        if verbose:
                            column_matched_pairs['exact'].append(str((name, idx, phrase, i, j)))
                    elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                        q_col_mat[range(i, j), idx] = 'question-column-partialmatch'
                        col_q_mat[idx, range(i, j)] = 'column-question-partialmatch'
                        if verbose:
                            column_matched_pairs['partial'].append(str((name, idx, phrase, i, j)))
            if self.db_content:
                db_file = os.path.join(self.db_dir, db['db_id'], db['db_id'] + '.sqlite')
                if not os.path.exists(db_file):
                    raise ValueError('[ERROR]: database file %s not found ...' % (db_file))
                conn = sqlite3.connect(db_file)
                conn.text_factory = lambda b: b.decode(errors='ignore')
                conn.execute('pragma foreign_keys=ON')
                for i, (tab_id, col_name) in enumerate(db['column_names_original']):
                    if i == 0 or 'id' in column_toks[i]: # ignore * and special token 'id'
                        continue
                    tab_name = db['table_names_original'][tab_id]
                    try:
                        cursor = conn.execute("SELECT DISTINCT \"%s\" FROM \"%s\";" % (col_name, tab_name))
                        cell_values = cursor.fetchall()
                        cell_values = [str(each[0]) for each in cell_values]
                        cell_values = [[str(float(each))] if is_number(each) else each.lower().split() for each in cell_values]
                    except Exception as e:
                        print(e)
                    for j, word in enumerate(raw_question_toks):
                        word = str(float(word)) if is_number(word) else word
                        for c in cell_values:
                            if word in c and 'nomatch' in q_col_mat[j, i] and word not in self.stopwords:
                                q_col_mat[j, i] = 'question-column-valuematch'
                                col_q_mat[i, j] = 'column-question-valuematch'
                                if verbose:
                                    column_matched_pairs['value'].append(str((column_names[i], i, word, j, j + 1)))
                                break
                conn.close()

        if self.semantic_graph:
            assert semantic_q_col_mat.shape[0] == c_num and semantic_q_col_mat.shape[1] == q_num
            for x in range(c_num):
                for y in range(q_num):
                    if question_toks[y] in self.stopwords or question_toks[y] in self.stop_symbol:
                        continue
                    if semantic_q_col_mat[x, y] > self.threshold:
                        q_col_weight[y, x] = 1

        # two symmetric schema linking matrix: q_num x (t_num + c_num), (t_num + c_num) x q_num
        q_col_mat[:, 0] = 'question-*-generic'
        col_q_mat[0] = '*-question-generic'

        q_tab_mat2 = np.array([['question-table-semanticmatch'] * t_num for _ in range(q_num)], dtype=dtype)
        tab_q_mat2 = np.array([['table-question-semanticmatch'] * q_num for _ in range(t_num)], dtype=dtype)
        q_col_mat2 = np.array([['question-column-semanticmatch'] * c_num for _ in range(q_num)], dtype=dtype)
        col_q_mat2 = np.array([['column-question-semanticmatch'] * q_num for _ in range(c_num)], dtype=dtype)

        schema_weight = np.concatenate([q_tab_weight, q_col_weight], axis=1)
        q_schema = np.concatenate([q_tab_mat, q_col_mat], axis=1)
        schema_q = np.concatenate([tab_q_mat, col_q_mat], axis=0)

        q_schema2 = np.concatenate([q_tab_mat2, q_col_mat2], axis=1)
        schema_q2 = np.concatenate([tab_q_mat2, col_q_mat2], axis=0)
        entry['schema_linking'] = (q_schema.tolist(), schema_q.tolist())

        entry['schema_linking2'] = (q_schema2.tolist(), schema_q2.tolist())
        entry['schema_weight'] = schema_weight
        if verbose:
            print('Question:', ' '.join(question_toks))
            print('Table matched: (table name, column id, question span, start id, end id)')
            print('Exact match:', ', '.join(table_matched_pairs['exact']) if table_matched_pairs['exact'] else 'empty')
            print('Partial match:', ', '.join(table_matched_pairs['partial']) if table_matched_pairs['partial'] else 'empty')
            print('Column matched: (column name, column id, question span, start id, end id)')
            print('Exact match:', ', '.join(column_matched_pairs['exact']) if column_matched_pairs['exact'] else 'empty')
            print('Partial match:', ', '.join(column_matched_pairs['partial']) if column_matched_pairs['partial'] else 'empty')
            print('Value match:', ', '.join(column_matched_pairs['value']) if column_matched_pairs['value'] else 'empty', '\n')
        return entry


class SemanticGraphGenerator():
    
    def __init__(self, args):
        self.device = torch.device("cuda:0")
        self.feature_size, self.max_batch_size = args.semantic_feature_size, args.semantic_batch_size
        self.model = AutoModel.from_pretrained(args.semantic_pretrain_model).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(args.semantic_pretrain_model)
    
    def tokenize_schema(self, word_names):
        word_list = [t.lower().split() for t in word_names]
        word_ids, sub_word_len, word_len = [], [], []
        for word in word_list:
            length = 0
            for w in word:
                toks = self.tokenizer.encode(w)[1:-1]
                word_ids.extend(toks)
                sub_word_len.append(len(toks))
                length += len(toks)
            word_len.append(length)
        return word_ids, sub_word_len, word_len

    def tokenize_question(self, question_toks):
        question = [q.lower() for q in question_toks]
        question_id, question_subword_len = [self.tokenizer.cls_token_id], []
        for w in question:
            toks = self.tokenizer.encode(w)[1:-1]
            question_id += toks
            question_subword_len.append(len(toks))
        question_mask_plm = [0] + [1] * (len(question_id) - 1) + [0]
        question_id.append(self.tokenizer.sep_token_id)
        return question_id, question_subword_len, question_mask_plm

    def generate_mask(self, question_id, column_id, table_id, question_mask_plm, table_mask_plm, column_mask_plm):
        question_mask_plm = question_mask_plm + [0] * (len(table_id) + len(column_id))
        table_mask_plm = [0] * len(question_id) + table_mask_plm + [0] * len(column_id)
        column_mask_plm = [0] * (len(question_id) + len(table_id)) + column_mask_plm
        return question_mask_plm, table_mask_plm, column_mask_plm

    def make_mask_input(self, question_id, table_id, column_id, question_subword_len):
        masked_question_id = [question_id]
        start = 1
        for i, sub_len in enumerate(question_subword_len):
            tmp_question_id = question_id.copy()
            for m in range(start, start + sub_len):
                tmp_question_id[m] = self.tokenizer.mask_token_id
            masked_question_id.append(tmp_question_id)
            start += sub_len

        input_id, atten_mask = [], []
        for i, msk_q_id in enumerate(masked_question_id):
            input_id.append(msk_q_id + table_id + column_id)
            atten_mask.append([1] * len(input_id[-1]))
        return input_id, atten_mask
    
    def generate_semantic_schema_linking(self, raw_question_toks, question_toks, table_toks, column_toks, table_names, column_names):
        assert len(column_names) == len(column_toks) and len(table_names) == len(table_toks) and len(raw_question_toks) == len(question_toks)
        
        # process question
        question_id, question_subword_len, question_mask_plm = self.tokenize_question(question_toks)
        
        # process table
        table_id, table_subword_len, table_word_len = self.tokenize_schema(table_names)
        table_mask_plm = [1] * len(table_id)
        
        # process column
        column_id, column_subword_len, column_word_len = self.tokenize_schema(column_names)
        column_mask_plm = [1] * len(column_id) + [0]
        exact_column_token = len(column_id)
        column_id.append(self.tokenizer.sep_token_id)
        
        # generate mask
        question_mask_plm, table_mask_plm, column_mask_plm = self.generate_mask(question_id, column_id, table_id, question_mask_plm, table_mask_plm, column_mask_plm)
        
        #mask quesiton id
        input_id, atten_mask = self.make_mask_input(question_id, table_id, column_id, question_subword_len)
        
        # genrate embedding
        output_arr = []
        for idx in range(0, len(input_id), self.max_batch_size):
            _input_id = torch.tensor(input_id[idx: idx + self.max_batch_size], dtype=torch.long, device=self.device)
            _input_mask = torch.tensor(atten_mask[idx: idx + self.max_batch_size], dtype=torch.float, device=self.device)
            outputs = self.model(_input_id, _input_mask)[0].squeeze()
            output_arr.append(outputs.reshape(-1, len(input_id[0]), self.feature_size))

        # aggregate
        outputs = torch.cat((output_arr), dim=0)
        # generate quesiton table relation
        q_tab_mat = outputs.new_zeros(len(raw_question_toks), len(table_names))
        
        tables_out = outputs.masked_select(torch.tensor(table_mask_plm, dtype=torch.bool, device=self.device).unsqueeze(-1).unsqueeze(0).repeat(
                outputs.size(0), 1, 1)).view(outputs.size(0), len(table_id), self.feature_size)
        
        current_len, table_features = 0, []
        for sub_len in table_word_len:
            table_features.append(agg(tables_out[:, current_len:current_len + sub_len]))
            current_len += sub_len

        table_features = torch.cat(table_features, 1)
        table_standard, table_masked = table_features[0: 1], table_features[1:]
        
        assert table_masked.size(0) == len(raw_question_toks)
        
        for i in range(len(table_word_len)):
            q_tab_mat[:, i] = torch.linalg.norm(table_standard[:, i] - table_masked[:, i], dim=1, ord=2)
        
        # generate question column relation
        q_col_mat = outputs.new_zeros(len(raw_question_toks), len(column_names))
        columns_out = outputs.masked_select(torch.tensor(column_mask_plm, dtype=torch.bool, device=self.device).unsqueeze(-1).unsqueeze(0).repeat(
                outputs.size(0), 1, 1)).view(outputs.size(0), exact_column_token, self.feature_size)
        
        current_len, column_features = 0, []
        for sub_len in column_word_len:
            column_features.append(agg(columns_out[:, current_len:current_len + sub_len]))
            current_len += sub_len
            
        column_features = torch.cat(column_features, 1)
        column_standard, column_masked  = column_features[0:1], column_features[1:]
        
        assert column_masked.size(0) == len(raw_question_toks)
        for i in range(len(column_word_len)):
            q_col_mat[:, i] = torch.linalg.norm(column_standard[:, i] - column_masked[:, i], dim=1, ord=2)
        
        # final process
        combine_matrix = torch.cat([q_tab_mat, q_col_mat], dim=1)
        combine_matrix = (combine_matrix - torch.min(combine_matrix)) / (torch.max(combine_matrix) - torch.min(combine_matrix))
        
        final_q_tab_mat = combine_matrix[:, :q_tab_mat.size(1)].transpose(0, 1).cpu().detach().numpy()
        final_q_col_mat = combine_matrix[:, q_tab_mat.size(1):].transpose(0, 1).cpu().detach().numpy()
        # visualize(question_toks, table_toks, column_toks, use_q_tab_mat > self.threshold, use_q_col_mat > self.threshold)
        assert final_q_tab_mat.shape[0] == len(table_toks) and final_q_col_mat.shape[0] == len(column_toks)
        return final_q_tab_mat, final_q_col_mat

    
    
    