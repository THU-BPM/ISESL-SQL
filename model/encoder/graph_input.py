# coding=utf8
import os, math
import torch
import torch.nn as nn
from model.model_utils import rnn_wrapper, lens2mask, PoolingFunction
from transformers import AutoModel, AutoConfig, AutoTokenizer
import numpy as np
import random
import copy

class GraphInputLayer(nn.Module):

    def __init__(self, embed_size, hidden_size, word_vocab, dropout=0.2, fix_grad_idx=60,
                 schema_aggregation='head+tail'):
        super(GraphInputLayer, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.word_vocab = word_vocab
        self.fix_grad_idx = fix_grad_idx
        self.word_embed = nn.Embedding(self.word_vocab, self.embed_size, padding_idx=0)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.rnn_layer = InputRNNLayer(self.embed_size, self.hidden_size, cell='lstm',
                                       schema_aggregation=schema_aggregation)

    def pad_embedding_grad_zero(self, index=None):
        self.word_embed.weight.grad[0].zero_()  # padding symbol is always 0
        if index is not None:
            if not torch.is_tensor(index):
                index = torch.tensor(index, dtype=torch.long, device=self.word_embed.weight.grad.device)
            self.word_embed.weight.grad.index_fill_(0, index, 0.)
        else:
            self.word_embed.weight.grad[self.fix_grad_idx:].zero_()

    def forward(self, batch):
        question, table, column = self.word_embed(batch.questions), self.word_embed(batch.tables), self.word_embed(
            batch.columns)
        if batch.question_unk_mask is not None:
            question = question.masked_scatter_(batch.question_unk_mask.unsqueeze(-1),
                                                batch.question_unk_embeddings[:, :self.embed_size])
        if batch.table_unk_mask is not None:
            table = table.masked_scatter_(batch.table_unk_mask.unsqueeze(-1),
                                          batch.table_unk_embeddings[:, :self.embed_size])
        if batch.column_unk_mask is not None:
            column = column.masked_scatter_(batch.column_unk_mask.unsqueeze(-1),
                                            batch.column_unk_embeddings[:, :self.embed_size])
        input_dict = {
            "question": self.dropout_layer(question),
            "table": self.dropout_layer(table),
            "column": self.dropout_layer(column)
        }
        inputs = self.rnn_layer(input_dict, batch)
        return inputs


class GraphInputLayerPLM(nn.Module):

    def __init__(self, args, plm='bert-base-uncased', hidden_size=256, dropout=0., subword_aggregation='mean',
                 schema_aggregation='head+tail', lazy_load=False):
        super(GraphInputLayerPLM, self).__init__()
        self.plm_model = AutoModel.from_config(AutoConfig.from_pretrained(plm)) \
            if lazy_load else AutoModel.from_pretrained(plm)
        self.config = self.plm_model.config
        self.subword_aggregation = SubwordAggregation(self.config.hidden_size, subword_aggregation=subword_aggregation)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.rnn_layer = InputRNNLayer(self.config.hidden_size, hidden_size, cell='lstm',
                                       schema_aggregation=schema_aggregation)
        self.classifier = nn.Sequential(
            nn.Linear(self.plm_model.config.hidden_size, 2),
        )
        self.softmax = nn.Softmax(dim=2)
        self.pruning_edge = args.pruning_edge
        self.noise_edge = args.noise_edge
        self.optimize_graph = args.optimize_graph
        self.schema_loss = args.schema_loss
        self.filter_edge = args.filter_edge
        self.use_semantic = args.semantic
        self.filter_gold_edge = args.filter_gold_edge
        self.not_use_weight_transform = args.not_use_weight_transform
        self.random_question_edge = args.random_question_edge
        self.dynamic_rate = args.dynamic_rate
        if self.optimize_graph:
            self.question_trans = nn.Linear(hidden_size, hidden_size)
            self.schema_trans = nn.Linear(hidden_size, hidden_size)

    def pad_embedding_grad_zero(self, index=None):
        pass

    def gather_schema_label(self, predict_sequence, begin_end, offset):
        result = []
        for n_b in range(len(begin_end)):
            batch_predict = []
            for index in range(len(begin_end[n_b])):
                begin, end = begin_end[n_b][index]
                sub_list = predict_sequence[n_b][begin + offset: end + offset + 1]
                if 1 in sub_list and 0 not in sub_list:
                    batch_predict.append(index)
            result.append(batch_predict)
        return result

    def value_with_threshold(self, logits, threshold):
        result = []
        for index in range(len(logits)):
            arr = logits[index]
            sub_result = []
            for sub_index in range(len(arr)):
                if arr[sub_index][1] > threshold:
                    sub_result.append(1)
                else:
                    sub_result.append(0)
            result.append(sub_result)
        return result

    def forward(self, batch, standard=False, train=True):
        batch.inputs.pop('token_type_ids')
        outputs = self.plm_model(**batch.inputs)[0]  # final layer hidden states
        
        # aggregate based on real word
        question, table, column, table_batch, column_batch = self.subword_aggregation(outputs, batch)
        
        semantic_column, semantic_table = [], []

        input_dict = {
            "question": self.dropout_layer(question),
            "table": self.dropout_layer(table),
            "column": self.dropout_layer(column)
        }
        inputs, q_outputs, s_outputs = self.rnn_layer(input_dict, batch)

        q_s_sims, loss = [], 0

        if self.optimize_graph:
            question_key = q_outputs if self.not_use_weight_transform else self.question_trans(q_outputs)
            schema_key = s_outputs if self.not_use_weight_transform else self.schema_trans(s_outputs)
            
            cos_sim = torch.matmul(schema_key, question_key.transpose(-1, -2))
            question_key_norm, schema_key_norm = torch.norm(question_key, p=2, dim=-1), torch.norm(schema_key, p=2, dim=-1)
            cos_sim /= (question_key_norm.unsqueeze(-2) * schema_key_norm.unsqueeze(-1))
            cos_sim = torch.relu(cos_sim)
            bce_lossfct = torch.nn.BCELoss()

            if self.filter_edge:
                semantic_column1, semantic_table1 = [], []
                
            for n_b in range(question.shape[0]):
                question_len = batch.question_lens.tolist()[n_b]
                schema_len = (batch.table_lens + batch.column_lens).tolist()[n_b]
                cos_sim_n = cos_sim[n_b][:schema_len, :question_len] #  remove padding part
                schema_logits, idx = torch.max(cos_sim_n, dim=1)  #  get the most relavent question for each schema
                cos_sim_n_new = cos_sim_n.new_zeros(cos_sim_n.shape)
                cos_sim_n_new[torch.arange(0, len(idx)), idx] = schema_logits  # weight in graph
                
                if self.schema_loss:
                    schema_label = batch.schema_labels[n_b][:schema_len].to(cos_sim.dtype)
                    loss += bce_lossfct(schema_logits, schema_label)

                if self.filter_edge:
                    semantic_column_n, semantic_table_n = [], []
                    idx = idx.cpu().detach().numpy()
                    
                    schema_label = batch.schema_labels[n_b][:schema_len].to(cos_sim.dtype)
                    cos_sim_n_new = cos_sim_n.new_zeros(cos_sim_n.shape)
                    if self.filter_gold_edge:
                        cos_sim_n_new[torch.arange(0, len(idx)), idx] = schema_label
                    else:
                        cos_sim_n_new[torch.arange(0, len(idx)), idx] = schema_label*schema_logits
                schema_weight = torch.from_numpy(batch.schema_weight[n_b].T).to(cos_sim_n_new.device)
                cos_sim_n_new = self.dynamic_rate * cos_sim_n_new + (1-self.dynamic_rate) * schema_weight
                
                question_sim, schema_sim = cos_sim_n.new_zeros((question_len, question_len)), cos_sim_n.new_zeros((schema_len, schema_len))
                up_m, down_m = torch.cat((question_sim,  cos_sim_n_new.t()), dim=1), torch.cat((cos_sim_n_new, schema_sim), dim=1)
                q_s_sims.append(torch.cat((up_m, down_m), dim=0))

        if self.filter_edge:
            semantic_column = semantic_column1
            semantic_table = semantic_table1
        return inputs, loss, (semantic_column, semantic_table), q_s_sims


class SubwordAggregation(nn.Module):
    """ Map subword or wordpieces into one fixed size vector based on aggregation method
    """

    def __init__(self, hidden_size, subword_aggregation='mean-pooling'):
        super(SubwordAggregation, self).__init__()
        self.hidden_size = hidden_size
        self.aggregation = PoolingFunction(self.hidden_size, self.hidden_size, method=subword_aggregation)

    def forward(self, inputs, batch):
        """ Transform pretrained model outputs into our desired format
        questions: bsize x max_question_len x hidden_size
        tables: bsize x max_table_word_len x hidden_size
        columns: bsize x max_column_word_len x hidden_size
        """
        old_questions, old_tables, old_columns = inputs.masked_select(batch.question_mask_plm.unsqueeze(-1)), \
                                                 inputs.masked_select(
                                                     batch.table_mask_plm.unsqueeze(-1)), inputs.masked_select(
            batch.column_mask_plm.unsqueeze(-1))
        questions = old_questions.new_zeros(batch.question_subword_lens.size(0), batch.max_question_subword_len,
                                            self.hidden_size)
        questions = questions.masked_scatter_(batch.question_subword_mask.unsqueeze(-1), old_questions)
        tables = old_tables.new_zeros(batch.table_subword_lens.size(0), batch.max_table_subword_len, self.hidden_size)
        tables = tables.masked_scatter_(batch.table_subword_mask.unsqueeze(-1), old_tables)
        columns = old_columns.new_zeros(batch.column_subword_lens.size(0), batch.max_column_subword_len,
                                        self.hidden_size)
        columns = columns.masked_scatter_(batch.column_subword_mask.unsqueeze(-1), old_columns)

        questions = self.aggregation(questions, mask=batch.question_subword_mask)
        tables = self.aggregation(tables, mask=batch.table_subword_mask)
        columns = self.aggregation(columns, mask=batch.column_subword_mask)

        new_questions, new_tables, new_columns, new_tables_batch, new_columns_batch = \
            questions.new_zeros(len(batch), batch.max_question_len, self.hidden_size), \
            tables.new_zeros(batch.table_word_mask.size(0), batch.max_table_word_len, self.hidden_size), \
            columns.new_zeros(batch.column_word_mask.size(0), batch.max_column_word_len, self.hidden_size), \
            tables.new_zeros(len(batch), batch.max_table_total_len, self.hidden_size), \
            columns.new_zeros(len(batch), batch.max_column_total_len, self.hidden_size)

        new_questions = new_questions.masked_scatter_(batch.question_mask.unsqueeze(-1), questions)
        new_tables = new_tables.masked_scatter_(batch.table_word_mask.unsqueeze(-1), tables)
        new_columns = new_columns.masked_scatter_(batch.column_word_mask.unsqueeze(-1), columns)
        new_tables_batch = new_tables_batch.masked_scatter_(batch.table_total_mask.unsqueeze(-1), tables)
        new_columns_batch = new_columns_batch.masked_scatter_(batch.column_total_mask.unsqueeze(-1), columns)
        return new_questions, new_tables, new_columns, new_tables_batch, new_columns_batch


class InputRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, cell='lstm', schema_aggregation='head+tail', share_lstm=False):
        super(InputRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = cell.upper()
        self.question_lstm = getattr(nn, self.cell)(self.input_size, self.hidden_size // 2, num_layers=1,
                                                    bidirectional=True, batch_first=True)
        self.schema_lstm = self.question_lstm if share_lstm else \
            getattr(nn, self.cell)(self.input_size, self.hidden_size // 2, num_layers=1, bidirectional=True,
                                   batch_first=True)
        self.schema_aggregation = schema_aggregation
        if self.schema_aggregation != 'head+tail':
            self.aggregation = PoolingFunction(self.hidden_size, self.hidden_size, method=schema_aggregation)

    def forward(self, input_dict, batch):
        """
            for question sentence, forward into a bidirectional LSTM to get contextual info and sequential dependence
            for schema phrase, extract representation for each phrase by concatenating head+tail vectors,
            batch.question_lens, batch.table_word_lens, batch.column_word_lens are used
        """
        questions, _ = rnn_wrapper(self.question_lstm, input_dict['question'], batch.question_lens, cell=self.cell)
        questions = questions.contiguous().view(-1, self.hidden_size)[
            lens2mask(batch.question_lens).contiguous().view(-1)]
        table_outputs, table_hiddens = rnn_wrapper(self.schema_lstm, input_dict['table'], batch.table_word_lens,
                                                   cell=self.cell)
        if self.schema_aggregation != 'head+tail':
            tables = self.aggregation(table_outputs, mask=batch.table_word_mask)
        else:
            table_hiddens = table_hiddens[0].transpose(0, 1) if self.cell == 'LSTM' else table_hiddens.transpose(0, 1)
            tables = table_hiddens.contiguous().view(-1, self.hidden_size)
        column_outputs, column_hiddens = rnn_wrapper(self.schema_lstm, input_dict['column'], batch.column_word_lens,
                                                     cell=self.cell)
        if self.schema_aggregation != 'head+tail':
            columns = self.aggregation(column_outputs, mask=batch.column_word_mask)
        else:
            column_hiddens = column_hiddens[0].transpose(0, 1) if self.cell == 'LSTM' else column_hiddens.transpose(0,
                                                                                                                    1)
            columns = column_hiddens.contiguous().view(-1, self.hidden_size)

        questions = questions.split(batch.question_lens.tolist(), dim=0)
        tables = tables.split(batch.table_lens.tolist(), dim=0)
        columns = columns.split(batch.column_lens.tolist(), dim=0)
        # dgl graph node feats format: q11 q12 ... t11 t12 ... c11 c12 ... q21 q22 ...

        outputs = [th for q_t_c in zip(questions, tables, columns) for th in q_t_c]

        outputs = torch.cat(outputs, dim=0)
        # transformer input format: bsize x max([q1 q2 ... t1 t2 ... c1 c2 ...]) x hidden_size
        q_outputs, s_outputs = [], []
        for q, t, c in zip(questions, tables, columns):
            q_zero_paddings = q.new_zeros((max(batch.question_lens.tolist()) - q.size(0), q.size(1)))
            s_zero_paddings = t.new_zeros((max((batch.table_lens + batch.column_lens).tolist())- t.size(0) - c.size(0), q.size(1)))

            q_cur_outputs = torch.cat([q, q_zero_paddings], dim=0)
            s_cur_outputs = torch.cat([t, c, s_zero_paddings], dim=0)

            q_outputs.append(q_cur_outputs)
            s_outputs.append(s_cur_outputs)
            # outputs.append(cur_outputs)
        q_outputs = torch.stack(q_outputs, dim=0)
        s_outputs = torch.stack(s_outputs, dim=0)
        return outputs, q_outputs, s_outputs
