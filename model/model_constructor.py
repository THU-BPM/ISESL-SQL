#coding=utf8
import torch
import torch.nn as nn
from model.model_utils import Registrable, PoolingFunction
from model.encoder.graph_encoder import Text2SQLEncoder
from model.decoder.sql_parser import SqlParser
from utils.example import Example
from model.encoder.graph_input import *
from utils.constants import RELATIONS, RELATIONS_INDEX
import copy
from asdls.sql.sql_transition_system import SelectColumnAction, SelectTableAction


class Text2SQL(nn.Module):
    def __init__(self, args, transition_system):
        super(Text2SQL, self).__init__()
        lazy_load = args.lazy_load if hasattr(args, 'lazy_load') else False
        self.input_layer = GraphInputLayerPLM(args, args.plm, args.gnn_hidden_size, dropout=args.dropout,
                                                        subword_aggregation=args.subword_aggregation,
                                                        schema_aggregation=args.schema_aggregation, lazy_load=lazy_load)
        self.encoder = Text2SQLEncoder(args)
        self.encoder2decoder = PoolingFunction(args.gnn_hidden_size, args.lstm_hidden_size, method='attentive-pooling')
        self.decoder = SqlParser(args, transition_system)
        self.args = args

    def forward(self, batch, cur_dataset, train=False, use_standard=False):

        graph_list = [ex.graph for ex in cur_dataset]
        new_graph_list = copy.deepcopy(graph_list)
        
        outputs, loss_token, edges, weight_m = self.input_layer(batch, use_standard)

        if self.args.optimize_graph:
            for n_b in range(len(graph_list)):
                new_graph_list[n_b].global_g = new_graph_list[n_b].global_g.to("cuda:0")
                new_graph_list[n_b].global_g.edata['weight'] = weight_m[n_b].reshape(-1)

        if self.args.semantic:
            new_graph_list = self.add_edge(new_graph_list, edges, cur_dataset)
            
        batch.graph = Example.graph_factory.batch_graphs(new_graph_list, "cuda:0", train=train)
        encodings, mask, gp_loss = self.encoder(batch, outputs)

        h0 = self.encoder2decoder(encodings, mask=mask)
        loss = self.decoder.score(encodings, mask, h0, batch)
        if self.args.token_task or self.args.schema_loss:
            loss = loss + loss_token
        return loss

    def get_exact_match_column_showed(self, cur_dataset, batch):

        column_part_match = RELATIONS.index('question-column-partialmatch')
        column_exact_match = RELATIONS.index('question-column-exactmatch')
        table_part_match = RELATIONS.index('question-table-partialmatch')
        table_exact_match = RELATIONS.index('question-table-exactmatch')
        column_value_match = RELATIONS.index('question-column-valuematch')
        columns = []
        tables = []
        for _index in range(len(cur_dataset)):
            data = cur_dataset[_index]
            keys = list(data.graph.local_edge_map.keys())
            cur_columns = []
            cur_tables = []
            for index in range(len(data.graph.local_edges)):
                edge = data.graph.local_edges[index]
                if edge == column_exact_match:
                    column = keys[index][1] - len(batch.questions[_index]) - len(batch.table_names[_index])
                    if column not in cur_columns:
                        cur_columns.append(column)
                if edge == table_exact_match:
                    table = keys[index][1] - len(batch.questions[_index])
                    if table not in cur_tables:
                        cur_tables.append(table)
            columns.append(cur_columns)
            tables.append(cur_tables)
        return columns, tables

    def check_edge(self, cur_dataset):
        for _index in range(len(cur_dataset)):
            data = cur_dataset[_index]
            for index in range(len(data.global_edges)):
                if data.global_edges[index] == 21:
                    print('already semantic')
                    import ipdb; ipdb.set_trace()

    def add_edge(self, graph_data, edges, cur_dataset):
        
        question_column_semanticmatch = RELATIONS.index('question-column-semanticmatch')
        column_question_semanticmatch = RELATIONS.index('column-question-semanticmatch')
        question_table_semanticmatch = RELATIONS.index('question-table-semanticmatch')
        table_question_semanticmatch = RELATIONS.index('table-question-semanticmatch')


        column_edges, table_edges = edges
        for _index in range(len(graph_data)):
            data = graph_data[_index]
            column_dict = cur_dataset[_index].column_dict
            table_dict = cur_dataset[_index].table_dict
            for edge in column_edges[_index]:
                column, question = edge
                if column in column_dict:
                    question = column_dict[column]
                index = data.global_edge_map[(column, question)]
                data.global_edges[index] = column_question_semanticmatch
                index = data.global_edge_map[(question, column)]
                data.global_edges[index] = question_column_semanticmatch
                if edge in data.local_edge_map:
                    index = data.local_edge_map[(column, question)]
                    data.local_edges[index] = column_question_semanticmatch
                    index = data.local_edge_map[(question, column)]
                    data.local_edges[index] = question_column_semanticmatch

            for edge in table_edges[_index]:
                table, question = edge
                if table in table_dict:
                    question = table_dict[table]
                index = data.global_edge_map[(table, question)]
                data.global_edges[index] = table_question_semanticmatch
                index = data.global_edge_map[(question, table)]
                data.global_edges[index] = question_table_semanticmatch
                if edge in data.local_edge_map:
                    index = data.local_edge_map[(table, question)]
                    data.local_edges[index] = table_question_semanticmatch
                    index = data.local_edge_map[(question, table)]
                    data.local_edges[index] = question_table_semanticmatch

        return graph_data

    def parse(self, batch, beam_size, cur_dataset, use_standard=False):
        """ This function is used for decoding, which returns a batch of [DecodeHypothesis()] * beam_size
        """
        graph_list = [ex.graph for ex in cur_dataset]
        new_graph_list = copy.deepcopy(graph_list)
        outputs, loss_token, edges, weight_m = self.input_layer(batch, use_standard, train=False)

        if self.args.optimize_graph:
            for n_b in range(len(graph_list)):
                new_graph_list[n_b].global_g = new_graph_list[n_b].global_g.to("cuda:0")
                new_graph_list[n_b].global_g.edata['weight'] = weight_m[n_b].reshape(-1)

        if self.args.semantic:
            new_graph_list = self.add_edge(new_graph_list, edges, cur_dataset)
        batch.graph = Example.graph_factory.batch_graphs(new_graph_list, "cuda:0", train=False)
        encodings, mask = self.encoder(batch, outputs)
        h0 = self.encoder2decoder(encodings, mask=mask)
        hyps = []
        table_in_sql, column_in_sql = [], []
        for i in range(len(batch)):
            """ 
                table_mappings and column_mappings are used to map original database ids to local ids,
                while reverse_mappings perform the opposite function, mapping local ids to database ids
            """
            table_in_sql_batch, column_in_sql_batch = [], []
            hyps_item = self.decoder.parse(encodings[i:i+1], mask[i:i+1], h0[i:i+1], batch, beam_size)
            for action in hyps_item[0].actions:
                if isinstance(action, SelectColumnAction):
                    if action.token not in column_in_sql_batch:
                        column_in_sql_batch.append(action.token)
                if isinstance(action, SelectTableAction):
                    if action.token not in table_in_sql_batch:
                        table_in_sql_batch.append(action.token)
            hyps.append(hyps_item)
            table_in_sql.append(table_in_sql_batch)
            column_in_sql.append(column_in_sql_batch)

        return hyps, (batch.table_used, batch.column_used), (table_in_sql, column_in_sql)

    def pad_embedding_grad_zero(self, index=None):
        """ 
            For glove.42B.300d word vectors, gradients for <pad> symbol is always 0;
            Most words (starting from index) in the word vocab are also fixed except most frequent words
        """
        self.input_layer.pad_embedding_grad_zero(index)
