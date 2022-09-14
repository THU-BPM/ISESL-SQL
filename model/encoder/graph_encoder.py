#coding=utf8
import torch
import torch.nn as nn
from model.encoder.rgatsql import RGATSQL
from model.encoder.graph_output import *
from model.model_utils import Registrable


class Text2SQLEncoder(nn.Module):

    def __init__(self, args):
        super(Text2SQLEncoder, self).__init__()
        self.hidden_layer = RGATSQL(args)
        self.output_layer = Registrable.by_name(args.output_model)(args)
    def forward(self, batch, x):
        outputs = self.hidden_layer(x, batch)
        return self.output_layer(outputs, batch)
