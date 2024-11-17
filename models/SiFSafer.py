import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from s3prl.nn import S3PRLUpstream, Featurizer
from peft import LoraConfig, get_peft_model
from .LCNN import BLSTMLayer

class Model(nn.Module):
    def __init__(self, args, sr_model, device):
        super().__init__()
        self.device = device
        
        #LSTM parameters 
        input_size = 256
        hidden_size = 384

        ####
        # create network wav2vec 2.0
        ####
        target_modules = []
        for i in range(args['lora_layers'][0], args['lora_layers'][1]):
            for modules in args['inject_modules']:
                target_modules.append('layers.{}.self_attn.{}'.format(i, modules))

        config = LoraConfig(
        target_modules=target_modules,
        bias='none')

        self.upstream_model = S3PRLUpstream(name=sr_model) 
        for p in self.parameters():
            p.requires_grad = False
        self.upstream_model_tuning = S3PRLUpstream(name=sr_model)
        self.upstream_model_tuning = get_peft_model(self.upstream_model_tuning, config)
        # for p in self.parameters():
        #     p.requires_grad = False
        # for p in self.upstream_model.upstream.model.feature_extractor.parameters():
        #     p.requires_grad = True
        # for i in range(0, 5):
        #     for p in self.upstream_model.upstream.model.encoder.layers[i].parameters():
        #         p.requires_grad = True
        if args["tuning_layers"][0] == -1:
            layers_id_tuning = [self.upstream_model_tuning.num_layers-1]
        elif args["tuning_layers"][1] == -1:
            layers_id_tuning = list(range(args["tuning_layers"][0],self.upstream_model_tuning.num_layers))
        else:
            layers_id_tuning = list(range(args["tuning_layers"][0],args["tuning_layers"][1]))
        print("tuning layers id:{}".format(layers_id_tuning))
        self.no_featurizer_tuning = True if len(layers_id_tuning)==1 else False
        if self.no_featurizer_tuning:
            self.layer_tuning = layers_id_tuning[0]
        else:
            self.featurizer_tuning = Featurizer(self.upstream_model, layers_id_tuning)

        if args["layers"][0] == -1:
            layers_id = [self.upstream_model.num_layers-1]
        elif args["layers"][1] == -1:
            layers_id = list(range(args["layers"][0],self.upstream_model.num_layers))
        else:
            layers_id = list(range(args["layers"][0],args["layers"][1]))
        print("layers id:{}".format(layers_id))
        self.no_featurizer = True if len(layers_id)==1 else False
        if self.no_featurizer:
            self.layer = layers_id[0]
        else:
            self.featurizer = Featurizer(self.upstream_model, layers_id)

        # self.LL = nn.Linear(1024, 256)
        # self.selu = nn.SELU(inplace=True)

        # self.LL_tuning = nn.Linear(1024, 256)
        # self.selu_tuning = nn.SELU(inplace=True)
        self.proj = nn.Linear(1024, input_size)
        self.proj_tuning = nn.Linear(1024, input_size)
        self.batchnorm = nn.BatchNorm1d(num_features=200)
        self.selu = nn.SELU()

        self.lstm = nn.Sequential(
                BLSTMLayer(input_size, hidden_size),
                BLSTMLayer(hidden_size, hidden_size)
        )    
        self.lstm_tuning = nn.Sequential(
                BLSTMLayer(input_size, hidden_size),
                BLSTMLayer(hidden_size, hidden_size)
        )

        self.fusion_weights = nn.Parameter(torch.ones(2, requires_grad=True))
        # torch.nn.init.xavier_uniform_(self.fusion_weights)
        self.out_layer = nn.Linear(hidden_size, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_len):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        all_hs, all_hs_len = self.upstream_model(x, x_len)
        all_hs_tuning, all_hs_len_tuning = self.upstream_model_tuning(x, x_len)

        if self.no_featurizer:
            hs = all_hs[self.layer]
        else:
            hs, hs_len = self.featurizer(all_hs, all_hs_len) 
        if self.no_featurizer_tuning:
            hs_tuning = all_hs_tuning[self.layer_tuning]
        else:
            hs_tuning, hs_len_tuning = self.featurizer_tuning(all_hs_tuning, all_hs_len_tuning)

        x = self.proj(hs)
        x_tuning = self.proj_tuning(hs_tuning)
        x = self.batchnorm(x)
        x_tuning = self.batchnorm(x_tuning)
        
        x = self.selu(x)
        x_tuning = self.selu(x_tuning)

        lstm_out = self.lstm(x)
        lstm_out = lstm_out[:,-1,:]
        lstm_out_tuning = self.lstm_tuning(x_tuning)
        lstm_out_tuning = lstm_out_tuning[:,-1,:]

        lstm_out_tmp = lstm_out.view(-1)
        lstm_out_tuning_tmp = lstm_out_tuning.view(-1)
        
        norm_weights = F.softmax(self.fusion_weights, dim=-1)
        x_fusion = torch.matmul(norm_weights, torch.vstack([lstm_out_tmp, lstm_out_tuning_tmp]))
        x_fusion = x_fusion.view(lstm_out.shape)
        output = self.out_layer(x_fusion)
        output = self.sigmoid(output)
        
        return output
