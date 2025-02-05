#!/usr/bin/python
#encoding=utf-8

import math
import torch
import torch.nn as nn
import editdistance as ed
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

np.random.seed(1234)
torch.random.manual_seed(1234)


__author__ = "Ruchao Fan"

class BatchRNN(nn.Module):
    """
    Add BatchNorm before rnn to generate a batchrnn layer
    """
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, 
                    bidirectional=False, batch_norm=True, dropout=0.1):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = nn.BatchNorm1d(input_size) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                                bidirectional=bidirectional, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        if self.batch_norm is not None:
            x = x.transpose(-1, -2)
            x = self.batch_norm(x)
            x = x.transpose(-1, -2)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        #self.rnn.flatten_parameters()
        return x

class LayerCNN(nn.Module):
    """
    One CNN layer include conv2d, batchnorm, activation and maxpooling
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, pooling_size=None, 
                        activation_function=nn.ReLU, batch_norm=True, dropout=0.1):
        super(LayerCNN, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channel) if batch_norm else None
        self.activation = activation_function(inplace=True)

        if pooling_size is not None:
            self.pooling = nn.MaxPool2d(pooling_size)
        else:
            self.pooling = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.dropout(x)
        return x

class CTC_Model(nn.Module):
    def __init__(self, add_cnn=False, cnn_param=None, rnn_param=None, num_class=39, drop_out=0.1, opts=None):
        """
        add_cnn   [bool]:  whether add cnn in the model
        cnn_param [dict]:  cnn parameters, only support Conv2d i.e.
            cnn_param = {"layer":[[(in_channel, out_channel), (kernel_size), (stride), (padding), (pooling_size)],...], 
                            "batch_norm":True, "activate_function":nn.ReLU}
        rnn_param [dict]:  rnn parameters i.e.
            rnn_param = {"rnn_input_size":201, "rnn_hidden_size":256, ....}
        num_class  [int]:  the number of modelling units, add blank to be the number of classes
        drop_out [float]:  drop_out rate for all
        """
        super(CTC_Model, self).__init__()
        self.add_cnn = add_cnn
        self.cnn_param = cnn_param
        if rnn_param is None or type(rnn_param) != dict:
            raise ValueError("rnn_param need to be a dict to contain all params of rnn!")
        self.rnn_param = rnn_param
        self.num_class = num_class
        self.num_directions = 2 if rnn_param["bidirectional"] else 1
        self.drop_out = drop_out
        
        cnns = []
        activation = cnn_param["activate_function"]
        batch_norm = cnn_param["batch_norm"]
        rnn_input_size = rnn_param["rnn_input_size"]
        cnn_layers = cnn_param["layer"]

        for n in range(len(cnn_layers)):
            in_channel = cnn_layers[n][0][0]
            out_channel = cnn_layers[n][0][1]
            kernel_size = cnn_layers[n][1]
            stride = cnn_layers[n][2]
            padding = cnn_layers[n][3]
            pooling_size = cnn_layers[n][4]
            
            cnn = LayerCNN(in_channel, out_channel, kernel_size, stride, padding, pooling_size, 
                            activation_function=activation, batch_norm=batch_norm, dropout=drop_out)
            cnns.append(('%d' % n, cnn))
            
            rnn_input_size = int(math.floor((rnn_input_size+2*padding[1]-kernel_size[1])/stride[1])+1)            

        self.conv = nn.Sequential(OrderedDict(cnns))
        rnn_input_size *= out_channel
        
        rnns = []
        rnn_hidden_size = rnn_param["rnn_hidden_size"]
        rnn_type = rnn_param["rnn_type"]
        rnn_layers = rnn_param["rnn_layers"]
        bidirectional = rnn_param["bidirectional"]
        batch_norm = rnn_param["batch_norm"]
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, 
                            bidirectional=bidirectional, dropout=drop_out, batch_norm=False)
        rnns.append(('0', rnn))
        for i in range(rnn_layers-1):
            rnn = BatchRNN(input_size=self.num_directions*rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, 
                                bidirectional=bidirectional, dropout=drop_out, batch_norm=batch_norm)
            rnns.append(('%d' % (i+1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        
        ## Character_embedding
        self.embeds = nn.Embedding(opts.phone_num, opts.char_embed_size)
        self.lstm_embeds = nn.LSTM(opts.char_embed_size, rnn_hidden_size, batch_first=True, bidirectional=True)
        self.score = nn.Linear(self.num_directions * rnn_hidden_size, self.num_directions * rnn_hidden_size, bias=False)

        if batch_norm:
            self.fc = nn.Sequential(nn.BatchNorm1d(self.num_directions*rnn_hidden_size),
                                nn.Linear(self.num_directions*rnn_hidden_size * 2, num_class, bias=False),)
        else:
            self.fc = nn.Linear(self.num_directions*rnn_hidden_size * 2, num_class, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x, x1):
        """
        x: audio:
        x1: sentence
        """
        # x: batch_size * max_seq_length * feat_size
        # torch.Size([64, 452, 243])
        x = self.conv(x.unsqueeze(1))
        # torch.Size([64, 32, 226, 61])
        
        x = x.transpose(1, 2).contiguous()
        # torch.Size([64, 226, 32, 61])
        sizes = x.size()
        x = x.view(sizes[0], sizes[1], sizes[2]*sizes[3])
        # torch.Size([64, 226, 1952])
        x = x.transpose(0,1).contiguous()
        # torch.Size([226, 64, 1952])

        x = self.rnns(x)
        # torch.Size([226, 64, 768])
        x = x.transpose(0,1)
        # torch.Size([64, 226, 768])
        
        x1 = self.embeds(x1)

        # [batch,Char_num,Hidden_unit]
        # value
        # torch.Size([64, 67, 768])
        x1 , _ = self.lstm_embeds(x1)
        
        # key
        # torch.Size([64, 67, 768])
        key = self.score(x1)
         
        attn_score = torch.bmm(x, key.transpose(1, 2))
        attn_max, _ = torch.max(attn_score, dim=-1, keepdim=True) 
        exp_score = torch.exp(attn_score - attn_max)

        attn_weights = exp_score
        weights_denom = torch.sum(attn_weights, dim=-1, keepdim=True)   
        attn_weights = attn_weights / (weights_denom + 1e-30)
        c = torch.bmm(attn_weights, x1)
        
        # query and context concat
        out1 = torch.cat((x, c), -1)
        #out1 = x + c
        # out1: torch.Size([64, 226, 1536])
        x = out1.transpose(0,1).contiguous()
        # torch.Size([226, 64, 1536])

        seq_len, batch, _ = x.size()
        x = x.view(seq_len * batch, -1)
        x = self.fc(x)
        x = x.view(seq_len, batch, -1)
        out = self.log_softmax(x)

        return out
        
  
    def compute_wer(self, index, input_sizes, targets, target_sizes):
        batch_errs = 0
        batch_tokens = 0
        for i in range(len(index)):
            label = targets[i][:target_sizes[i]]
            pred = []
            for j in range(len(index[i][:input_sizes[i]])):
                if index[i][j] == 0:
                    continue
                if j == 0:
                    pred.append(index[i][j])
                if j > 0 and index[i][j] != index[i][j-1]:
                    pred.append(index[i][j])
            batch_errs += ed.eval(label, pred)
            batch_tokens += len(label)
            # total_pred.append(pred)
            
        return batch_errs, batch_tokens

    def add_weights_noise(self):
        for param in self.parameters():
            weight_noise = param.data.new(param.size()).normal_(0, 0.075).type_as(param.type())
            param = torch.nn.parameter.Parameter(param.data + weight_noise)

    @staticmethod
    def save_package(model, optimizer=None, decoder=None, epoch=None, loss_results=None, dev_loss_results=None, dev_cer_results=None):
        package = {
                'rnn_param': model.rnn_param,
                'add_cnn': model.add_cnn,
                'cnn_param': model.cnn_param,
                'num_class': model.num_class,
                '_drop_out': model.drop_out,
                'state_dict': model.state_dict()
                }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if decoder is not None:
            package['decoder'] = decoder
        if epoch is not None:
            package['epoch'] = epoch
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['dev_loss_results'] = dev_loss_results
            package['dev_cer_results'] = dev_cer_results
        return package

if __name__ == '__main__':
    model = CTC_Model(add_cnn=True, cnn_param={"batch_norm":True, "activativate_function":nn.ReLU, "layer":[[(1,32), (3,41), (1,2), (0,0), None],
                            [(32,32), (3,21), (2,2), (0,0), None]]}, num_class=48, drop_out=0)
    for idx, m in CTC_Model.modules():
        print(idx, m) 
    
