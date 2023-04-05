#!/usr/bin/python
#encoding=utf-8

import torch
import kaldiio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from utils.tools import load_wave, F_Mel, make_context, skip_feat, spec_augment, data_enhancement

audio_conf = {"sample_rate":16000, 'window_size':0.025, 'window_stride':0.01, 'window': 'hamming'}

class Vocab(object):
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        self.word2index = {"blank": 0, "UNK": 1}
        self.index2word = {0: "blank", 1: "UNK"}
        # TODO
        self.consonant2index = {}
        self.index2consonant = {}
        self.vowel2index = {}
        self.index2vowel = {}
        self.word2count = {}
        self.n_words = 2
        self.read_lang()

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def read_lang(self):
        print("Reading vocabulary from {}".format(self.vocab_file))
        with open(self.vocab_file, 'r') as rf:
            line = rf.readline()
            while line:
                line = line.strip().split(' ')
                if len(line) > 1:
                    sen = ' '.join(line[1:])
                else:
                    sen = line[0]
                self.add_sentence(sen)
                line = rf.readline()
        print("Vocabulary size is {}".format(self.n_words))
    
    
class SpeechDataset(Dataset):
    def __init__(self, vocab, wavscp_path, scp_path, lab_path, trans_path, opts, train=False):
        self.vocab = vocab
        self.wavscp_path = wavscp_path
        self.scp_path = scp_path
        self.lab_path = lab_path
        self.trans_path = trans_path
        self.left_ctx = opts.left_ctx
        self.right_ctx = opts.right_ctx
        self.n_skip_frame = opts.n_skip_frame
        self.n_downsample = opts.n_downsample
        self.feature_type = opts.feature_type
        self.mel = opts.mel
        self.train = train
        self.use_specaug = opts.use_specaug
        self.use_textaug = opts.use_textaug
        self.mutation_prob = opts.mutation_prob
        self.enhancement_type = opts.enhancement_type
        self.phone_num = opts.phone_num

        print("phone_num", self.phone_num)
        
        if self.train and self.use_specaug:
            print("use specaug")
        else:
            print("not use specaug")
        
        if self.train and self.use_textaug:
            print("use textaug")
        else:
            print("not use textaug")
        
        self.process_feature_label()
    
    def process_feature_label(self):
        wav_path_dict = []
        feat_path_dict = []
        #read the wav path
        with open(self.wavscp_path, 'r') as rf:
            line = rf.readline()
            while line:
                utt, path = line.strip().split(' ')
                wav_path_dict.append((utt, path))
                line = rf.readline()
        
        #read the ark path
        with open(self.scp_path, 'r') as rf:
            line = rf.readline()
            while line:
                utt, path = line.strip().split(' ')
                feat_path_dict.append((utt, path))
                line = rf.readline()
        
       	#read the label
        label_dict = dict()
        with open(self.lab_path, 'r') as rf:
            line = rf.readline()
            while line:
                utt, label = line.strip().split(' ', 1)
                label_dict[utt] = [self.vocab.word2index[c] if c in self.vocab.word2index else self.vocab.word2index['UNK'] for c in label.split()]
                
                line = rf.readline() 
                
        #read the transcript
        trans_dict = dict()
        with open(self.trans_path, 'r') as rf:
            line = rf.readline()
            while line:
                utt, trans = line.strip().split(' ', 1)
                trans_dict[utt] = [self.vocab.word2index[c] if c in self.vocab.word2index else self.vocab.word2index['UNK'] for c in trans.split()]
                line = rf.readline() 
        
        assert len(feat_path_dict) == len(label_dict)
        print("Reading %d lines from %s" % (len(label_dict), self.lab_path))
        
        self.item = []
        for i in range(len(feat_path_dict)):
            utt, wav_path = wav_path_dict[i]
            utt, feat_path = feat_path_dict[i]
            self.item.append((wav_path, feat_path, label_dict[utt], trans_dict[utt], utt))

    def __getitem__(self, idx):
        wav_path, feat_path, label, trans, utt = self.item[idx]
            
        feat = kaldiio.load_mat(feat_path)
            
        if self.train:
            if self.use_specaug:
                feat = spec_augment(feat)
            if self.use_textaug:
                trans = [data_enhancement(phone=tran, mutation_prob=self.mutation_prob, enhancement_type=self.enhancement_type, phone_num=self.phone_num, vocab=self.vocab) for tran in trans]
                trans = sum(trans, [])
        
        feat = skip_feat(make_context(feat, self.left_ctx, self.right_ctx), self.n_skip_frame)
        seq_len, dim = feat.shape
        
        #pad_len = self.n_downsample - seq_len % self.n_downsample
        #feat = np.vstack([feat, np.zeros((pad_len, dim))])
        
        if self.mel:
            return (F_Mel(torch.from_numpy(feat), audio_conf), label)
        else:
            return (torch.from_numpy(feat), torch.LongTensor(label),torch.LongTensor(trans), utt)

    def __len__(self):
        return len(self.item) 
    
'''
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, 
                                                    collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
subclass of DataLoader and rewrite the collate_fn to form batch
'''

class SpeechDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(SpeechDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self.create_input

    def create_input(self, batch):
        inputs_max_length = max(x[0].size(0) for x in batch)
        feat_size = batch[0][0].size(1)
        targets_max_length = max(x[1].size(0) for x in batch)
        
        trans_max_length = max(x[2].size(0) for x in batch)
        
        batch_size = len(batch)
        batch_data = torch.zeros(batch_size, inputs_max_length, feat_size) 
        batch_label = torch.zeros(batch_size, targets_max_length)
        batch_trans = torch.zeros(batch_size, trans_max_length)
        input_sizes = torch.zeros(batch_size)
        target_sizes = torch.zeros(batch_size)
        trans_sizes = torch.zeros(batch_size)
        utt_list = []

        for x in range(batch_size):
            feature, label, trans, utt = batch[x]
            feature_length = feature.size(0)
            label_length = label.size(0)
            trans_length = trans.size(0)

            batch_data[x].narrow(0, 0, feature_length).copy_(feature)
            batch_label[x].narrow(0, 0, label_length).copy_(label)
            batch_trans[x].narrow(0, 0, trans_length).copy_(trans)
            
            input_sizes[x] = feature_length / inputs_max_length
            target_sizes[x] = label_length
            trans_sizes[x] = trans_length
            utt_list.append(utt)
        return batch_data.float(), input_sizes.float(), batch_label.long(), target_sizes.long(), batch_trans.long(), trans_sizes.long(),utt_list 

if __name__ == '__main__':
    dev_dataset = SpeechDataset()
    dev_dataloader = SpeechDataLoader(dev_dataset, batch_size=2, shuffle=True)
    
    import visdom
    viz = visdom.Visdom(env='fan')
    for i in range(1):
        show = dev_dataset[i][0].transpose(0, 1)
        text = dev_dataset[i][1]
        for num in range(len(text)):
            text[num] = dev_dataset.int2class[text[num]]
        text = ' '.join(text)
        opts = dict(title=text, xlabel='frame', ylabel='spectrum')
        viz.heatmap(show, opts = opts)
