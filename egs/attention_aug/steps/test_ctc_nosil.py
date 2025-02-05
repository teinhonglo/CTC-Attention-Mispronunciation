#!/usr/bin/python
#encoding=utf-8

import os
import time
import sys
import torch
import yaml
import argparse
import torch.nn as nn

sys.path.append('./')
from models.model_ctc import *
from utils.ctcDecoder import GreedyDecoder, BeamDecoder
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader
from steps.train_ctc import Config

parser = argparse.ArgumentParser()
parser.add_argument('--conf', help='conf file for training')
parser.add_argument('--affix', default="")

def add_affix(origin_path, affix):
    info = origin_path.split("/")
    info[-2] += affix
    new_path = "/".join(info)
    return new_path

def test():
    args = parser.parse_args()
    affix = args.affix
    try:
        conf = yaml.safe_load(open(args.conf,'r'))
    except:
        print("Config file not exist!")
        sys.exit(1)    
    
    opts = Config()
    for k,v in conf.items():
        setattr(opts, k, v)
        print('{:50}:{}'.format(k, v))

    use_cuda = opts.use_gpu
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
    
    model_path = os.path.join(opts.checkpoint_dir, opts.exp_name, 'ctc_best_model.pkl')
    package = torch.load(model_path)
    
    rnn_param = package["rnn_param"]
    add_cnn = package["add_cnn"]
    cnn_param = package["cnn_param"]
    num_class = package["num_class"]
    feature_type = package['epoch']['feature_type']
    n_feats = package['epoch']['n_feats']
    drop_out = package['_drop_out']
    mel = opts.mel

    beam_width = opts.beam_width
    lm_alpha = opts.lm_alpha
    decoder_type =  opts.decode_type
    vocab_file = opts.vocab_file
    
    vocab = Vocab(vocab_file)
    opts.test_wavscp_path = add_affix(opts.test_wavscp_path, affix)
    opts.test_scp_path = add_affix(opts.test_scp_path, affix)
    opts.test_lab_path = add_affix(opts.test_lab_path, affix)
    opts.test_trans_path = add_affix(opts.test_trans_path, affix)
    test_dataset = SpeechDataset(vocab, opts.test_wavscp_path, opts.test_scp_path, opts.test_lab_path, opts.test_trans_path, opts)
    test_loader = SpeechDataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers, pin_memory=False)
    
    model = CTC_Model(rnn_param=rnn_param, add_cnn=add_cnn, cnn_param=cnn_param, num_class=num_class, drop_out=drop_out, opts=opts)
    model.to(device)
    model.load_state_dict(package['state_dict'])
    model.eval()
    
    
    if decoder_type == 'Greedy':
        decoder  = GreedyDecoder(vocab.index2word, space_idx=-1, blank_index=0)
    else:
        decoder = BeamDecoder(vocab.index2word, beam_width=beam_width, blank_index=0, space_idx=-1, lm_path=opts.lm_path, lm_alpha=opts.lm_alpha) 

    save_dir = os.path.join(opts.checkpoint_dir, opts.exp_name)    
    w1 = open(save_dir + "/decode_seq" + affix,'w+')
    w2 = open(save_dir + "/human_seq" + affix,'w+') 
    total_wer = 0
    total_cer = 0
    start = time.time()
    with torch.no_grad():
        for data in test_loader:
            inputs, input_sizes, targets, target_sizes, trans, trans_sizes, utt_list = data 
            inputs = inputs.to(device)
            trans = trans.to(device)
            #rnput_sizes = input_sizes.to(device) 
            #target = target.to(device)
            #target_sizes = target_sizes.to(device)
            
            probs = model(inputs,trans)

            max_length = probs.size(0)
            input_sizes = (input_sizes * max_length).long()

            probs = probs.cpu()
            decoded = decoder.decode(probs, input_sizes.numpy().tolist())
            
            targets, target_sizes = targets.numpy(), target_sizes.numpy()
            labels = []
            for i in range(len(targets)):
                label = [ vocab.index2word[num] for num in targets[i][:target_sizes[i]]]
                labels.append(' '.join(label))
                
                
            ## compute with out sil     
            decoded_nosil = []
            labesl_nosil = []
            for i in range(len(labels)):
                hyp = decoded[i].split(" ")
                ref = labels [i].split(" ")
                ref_precess = [ i   for i in ref if(i != "sil")  ]
                hyp_precess = [ i   for i in hyp if(i != "sil")  ]
                labesl_nosil.append(' '.join(ref_precess))
                decoded_nosil.append(' '.join(hyp_precess))    
            for x in range(len(targets)):
                w2.write(utt_list[x] + " " + labesl_nosil[x] + "\n")
                w1.write(utt_list[x] + " " + decoded_nosil[x] + "\n")    
            
            wer = 0
            for x in range(len(targets)):
                print("origin : " + labesl_nosil[x])
                print("decoded: " + decoded_nosil[x])
            for x in range(len(labesl_nosil)):
                wer += decoder.wer(decoded_nosil[x], labesl_nosil[x])
                decoder.num_word += len(labesl_nosil[x].split())
            total_wer += wer
            
            ##  

            #for x in range(len(targets)):
            #    print("origin : " + labels[x])
            #    print("decoded: " + decoded[x])
            #cer = 0
            #wer = 0
            #for x in range(len(labels)):
            #    cer += decoder.cer(decoded[x], labels[x])
            #    wer += decoder.wer(decoded[x], labels[x])
            #    decoder.num_word += len(labels[x].split())
            #    decoder.num_char += len(labels[x])
            #total_cer += cer
            #total_wer += wer
            
            ##
    print("total_error:",total_wer)
    print("total_phoneme:",decoder.num_word)
    WER = (float(total_wer) / decoder.num_word)*100
    print("Phoneme error rate on test set: %.4f" % WER)
    end = time.time()
    time_used = (end - start) / 60.0
    print("time used for decode %d sentences: %.4f minutes." % (len(test_dataset), time_used))
    w1.close()
    w2.close()
if __name__ == "__main__":
    test()
