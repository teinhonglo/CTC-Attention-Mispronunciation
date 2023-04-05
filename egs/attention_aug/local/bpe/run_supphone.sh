#!/bin/bash

stage=1

# bpe model config
train_text=/share/nas167/fuann/asr/librispeech/s5/data/local/lm/librispeech-lm-corpus-76M-supphone.txt
vocab_size=80
exp_dir=bpe_model/tokenizer-ls76M-supphone-80

. ./utils/parse_options.sh

if [ $# -ne 3 ]; then
    echo "Usage: <kaldi-text-phone> <kaldi-text-group-phone> <kaldi-text-supphone>"
    echo "  e.g. $0 /share/nas167/fuann/asr/gop_speechocean762/s5/data/train/text_phone /share/nas167/fuann/asr/gop_speechocean762/s5/data/train/text_group_phone /share/nas167/fuann/asr/gop_speechocean762/s5/data/train/text_supphone"
    echo
    echo "  BPE training options:"
    echo "    --train-text /share/nas167/fuann/asr/librispeech/s5/data/local/lm/librispeech-lm-corpus-76M-supphone.txt"
    echo "    --vocab-size 100"
    echo "    --exp-dir bpe_model/tokenizer-ls76M-supphone-100"
    echo
    echo "  Input:"
    echo "    <kaldi-text-phone>"
    echo "       000010011 W IY K AO L IH T B EH R"
    echo "    <kaldi-text-group-phone>"
    echo "       000010011 WIY KAOL IHT BEHR"
    echo "  Output:"
    echo "    <kaldi-text-supphone>"
    echo "       000010011 W IY K AO L IHT IHT B EHR EHR"
    echo
    exit 0
fi

kaldi_text_phone=$1
kaldi_text_group_phone=$2
kaldi_text_supphone=$3

if [ $stage -le 1 ]; then

    [ ! -d $exp_dir ] && mkdir -p $exp_dir
    python local/bpe/huggingface_bpe_train.py \
        --train-files $train_text \
        --exp-dir $exp_dir \
        --vocab-size $vocab_size || exit 1
fi

if [ $stage -le 2 ]; then
    echo "Encoding $kaldi_text ..."
    python local/bpe/huggingface_bpe_test.py \
        --bpe-model $exp_dir/model.json \
        --kaldi-text $kaldi_text_group_phone \
        --kaldi-text-bpe ${kaldi_text_group_phone}_bpe || exit 1
    echo "DONE"
fi

if [ $stage -le 3 ]; then

    echo "Align supphone ..."
    python local/bpe/align_supphone.py \
        --text-phone $kaldi_text_phone \
        --text-group-phone-bpe ${kaldi_text_group_phone}_bpe \
        --text-supphone $kaldi_text_supphone
fi
