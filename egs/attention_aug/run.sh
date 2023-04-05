#!/usr/bin/env bash

#Author: Kaiqi Fu, JinHong Lin
. ./path.sh

stage=0
stop_stage=1000
l2arctic_dir="/share/corpus/l2arctic_release_v4.0" 
timit_dir='/share/corpus/TIMIT'
phoneme_map='60-39'
feat_dir='data'                            #dir to save feature
feat_type='fbank'                          #fbank, mfcc, spectrogram
config_file='conf/tuning/ctc_cnn_fbank_vc10.yaml'

. ./parse_options.sh
set -euo pipefail

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Step 0: Data Preparation ..."

    local/timit_data_prep.sh $timit_dir $phoneme_map || exit 1;
    python3 local/l2arctic_prep.py --l2_path=$l2arctic_dir --save_path=${feat_dir}/l2_train  
    python3 local/l2arctic_prep.py --l2_path=$l2arctic_dir --save_path=${feat_dir}/l2_dev  
    python3 local/l2arctic_prep.py --l2_path=$l2arctic_dir --save_path=${feat_dir}/l2_test
    mv ${feat_dir}/l2_dev ${feat_dir}/dev  
    mv ${feat_dir}/l2_test ${feat_dir}/test
    local/timit_l2_merge.sh ${feat_dir}/train_timit ${feat_dir}/l2_train ${feat_dir}/train
    rm -rf l2_train train_timit

    python3 steps/get_model_units.py $feat_dir/train/phn_text
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    #python3 steps/get_model_units.py $feat_dir/train/phn_text
    echo "Step 1: Feature Extraction..."
    steps/make_feat.sh $feat_type $feat_dir || exit 1;
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "Step 2: LM Model Training..."
    steps/train_lm.sh $feat_dir || exit 1;
fi

checkpoint_dir=`grep "checkpoint_dir" $config_file | cut -d":" -f2 | sed "s/['|\/| ]//g"`
exp_dir=`grep "exp_name" $config_file | cut -d":" -f2 | sed "s/['|\/| ]//g"`

save_dir=$checkpoint_dir/$exp_dir

if [ -d $save_dir ]; then
    echo "Directory $save_dir is already existed. This is OK if you know what you want to do."
    sleep 5
fi

echo $save_dir
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    echo "Step 3: Acoustic Model(CTC) Training..."
    python3 steps/train_ctc.py --conf $config_file || exit 1;
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    echo "Step 4: Decoding..."
    python3 steps/test_ctc_nosil.py --conf $config_file --affix _comm > $save_dir/decode_comm.log
    python3 steps/test_ctc_nosil.py --conf $config_file --affix _others > $save_dir/decode_others.log
    python3 steps/test_ctc_nosil.py --conf $config_file > $save_dir/decode.log
    tail $save_dir/decode.log
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    echo "Step 5: Evaluating..."
    for affix in _comm _others; do
        sort -k1,1 data/test$affix/transcript_phn_text | sed "s/ sil//g" > result/ref_seq
        cp $save_dir/decode_seq$affix result/decode_seq
        cp $save_dir/human_seq$affix result/human_seq  
        cd result/
        ./mdd_result.sh > ../$save_dir/result${affix}.log
        cd -
    done

    sort -k1,1 data/test/transcript_phn_text | sed "s/ sil//g" > result/ref_seq
    cp $save_dir/decode_seq  $save_dir/human_seq result/
    cd result/
    ./mdd_result.sh > ../$save_dir/result.log
    cd -
    cat $save_dir/result.log
fi


if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    echo "Step 5: Evaluating (Ours) ..."
    for affix in _comm _others; do
        sort -k1,1 data/test$affix/transcript_phn_text | sed "s/ sil//g" > result/ref_seq
        cp $save_dir/decode_seq$affix result/decode_seq
        cp $save_dir/human_seq$affix result/human_seq  
        ./local/compute_capt_accuracy.sh --should_say result/ref_seq --actual_say result/human_seq --predict_say result/decode_seq --capt_dir $save_dir/capt${affix}
    done

    sort -k1,1 data/test/transcript_phn_text | sed "s/ sil//g" > result/ref_seq
    cp $save_dir/decode_seq  $save_dir/human_seq result/
    ./local/compute_capt_accuracy.sh --should_say result/ref_seq --actual_say result/human_seq --predict_say result/decode_seq --capt_dir $save_dir/capt
fi
