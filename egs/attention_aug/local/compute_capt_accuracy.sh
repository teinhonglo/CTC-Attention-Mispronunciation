#!/usr/bin/env bash

#. ./cmd.sh
. ./path.sh

stage=0

#phonemap="conf/phones.60-48-39.map"
should_say="data/test_l2_should_say/text"
actual_say="data/test_l2/text"
predict_say=
capt_dir=
ctm_file=

. parse_options.sh || exit 1;

if [ -z $predict_say ] || [ -z $capt_dir ]; then
    echo "data_set=test_l2; dict=data/lang_1char/train_tr_units.txt; dir=exp/chain/cnn_tdnn1c_sp/decode_4grl2mix05_test_l2"
	echo 'local/compute_capt_accuracy.sh --should_say data/${data_set}_should_say/text --actual_say data/${data_set}/text --predict_say ${dir}/scoring/4.tra --capt-dir ${dir}/capt --ctm-file ${dir}/ctm/ctm'
	exit 0;
fi

if [ ! -d $capt_dir ]; then
    mkdir -p $capt_dir;
fi

# Map reference to 39 phone classes
if [ $stage -le 0 ]; then
    #cat $should_say | local/timit_norm_trans.pl -i - -m $phonemap -from 48 -to 39 > $capt_dir/should_say.txt
    #cat $actual_say | local/timit_norm_trans.pl -i - -m $phonemap -from 48 -to 39 > $capt_dir/actual_say.txt
    #cat $predict_say | local/timit_norm_trans.pl -i - -m $phonemap -from 48 -to 39 > $capt_dir/predict_say.txt
	
	# remove sil or not?
	#for fn in $capt_dir/should_say.txt $capt_dir/actual_say.txt $capt_dir/predict_say.txt; do
	#	sed -i "s/sil//g" $fn
	#done
    cat $should_say > $capt_dir/should_say.txt
    cat $actual_say > $capt_dir/actual_say.txt
    cat $predict_say > $capt_dir/predict_say.txt
fi

# Alignment
if [ $stage -le 1 ]; then
    # Annotation
    align-text --special-symbol="'***'" ark:$capt_dir/should_say.txt ark:$capt_dir/actual_say.txt ark,t:- | \
	utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" > $capt_dir/annotation.txt
    # Prediction
    align-text --special-symbol="'***'" ark:$capt_dir/should_say.txt ark:$capt_dir/predict_say.txt ark,t:- | \
	utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" > $capt_dir/prediction.txt
fi

# Performance (recall, precision, and f1)
if [ $stage -le 2 ]; then
    # python
	python local/compute_capt_accuracy.py --anno $capt_dir/annotation.txt \
                                          --pred $capt_dir/prediction.txt \
                                          --capt_dir $capt_dir > $capt_dir/results.log
fi

if [ $stage -le 3 ]; then
    if [ ! -z $ctm_file ]; then
        cat $ctm_file | local/timit_norm_trans.pl -i - -m $phonemap -from 48 -to 39 | grep -v "sil" > $capt_dir/ctm.39.txt
    	python local/report_capt_raw_data.py --anno $capt_dir/annotation.txt \
                                         --pred $capt_dir/prediction.txt \
                                         --capt_dir $capt_dir \
                                         --ctm_file $capt_dir/ctm.39.txt
    else
        echo "Parameter ctm_file is unset, don't report raw_data"
    fi
fi


cat $capt_dir/results.log
