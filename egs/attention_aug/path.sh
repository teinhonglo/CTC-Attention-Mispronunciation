KALDI_ROOT=/share/nas167/teinhonglo/kaldis/kaldi-cu11

#export KALDI_ROOT=/share/nas165/teinhonglo/kaldi-nnet3-specaug
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/gopbin:$PWD:$KALDI_ROOT/src/gopbin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

# For now, don't include any of the optional dependenices of the main
# librispeech recipe

eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
conda activate mdd_ca
