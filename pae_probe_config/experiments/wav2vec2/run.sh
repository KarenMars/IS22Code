#!/bin/bash

set -e  # Exit on error

DATA_DIR=../../data_config/TIMIT/
FEATS_DIR=../../feats
# model name
MODEL_NAME=wav2vec2-base-ls960
# checkpoint of the pretrained model 
W2V2_MODEL=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/hangji/open_source_ckpt/huggingface/wav2vec-base 
# Path to the manner set 
M_FEATURE_SET_DIR=../../bin/utils/omfile.pickle
# Path to the place set 
P_FEATURE_SET_DIR=../../bin/utils/opfile.pickle
# Path to the high set 
H_FEATURE_SET_DIR=../../bin/utils/ohfile.pickle
# Path to the front set 
F_FEATURE_SET_DIR=../../bin/utils/offile.pickle
# Path to the round set
R_FEATURE_SET_DIR=../../bin/utils/orfile.pickle
# Path to the static set 
S_FEATURE_SET_DIR=../../bin/utils/osfile.pickle


##############################################################################
# Configuration
##############################################################################
nj=40   # Number of parallel jobs for CPU operations.
stage=0

. path.sh

mkdir -p logs/


##############################################################################
# Extract features
##############################################################################

# if [ $stage -le 0 ]; then
#     for corpus in timit ; do
#         echo "Extracting wav2vec 2.0 features for ${corpus}..."
#         export CUDA_VISIBLE_DEVICES=1
#         gen_wav2vec2_feats.py \
#             --use_gpu \
#             --disable-progress \
# 	        --wav2vec2 \
# 	        $W2V2_MODEL $FEATS_DIR/$corpus/${MODEL_NAME} \
#             $DATA_DIR/${corpus}/wav/*.wav \
#             > logs/extract_${MODEL_NAME}_${corpus}.stdout \
#             2> logs/extract_${MODEL_NAME}_${corpus}.stderr
# done
# fi


# ##############################################################################
# # Run classification tasks.
# ##############################################################################
# if [ $stage -le 1 ]; then
#     echo "$0: Preparing config files..."
#     gen_config_files.py \
# 	--step 0.020 \
#         $FEATS_DIR  wav2vec2-base-ls960 configs/tasks $DATA_DIR
# fi



if [ $stage -le 2 ]; then
    echo "$0: Running classification experiments..."
    # clean logs folder
    rm -rf logs/*
    echo "Clean logs folder ... Done!"
    for config in `ls configs/tasks/*.yaml`; do
        bn=${config##*/}
        name=${bn%.yaml}
        echo $name
        # check which task
        if [ $name = voice_svm ]
        then
            mkdir logs/voice
            ../../bin/svm/exp_voice_ova.py \
                --n-jobs $nj $config \
                > logs/voice/${name}.stdout \
                2> logs/voice/${name}.stderr &
        elif [ $name = manner_svm ]
        then
            mkdir logs/manner
            ../../bin/svm/exp_manner_ova.py \
                --n-jobs $nj $config $M_FEATURE_SET_DIR \
                > logs/manner/${name}.stdout \
                2> logs/manner/${name}.stderr &
        elif [ $name = place_svm ]
        then
            mkdir logs/place
            ../../bin/svm/exp_place_ova.py \
                --n-jobs $nj $config $P_FEATURE_SET_DIR \
                > logs/place/${name}.stdout \
                2> logs/place/${name}.stderr &
        elif [ $name = high_svm ]
        then
            mkdir logs/high
            ../../bin/svm/exp_high_ova.py \
                --n-jobs $nj $config $H_FEATURE_SET_DIR \
                > logs/high/${name}.stdout \
                2> logs/high/${name}.stderr &
        elif [ $name = front_svm ]
        then 
            mkdir logs/front
            ../../bin/svm/exp_front_ova.py \
                --n-jobs $nj $config $F_FEATURE_SET_DIR \
                > logs/front/${name}.stdout \
                2> logs/front/${name}.stderr &
        elif [ $name = round_svm ]
        then 
            mkdir logs/round
            ../../bin/svm/exp_round_ova.py \
                --n-jobs $nj $config $R_FEATURE_SET_DIR \
                > logs/round/${name}.stdout \
                2> logs/round/${name}.stderr &
        elif [ $name = static_svm ]
        then 
            mkdir logs/static
            ../../bin/svm/exp_static_ova.py \
                --n-jobs $nj $config $S_FEATURE_SET_DIR \
                > logs/static/${name}.stdout \
                2> logs/static/${name}.stderr &
        fi
    done
    wait
fi
