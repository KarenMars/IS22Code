#!/bin/bash

# exit on error
set -e 

# Directory for data
DATA_DIR=../../data_config/TIMIT/
# Parent directory of output features.
FEATS_DIR=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/hangji/software/pae_probe_config/feats 
# Path to cpc model files
CPC_MODEL=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/hangji/software/zs_CPC/CPC_audio/checkpoints/ckp_960/checkpoint_10.pt
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
nj=8   # Number of parallel jobs for CPU operations.
stage=0

. path.sh

mkdir -p logs/


# ##############################################################################
# # Extract features
# ##############################################################################
# if [ $stage -le 1 ]; then
#     for corpus in timit; do
#         echo "Extracting cpc features for ${corpus}..."
#         # export CUDA_VISIBLE_DEVICES=`free-gpu`
#         gen_cpc_feats.py $CPC_MODEL $DATA_DIR/${corpus}/wav/ $FEATS_DIR/$corpus/cpc_960_10/ \
#             --cpu \
#             > logs/extract_cpc_${corpus}.stdout \
#             2> logs/extract_cpc_${corpus}.stderr
# done
# fi


# ##############################################################################
# # Run classification tasks.
# ##############################################################################
# if [ $stage -le 2 ]; then
#     echo "$0: Preparing config files..."
#     gen_config_files.py \
# 	--step 0.010 \
#         $FEATS_DIR  cpc_960_10 configs/tasks $DATA_DIR
# fi


if [ $stage -le 3 ]; then
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
