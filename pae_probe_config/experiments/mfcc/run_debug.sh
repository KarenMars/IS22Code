#!/bin/bash 

# set the language
# LANG = timit 
LANG=mboshi
# DATA_DIR=../../data/
# 04-10-2021 change data to configured data set
# TIMIT DATA SET
# DATA_DIR=../../data_config/TIMIT/
# FEATS_DIR=../../feats/

# # Path to the manner set 
# M_FEATURE_SET_DIR=../../bin/utils/omfile.pickle
# # Path to the place set 
# P_FEATURE_SET_DIR=../../bin/utils/opfile.pickle
# # Path to the high set 
# H_FEATURE_SET_DIR=../../bin/utils/ohfile.pickle
# # Path to the front set 
# F_FEATURE_SET_DIR=../../bin/utils/offile.pickle
# # Path to the round set
# R_FEATURE_SET_DIR=../../bin/utils/orfile.pickle
# # Path to the static set 
# S_FEATURE_SET_DIR=../../bin/utils/osfile.pickle

# mboshi DATA SET
DATA_DIR=../../data_config/Mboshi/
FEATS_DIR=../../feats/

# Path to the manner set 
M_FEATURE_SET_DIR=../../bin/utils/mboshi_map/omfile.pickle
# Path to the place set 
P_FEATURE_SET_DIR=../../bin/utils/mboshi_map/opfile.pickle
# Path to the high set 
H_FEATURE_SET_DIR=../../bin/utils/mboshi_map/ohfile.pickle
# Path to the front set 
F_FEATURE_SET_DIR=../../bin/utils/mboshi_map/offile.pickle
# Path to the round set
R_FEATURE_SET_DIR=../../bin/utils/mboshi_map/orfile.pickle
# Path to the static set 
S_FEATURE_SET_DIR=../../bin/utils/mboshi_map/osfile.pickle
# Path to the voice set 
V_FEATURE_SET_DIR=../../bin/utils/mboshi_map/ovfile.pickle


##############################################################################
# Configuration
##############################################################################
nj=8   # Number of parallel jobs for CPU operations.
stage=1

. path.sh

# mkdir -p ${LANG}/logs/


##############################################################################
# Extract features
##############################################################################
wl=0.025   # Window length in seconds.
step=0.01  # Step size in seconds.
if [ $stage -le 0 ]; then
    # configure corpus just for timit
    # for corpus in ctimit ffmtimit ntimit stctimit timit wtimit; do
    for corpus in $LANG; do
        echo "Extracting MFCC features for ${corpus}..."
        gen_librosa_feats.py \
            --ftype mfcc --config configs/feats/mfcc.yaml \
            --step $step --wl $wl --n_jobs $nj --disable-progress \
            $FEATS_DIR/${corpus}/mfcc $DATA_DIR/${corpus}/wav/*.wav \
            > ${LANG}/logs/extract_mfcc_${corpus}.stdout \
            2> ${LANG}/logs/extract_mfcc_${corpus}.stderr
    done
fi


#############################################################################
# Run binary classification tasks.
#############################################################################
if [ $stage -le 1 ]; then
    echo "$0: Preparing config files..."
    gen_config_files.py \
        --context-size 5 $FEATS_DIR mfcc configs/tasks $DATA_DIR
fi


if [ $stage -le 3 ]; then
    echo "$0: Running classification experiments..."
    # clean logs folder
    # rm -rf ${LANG}/logs/*
    echo "Clean logs folder ... Done!"
    for config in `ls configs/tasks/*.yaml`; do
        bn=${config##*/}
        name=${bn%.yaml}
        echo $name
        # check which task
        if [ $name = voice_svm ]
        then
            echo "running voice_svm"
            rm -rf ${LANG}/logs/voice
            mkdir ${LANG}/logs/voice
            ../../bin/svm_config/mboshi_svm/exp_voice_ova.py \
                --n-jobs $nj $config $V_FEATURE_SET_DIR \
                > ${LANG}/logs/voice/${name}.stdout \
                2> ${LANG}/logs/voice/${name}.stderr &
        # elif [ $name = manner_svm ]
        # then
        #     mkdir ${LANG}/logs/manner
        #     ../../bin/svm/mboshi_svm/exp_manner_ova.py \
        #         --n-jobs $nj $config $M_FEATURE_SET_DIR \
        #         > ${LANG}/logs/manner/${name}.stdout \
        #         2> ${LANG}/logs/manner/${name}.stderr &
        # elif [ $name = place_svm ]
        # then
        #     mkdir ${LANG}/logs/place
        #     ../../bin/svm/mboshi_svm/exp_place_ova.py \
        #         --n-jobs $nj $config $P_FEATURE_SET_DIR \
        #         > ${LANG}/logs/place/${name}.stdout \
        #         2> ${LANG}/logs/place/${name}.stderr &
        # elif [ $name = high_svm ]
        # then
        #     mkdir ${LANG}/logs/high
        #     ../../bin/svm/mboshi_svm/exp_high_ova.py \
        #         --n-jobs $nj $config $H_FEATURE_SET_DIR \
        #         > ${LANG}/logs/high/${name}.stdout \
        #         2> ${LANG}/logs/high/${name}.stderr &
        # elif [ $name = front_svm ]
        # then 
        #     mkdir ${LANG}/logs/front
        #     ../../bin/svm/mboshi_svm/exp_front_ova.py \
        #         --n-jobs $nj $config $F_FEATURE_SET_DIR \
        #         > ${LANG}/logs/front/${name}.stdout \
        #         2> ${LANG}/logs/front/${name}.stderr &
        # elif [ $name = round_svm ]
        # then 
        #     mkdir ${LANG}/logs/round
        #     ../../bin/svm/mboshi_svm/exp_round_ova.py \
        #         --n-jobs $nj $config $R_FEATURE_SET_DIR \
        #         > ${LANG}/logs/round/${name}.stdout \
        #         2> ${LANG}/logs/round/${name}.stderr &
        # elif [ $name = static_svm ]
        # then 
        #     mkdir ${LANG}/logs/static
        #     ../../bin/svm/mboshi_svm/exp_static_ova.py \
        #         --n-jobs $nj $config $S_FEATURE_SET_DIR \
        #         > ${LANG}/logs/static/${name}.stdout \
        #         2> ${LANG}/logs/static/${name}.stderr &
        fi
    done
    wait
fi
