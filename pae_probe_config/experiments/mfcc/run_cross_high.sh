#!/bin/bash 

# set the language
Train_LANG=timit 

Test_LANG=mboshi

FEATS_DIR=../../feats/

# TIMIT DATA SET
Train_DATA_DIR=../../data_config/TIMIT/
# Path to the manner set 
Train_M_FEATURE_SET_DIR=../../bin/utils/timit_map/omfile.pickle
# Path to the place set 
Train_P_FEATURE_SET_DIR=../../bin/utils/timit_map/opfile.pickle
# Path to the high set 
Train_H_FEATURE_SET_DIR=../../bin/utils/timit_map/ohfile.pickle
# Path to the high set 
Train_F_FEATURE_SET_DIR=../../bin/utils/timit_map/offile.pickle
# Path to the high set
Train_R_FEATURE_SET_DIR=../../bin/utils/timit_map/orfile.pickle
# Path to the high set 
Train_S_FEATURE_SET_DIR=../../bin/utils/timit_map/osfile.pickle

# mboshi DATA SET
Test_DATA_DIR=../../data_config/Mboshi/
# Path to the manner set 
Test_M_FEATURE_SET_DIR=../../bin/utils/mboshi_map/omfile.pickle
# Path to the place set 
Test_P_FEATURE_SET_DIR=../../bin/utils/mboshi_map/opfile.pickle
# Path to the high set 
Test_H_FEATURE_SET_DIR=../../bin/utils/mboshi_map/ohfile.pickle
# Path to the high set 
Test_F_FEATURE_SET_DIR=../../bin/utils/mboshi_map/offile.pickle
# Path to the high set
Test_R_FEATURE_SET_DIR=../../bin/utils/mboshi_map/orfile.pickle
# Path to the high set 
Test_S_FEATURE_SET_DIR=../../bin/utils/mboshi_map/osfile.pickle
# Path to the voice set 
Test_V_FEATURE_SET_DIR=../../bin/utils/mboshi_map/ovfile.pickle


##############################################################################
# Configuration
##############################################################################
nj=8   # Number of parallel jobs for CPU operations.
stage=1

. path.sh

mkdir -p ${Train_LANG}_${Test_LANG}/logs/


##############################################################################
# Extract features
##############################################################################
wl=0.035   # Window length in seconds.
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

##############################################################################
# Run binary classification tasks.
##############################################################################
if [ $stage -le 1 ]; then
    echo "$0: Preparing config files..."
    gen_config_files.py \
        --context-size 5 $FEATS_DIR mfcc configs/tasks $Train_DATA_DIR $Test_DATA_DIR
fi


if [ $stage -le 2 ]; then
    echo "$0: Running classification experiments..."
    # clean logs folder
    # rm -rf ${LANG}/logs/*
    echo "Clean logs folder ... Done!"
    for config in `ls configs/tasks/*.yaml`; do
        bn=${config##*/}
        name=${bn%.yaml}
        echo $name
        # check which task
        if [ $name = high_svm ]
        then
            rm -rf ${Train_LANG}_${Test_LANG}/logs/high
            mkdir ${Train_LANG}_${Test_LANG}/logs/high
            ../../bin/svm_cross/exp_high_ova.py \
                --n-jobs $nj $config $Train_H_FEATURE_SET_DIR $Test_H_FEATURE_SET_DIR \
                > ${Train_LANG}_${Test_LANG}/logs/high/${name}.stdout \
                2> ${Train_LANG}_${Test_LANG}/logs/high/${name}.stderr &
        fi
    done
    wait
fi
