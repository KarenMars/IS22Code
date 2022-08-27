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
# Path to the front set 
Train_F_FEATURE_SET_DIR=../../bin/utils/timit_map/offile.pickle
# Path to the front set
Train_R_FEATURE_SET_DIR=../../bin/utils/timit_map/orfile.pickle
# Path to the front set 
Train_S_FEATURE_SET_DIR=../../bin/utils/timit_map/osfile.pickle
# Path to the voice set 
Train_V_FEATURE_SET_DIR=../../bin/utils/timit_map/ovfile.pickle

# mboshi DATA SET
Test_DATA_DIR=../../data_config/Mboshi/
# Path to the manner set 
Test_M_FEATURE_SET_DIR=../../bin/utils/mboshi_map/omfile.pickle
# Path to the place set 
Test_P_FEATURE_SET_DIR=../../bin/utils/mboshi_map/opfile.pickle
# Path to the high set 
Test_H_FEATURE_SET_DIR=../../bin/utils/mboshi_map/ohfile.pickle
# Path to the front set 
Test_F_FEATURE_SET_DIR=../../bin/utils/mboshi_map/offile.pickle
# Path to the front set
Test_R_FEATURE_SET_DIR=../../bin/utils/mboshi_map/orfile.pickle
# Path to the front set 
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
    --step 0.020 \
        $FEATS_DIR wav2vec2-base-ls960 configs/tasks $Train_DATA_DIR $Test_DATA_DIR
fi

if [ $stage -le 2 ]; then
    echo "$0: Running classification experiments..."
    # clean logs folder
    rm -rf ${Train_LANG}_${Test_LANG}/logs/*
    echo "Clean logs folder ... Done!"
    for config in `ls configs/tasks/*.yaml`; do
        bn=${config##*/}
        name=${bn%.yaml}
        echo $name
        # check which task
        if [ $name = front_svm ]
        then
            rm -rf ${Train_LANG}_${Test_LANG}/logs/front
            mkdir ${Train_LANG}_${Test_LANG}/logs/front
            ../../bin/svm_cross/exp_front_ova.py \
                --n-jobs $nj $config $Train_F_FEATURE_SET_DIR $Test_F_FEATURE_SET_DIR \
                > ${Train_LANG}_${Test_LANG}/logs/front/${name}.stdout \
                2> ${Train_LANG}_${Test_LANG}/logs/front/${name}.stderr &
        elif [ $name = voice_svm ]
        then
            mkdir ${Train_LANG}_${Test_LANG}/logs/voice
            ../../bin/svm_cross/exp_voice_ova.py \
                --n-jobs $nj $config $Train_V_FEATURE_SET_DIR $Test_V_FEATURE_SET_DIR \
                > ${Train_LANG}_${Test_LANG}/logs/voice/${name}.stdout \
                2> ${Train_LANG}_${Test_LANG}/logs/voice/${name}.stderr &
        elif [ $name = manner_svm ]
        then
            mkdir ${Train_LANG}_${Test_LANG}/logs/manner
            ../../bin/svm_cross/exp_manner_ova.py \
                --n-jobs $nj $config $Train_M_FEATURE_SET_DIR $Test_M_FEATURE_SET_DIR \
                > ${Train_LANG}_${Test_LANG}/logs/manner/${name}.stdout \
                2> ${Train_LANG}_${Test_LANG}/logs/manner/${name}.stderr &
        elif [ $name = place_svm ]
        then
            mkdir ${Train_LANG}_${Test_LANG}/logs/place
            ../../bin/svm_cross/exp_place_ova.py \
                --n-jobs $nj $config $Train_P_FEATURE_SET_DIR $Test_P_FEATURE_SET_DIR \
                > ${Train_LANG}_${Test_LANG}/logs/place/${name}.stdout \
                2> ${Train_LANG}_${Test_LANG}/logs/place/${name}.stderr &
        elif [ $name = high_svm ]
        then
            mkdir ${Train_LANG}_${Test_LANG}/logs/high
            ../../bin/svm_cross/exp_high_ova.py \
                --n-jobs $nj $config $Train_H_FEATURE_SET_DIR $Test_H_FEATURE_SET_DIR \
                > ${Train_LANG}_${Test_LANG}/logs/high/${name}.stdout \
                2> ${Train_LANG}_${Test_LANG}/logs/high/${name}.stderr &
        elif [ $name = round_svm ]
        then 
            mkdir ${Train_LANG}_${Test_LANG}/logs/round
            ../../bin/svm_cross/exp_round_ova.py \
                --n-jobs $nj $config $Train_R_FEATURE_SET_DIR $Test_R_FEATURE_SET_DIR \
                > ${Train_LANG}_${Test_LANG}/logs/round/${name}.stdout \
                2> ${Train_LANG}_${Test_LANG}/logs/round/${name}.stderr &
        elif [ $name = static_svm ]
        then 
            mkdir ${Train_LANG}_${Test_LANG}/logs/static
            ../../bin/svm_cross/exp_static_ova.py \
                --n-jobs $nj $config $Train_S_FEATURE_SET_DIR $Test_S_FEATURE_SET_DIR \
                > ${Train_LANG}_${Test_LANG}/logs/static/${name}.stdout \
                2> ${Train_LANG}_${Test_LANG}/logs/static/${name}.stderr &
        fi
    done
    wait
fi
