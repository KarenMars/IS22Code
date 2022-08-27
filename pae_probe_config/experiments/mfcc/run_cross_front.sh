#!/bin/bash 

# set the language
Train_LANG=timit 

Test_LANG=timit

FEATS_DIR=../../feats/

# TIMIT DATA SET
timit_DATA_DIR=../../data_config/TIMIT/
# Path to the manner set 
timit_M_FEATURE_SET_DIR=../../bin/utils/timit_map/omfile.pickle
# Path to the place set 
timit_P_FEATURE_SET_DIR=../../bin/utils/timit_map/opfile.pickle
# Path to the high set 
timit_H_FEATURE_SET_DIR=../../bin/utils/timit_map/ohfile.pickle
# Path to the front set 
timit_F_FEATURE_SET_DIR=../../bin/utils/timit_map/offile.pickle
# Path to the front set
timit_R_FEATURE_SET_DIR=../../bin/utils/timit_map/orfile.pickle
# Path to the front set 
timit_S_FEATURE_SET_DIR=../../bin/utils/timit_map/osfile.pickle
# Path to the voice set
timit_V_FEATURE_SET_DIR=../../bin/utils/timit_map/ovfile.pickle


# mboshi DATA SET
mboshi_DATA_DIR=../../data_config/Mboshi/
# Path to the manner set 
mboshi_M_FEATURE_SET_DIR=../../bin/utils/mboshi_map/omfile.pickle
# Path to the place set 
mboshi_P_FEATURE_SET_DIR=../../bin/utils/mboshi_map/opfile.pickle
# Path to the high set 
mboshi_H_FEATURE_SET_DIR=../../bin/utils/mboshi_map/ohfile.pickle
# Path to the front set 
mboshi_F_FEATURE_SET_DIR=../../bin/utils/mboshi_map/offile.pickle
# Path to the front set
mboshi_R_FEATURE_SET_DIR=../../bin/utils/mboshi_map/orfile.pickle
# Path to the front set 
mboshi_S_FEATURE_SET_DIR=../../bin/utils/mboshi_map/osfile.pickle
# Path to the voice set 
mboshi_V_FEATURE_SET_DIR=../../bin/utils/mboshi_map/ovfile.pickle


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
        --context-size 5 $FEATS_DIR mfcc configs/tasks $timit_DATA_DIR $timit_DATA_DIR 'timit' 'timit'
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
        if [ $name = front_svm ]
        then
            rm -rf ${Train_LANG}_${Test_LANG}/logs/front
            mkdir ${Train_LANG}_${Test_LANG}/logs/front
            ../../bin/svm_cross/exp_front_ova.py \
                --n-jobs $nj $config $timit_F_FEATURE_SET_DIR $timit_F_FEATURE_SET_DIR ${Train_LANG}_${Test_LANG} 'timit' 'timit' \
                > ${Train_LANG}_${Test_LANG}/logs/front/${name}.stdout \
                2> ${Train_LANG}_${Test_LANG}/logs/front/${name}.stderr &
        fi
    done
    wait
fi
