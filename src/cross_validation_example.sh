#!/bin/bash

# -f : fold
# --feature: all/raw/wig/nqc
# --directory: data directory $ISCR_HOME/data/target_directory/
# --result: result directory $ISCR_HOME/result/
# --name: experiment_name

for i in {1..10}
do
    THEANO_FLAGS=device=gpu0,floatX=float32 python run_training.py -f ${i} --feature raw --directory /home_local/chung95191/ISCR-DRL/data/onebest_CMVN --result ../result_survey --name 4_
done
