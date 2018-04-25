#!/bin/bash
fold=$1
feature_type=$2
data=$3
type=$4
name=${data}_${feature_type}_${type}

if [ $fold -lt -1 ] || [ $fold -gt 10 ] || [ $fold -eq 0 ]
then
  echo "Wrong fold!"
  exit
fi

if [ $feature_type != "raw" ] && [ $feature_type != "all" ]
then
  echo "Wrong feature type!"
  exit
fi

if [ $type = "dqn" ]
then
  type_args=""
elif [ $type = "double" ]
then
  type_args="--agent_double"
elif [ $type = "dueling" ]
then
  type_args="--agent_dueling"
elif [ $type = "doubledueling" ]
then
  type_args="--agent_double --agent_dueling"
else
  echo "Wrong type!"
  exit
fi

THEANO_FLAGS=device=gpu0,floatX=float32 python run_training.py -f $fold --feature $feature_type --directory /home_local/chung95191/ISCR-SQuAD/data/$data --result /home_local/chung95191/ISCR-SQuAD/result_survey --name $name $type_args
