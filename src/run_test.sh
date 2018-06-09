#!/bin/bash
fold=$1
feature_type=$2
data=$3
type=$4
model=$5
name=test_${data}_${feature_type}_${type}_${model}

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

THEANO_FLAGS=device=gpu0,floatX=float32 python run_training.py -f $fold --feature $feature_type --directory /home/tung/ISCR-DRL/data/$data --result /home/tung/ISCR-DRL/result_survey --name $name $type_args --model_dir /home/tung/ISCR-DRL/data/model/$model.pkl
