#!/bin/bash

window_list=(2 8 16 20)
horizon_list=(1 2 4 8 16 20)

hid_list=(5 10 20 40)
dropout_list=(0.0 0.2 0.5)

DATA=$1
TW=$2
GPU=$3
NORM=$4


for horizon in "${horizon_list[@]}"
do
    for window in "${window_list[@]}"; do
    for dropout in "${dropout_list[@]}"; do
    for hid in "${hid_list[@]}"; do
        rnn_option="--hidRNN ${hid}"
        tweets_option = "--tweets ${TW}"
        option="--dropout ${dropout} --normalize ${NORM} --epochs 2000 --data ${DATA} --model RNNCON_Res --save_dir save --save_name rnncon-res.w-${window}.h-${horizon}.pt --horizon ${horizon} --window ${window} --gpu ${GPU} --metric 0"
        cmd="stdbuf -o L python ./main.py ${option} ${rnn_option} ${tweets_option} | tee log/rnncon-res/rnncon-res.hid-${hid}.drop-${dropout}.w-${window}.h-${horizon}.out"
        echo $cmd
        eval $cmd
    done
    done
    done
done
