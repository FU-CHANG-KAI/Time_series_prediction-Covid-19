#!/bin/bash

log=${1:-.}
mkdir ${log}

dirlist=(ar gar var rnn rnn_res rnncon_res)
for subfolder in "${dirlist[@]}"; do
    mkdir ${log}/${subfolder}
done

mkdir ${log}/save
