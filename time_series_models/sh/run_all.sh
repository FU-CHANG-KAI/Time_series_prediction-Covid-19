#!/bin/bash

bash ./sh/grid_AR.sh ./data/data.txt 0 1
bash ./sh/grid_VAR.sh ./data/data.txt 0 1
bash ./sh/grid_GAR.sh ./data/data.txt 0 1
bash ./sh/grid_RNN.sh ./data/data.txt 0 1
bash ./sh/grid_RNN_Res.sh ./data/data.txt 0 1
bash ./sh/grid_RNNCON_Res.sh ./data/data.txt ./data/tweets_cases.txt hhs 0 1