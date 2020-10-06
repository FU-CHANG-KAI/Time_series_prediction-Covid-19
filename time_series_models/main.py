#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
import math
import time

import torch
import torch.nn as nn
from models import AR, VAR, GAR, RNN
from models import RNNCON_Res, RNN_Res, RNNCON
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from utils import *
import Optim
import pickle

states_full = ['Alabama','Alaska','American Samoa','Arizona','Arkansas','California',
 'Colorado','Connecticut','Delaware','Diamond Princess','District of Columbia','Florida','Georgia',
 'Grand Princess','Guam','Hawaii','Idaho','Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana',
 'Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri',
 'Montana','Nebraska','Nevada','New Hampshire','New Jersey','New Mexico','New York',
 'North Carolina','North Dakota','Northern Mariana Islands','Ohio','Oklahoma','Oregon','Pennsylvania','Puerto Rico',
 'Rhode Island','South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont',
 'Virgin Islands','Virginia','Washington','West Virginia','Wisconsin','Wyoming']

def save(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    return 1

##-- Load obj from file    
def load(filename):
    with open(filename, 'rb') as input: 
        obj = pickle.load(input)
    return obj   

def evaluate(loader, data, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    LOSS = 0
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;

    for inputs in loader.get_batches(data, batch_size, False):
        if args.tweets:
            X, Y, Z = inputs[0], inputs[1], inputs[2]
            output = model(X, Z);
        else:
            X, Y = inputs[0], inputs[1]
            output = model(X);

        if predict is None:
            predict = output.cpu();
            test = Y.cpu();
        else:
            predict = torch.cat((predict,output.cpu()));
            test = torch.cat((test, Y.cpu()));

        scale = loader.scale.expand(output.size(0), loader.m)
        loss = criterion(output * scale, Y * scale);

        if torch.__version__ < '0.4.0':
            LOSS += loss.data[0]
        else:
            LOSS += loss.item()


        if torch.__version__ < '0.4.0':
            total_loss += evaluateL2(output * scale , Y * scale ).data[0]
            total_loss_l1 += evaluateL1(output * scale , Y * scale ).data[0]
        else:
            total_loss += evaluateL2(output * scale , Y * scale ).item()
            total_loss_l1 += evaluateL1(output * scale , Y * scale ).item()
        n_samples += (output.size(0) * loader.m);

    rse = math.sqrt(total_loss / n_samples)/loader.rse
    rae = (total_loss_l1/n_samples)/loader.rae
    correlation = 0;
    PREDICT = predict
    predict = predict.data.numpy();
    Ytest = test.data.numpy();
    sigma_p = (predict).std(axis = 0);
    sigma_g = (Ytest).std(axis = 0);
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0);

    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g);
    correlation = (correlation[index]).mean();
    # root-mean-square error, absolute error, correlation
    scale = loader.scale.expand(PREDICT.shape[0], loader.m)
    PREDICT  = PREDICT  * scale
    PREDICT = PREDICT.data.numpy()
    return rse, rae, correlation, PREDICT;


def train(loader, data, model, criterion, optim, batch_size):
    model.train();
    total_loss = 0;
    n_samples = 0;
    counter = 0
    for inputs in loader.get_batches(data, batch_size, True):
        counter += 1
        model.zero_grad();
        if args.tweets:
            X, Y, Z = inputs[0], inputs[1], inputs[2]
            output = model(X, Z);
        else:
            X, Y = inputs[0], inputs[1]
            output = model(X);
        scale = loader.scale.expand(output.size(0), loader.m)
        loss = criterion(output * scale, Y * scale);
        loss.backward();
        optim.step();
        if torch.__version__ < '0.4.0':
            total_loss += loss.data[0]
        else:
            total_loss += loss.item()
        n_samples += (output.size(0) * loader.m);
    return total_loss / n_samples

def visual_plot(df1, df2): 
    fig_save_dir = './figs/prediction-{}.h-{}.w-{}.rw-{}.m-{}.n-{}.png'.format(args.model, args.hidRNN, args.window, args.residual_window, args.metric, args.normalize)
    plt.plot(df1, color = 'blue')
    plt.plot(df2, color = 'salmon')
    plt.savefig(fig_save_dir)
    #plt.show()
    plt.clf()

def convert_to_prediction_df(predict, Data):
    p = Data.P # windonw size
    h = Data.h # horizon
    predict = predict.tolist()
    data = Data.rawdat[:142]
    n, m = data.shape
    data = data.tolist()

    predict_dict = {}
    df_predict = pd.DataFrame()
    
    for j in range(h):
        for i in range(Data.m):
            predict_dict[states_full[i]] = predict[j][i]
        predict_new = pd.DataFrame(predict_dict,index = [j])
        df_predict = df_predict.append(predict_new)

    df_predict = pd.DataFrame(predict[-h:], columns = states_full,
    index = [i for i in range(n-args.horizon, n)])
    usa_count_predict = [sum(df_predict.iloc[i].tolist()) for i in range(h)]
    df_predict['usa'] = usa_count_predict

    true_dict = {}
    df_true = pd.DataFrame()
    for j in range(n):
        for i in range(58):
            true_dict[states_full[i]] = data[j][i]
        true_new = pd.DataFrame(true_dict, index = [j])
        df_true = df_true.append(true_new)

    usa_count_true = [sum(df_true.iloc[i].tolist()) for i in range(df_true.shape[0])]
    df_true['usa'] = usa_count_true
    
    save_dir = './figs/pickle/{}.pkl'.format(args.model)
    save(df_predict['usa'], save_dir)


    save_dir = './figs/pickle/True value.pkl'
    if not os.path.exists(save_dir):
        save(df_true['usa'], save_dir)


        # State level prediction
    for state in ['New York', 'New Jersey', 'Connecticut', 'Illinois', 'Michigan', 'Alabama', 'California', 'Arizona', 'Utah', 'North Carolina']:
        fig_save_dir = './figs/pickle/{}/{}.pkl'.format(state, args.model)
        save(df_predict[state], fig_save_dir)
        print("predicted values are successfully saves in {}".format(state))
        fig_exist_dir = './figs/pickle/{}/True value.pkl'.format(state)

        save(df_true[state],fig_exist_dir)


def visual_plot(df1, df2): 
    fig_save_dir = './figs/predict/prediction-{}.h-{}.w-{}.rw-{}.m-{}.n-{}.png'.format(args.model, args.hidRNN, args.window, args.residual_window, args.metric, args.normalize)
    plt.plot(df1, color = 'blue')
    plt.plot(df2, color = 'salmon')
    plt.savefig(fig_save_dir)
    #plt.show()
    plt.clf()

parser = argparse.ArgumentParser(description='COVID-19 time-series prediction')
# --- Data option
parser.add_argument('--data', type=str, required=True,help='location of the data file')
parser.add_argument('--train', type=float, default=0.6,help='how much data used for training')
parser.add_argument('--valid', type=float, default=0.2,help='how much data used for validation')
parser.add_argument('--model', type=str, default='AR',help='model to select')
# --- RNN option
parser.add_argument('--tweets', type=str, default= None, help='file of tweets (Required for RNNCON_Res)')
parser.add_argument('--hidRNN', type=int, default=50, help='number of RNN hidden units')
parser.add_argument('--residual_window', type=int, default=4,help='The window size of the residual component')
parser.add_argument('--ratio', type=float, default=1.,help='The ratio between CNNRNN and residual')
parser.add_argument('--output_fun', type=str, default=None, help='the output function of neural net')
# --- Logging option
parser.add_argument('--save_dir', type=str,  default='./save',help='dir path to save the final model')
parser.add_argument('--save_name', type=str,  default='tmp', help='filename to save the final model')
# --- Optimization option
parser.add_argument('--optim', type=str, default='adam', help='optimization method')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--epochs', type=int, default=100,help='upper epoch limit')
parser.add_argument('--clip', type=float, default=1.,help='gradient clipping')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (L2 regularization)')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',help='batch size')
# --- Misc prediction option
parser.add_argument('--horizon', type=int, default=12, help='predict horizon')
parser.add_argument('--window', type=int, default=24 * 7,help='window size')
parser.add_argument('--metric', type=int, default=1, help='whether (1) or not (0) normalize rse and rae with global variance/deviation ')
parser.add_argument('--normalize', type=int, default=0, help='the normalized method used, detail in the utils.py')

parser.add_argument('--seed', type=int, default=54321,help='random seed')
parser.add_argument('--gpu', type=int, default=None, help='GPU number to use')
parser.add_argument('--cuda', type=str, default=True, help='use gpu or not')

args = parser.parse_args()
print(args);
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if args.model in ['RNNCON_Res', 'RNNCON'] and args.tweets is None:
    print('RNNCON_Res requires "tweets" option')
    sys.exit(0)

args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_utility(args);

model = eval(args.model).Model(args, Data);
print('model:', model)
if args.cuda:
    model.cuda()

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

criterion = nn.MSELoss(size_average=False);
evaluateL2 = nn.MSELoss(size_average=False);
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda();
    evaluateL2 = evaluateL2.cuda();


best_val = 10000000;
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip, weight_decay = args.weight_decay,
)

# At any point you can hit Ctrl + C to break out of training early.
try:
    print('begin training');
    val_loss_lst = []
    time_lst = []
    time_track = []
    final_epoch = 0
    time_stamp = 0
    for epoch in range(1, args.epochs+1):
        final_epoch += 1
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train, model, criterion, optim, args.batch_size)
        val_loss, val_rae, val_corr, __ = evaluate(Data, Data.valid, model, evaluateL2, evaluateL1, args.batch_size);
        val_loss_lst.append(val_loss) 
        time_track.append(time_stamp + time.time() - epoch_start_time)
        time_stamp += (time.time() - epoch_start_time)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.8f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val:
            best_val = val_loss
            model_path = '%s/%s.pt' % (args.save_dir, args.save_name)
            with open(model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            print('best validation');
            test_acc, test_rae, test_corr, ____  = evaluate(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size);
            print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_path = '%s/%s.pt' % (args.save_dir, args.save_name)
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f));
test_acc, test_rae, test_corr, predict  = evaluate(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size);
print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

convert_to_prediction_df(predict, Data)

val_loss_epoch = pd.DataFrame(val_loss_lst, index = range(1, final_epoch))
val_loss_time =  pd.DataFrame(val_loss_lst, index = time_track)

# Plot the consumtion number of epoch versus validation loss based on RMSE
plt.plot(val_loss_epoch, color = 'blue')
plt.xlabel('Epoch')
plt.ylabel('RMSE loss')
plt.savefig('./figs/loss-epoch.{}.h-{}.rw-{}.c-{}.wd-{}.hr-{}.png'\
.format(args.model, args.horizon, args.residual_window, args.clip, args.weight_decay, args.hidRNN))
plt.clf()

# Plot the consumtion number of time\ versus validation loss based on RMSE
plt.plot(val_loss_time, color = 'salmon')
plt.xlabel('time(s)')
plt.ylabel('RMSE loss')
plt.savefig('./figs/loss-time.{}.h-{}.rw-{}.c-{}.wd-{}.hr-{}.png'\
.format(args.model, args.horizon, args.residual_window, args.clip, args.weight_decay, args.hidRNN))
plt.clf()