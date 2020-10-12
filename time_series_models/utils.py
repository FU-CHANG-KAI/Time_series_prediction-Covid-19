import torch
import numpy as np
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, args):
        self.cuda = args.cuda
        self.P = args.window
        self.h = args.horizon

        fin = open(args.data)
        self.rawdat = np.loadtxt(fin,delimiter=',')
        if args.tweets:
            fin2 = open(args.tweets)
            self.twtdat = np.loadtxt(fin2,delimiter=',')
            self.tweets = args.tweets
            self.TWTdat = np.zeros(self.twtdat.shape)
        else:
            self.tweets = None

        if (len(self.rawdat.shape)==1):
            self.rawdat = self.rawdat.reshape((self.rawdat.shape[0], 1))
        
        self.dat = np.zeros(self.rawdat[41:141].shape)
        self.n, self.m = self.dat.shape
        self.normalize = args.normalize
        self.scale = np.ones(self.m)
        self._normalized(self.normalize)

        self._split(int(args.train * self.n), int((args.train+args.valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()

        #compute denominator of the RSE and RAE
        self.compute_metric(args)

        if self.cuda:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)


    def compute_metric(self, args):
        #use the normal rmse and mae when args.metric == 0
        if (args.metric == 0):
            self.rse = 1.
            self.rae = 1.
            return

        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)
        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))



    def _normalized(self, normalize):
        if (normalize == 0):
            self.dat = self.rawdat[41:141]
            if self.tweets:
                self.TWTdat = self.twtdat
        #normalized by the maximum value of entire matrix.
        if (normalize == 1):
            self.scale = self.scale * (np.mean(np.abs(self.rawdat[41:141]))) * 2
            self.dat = self.rawdat[41:141] / (np.mean(np.abs(self.rawdat[41:141])) * 2)
            if self.tweets:
                self.TWTdat = self.twtdat/ (np.mean(np.abs(self.twtdat)) * 2)

        #normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[41:141][:,i]))
                self.dat[:,i] = self.rawdat[41:141][:,i] / np.max(np.abs(self.rawdat[41:141][:,i]))
                if self.tweets:
                    self.TWTdat[:,i] = self.twtdat[:,i] / np.max(np.abs(self.twtdat[:,i]))


    def _split(self, train, valid, test):

        train_set = range(self.P+self.h-1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

        if (train==valid):
            self.valid = self.test


    def _batchify(self, idx_set, horizon):

        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        Z = torch.zeros((n, self.P, self.m))
        if self.tweets:
            for i in range(n):
                end = idx_set[i] - self.h + 1
                start = end - self.P
                X[i,:self.P,:] = torch.from_numpy(self.dat[start:end, :])
                Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :])
                Z[i,:self.P,:] = torch.from_numpy(self.TWTdat[start:end, :])

            return [X, Y, Z]
        
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i,:self.P,:] = torch.from_numpy(self.dat[start:end, :])
            Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :])
            
        return [X, Y]


    def get_batches(self, data, batch_size, shuffle=True):
        if self.tweets:
            inputs = data[0]
            targets = data[1]
            tweets = data[2]
            length = len(inputs)
            if shuffle:
                index = torch.randperm(length)
            else:
                index = torch.LongTensor(range(length))
            start_idx = 0
            while (start_idx < length):
                end_idx = min(length, start_idx + batch_size)
                excerpt = index[start_idx:end_idx]
                X = inputs[excerpt]
                Y = targets[excerpt]
                Z = tweets[excerpt]
                if (self.cuda):
                    X = X.cuda()
                    Y = Y.cuda()
                    Z = Z.cuda()
                model_inputs = Variable(X)

                data = [model_inputs, Variable(Y), Variable(Z)]
                yield data
                start_idx += batch_size  
        else:
            inputs = data[0]
            targets = data[1]
            length = len(inputs)
            if shuffle:
                index = torch.randperm(length)
            else:
                index = torch.LongTensor(range(length))
            start_idx = 0
            while (start_idx < length):
                end_idx = min(length, start_idx + batch_size)
                excerpt = index[start_idx:end_idx]
                X = inputs[excerpt]
                Y = targets[excerpt]
                if (self.cuda):
                    X = X.cuda()
                    Y = Y.cuda()
                model_inputs = Variable(X)

                data = [model_inputs, Variable(Y)]
                yield data
                start_idx += batch_size
