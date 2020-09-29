import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.ratio = args.ratio
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.GRU1 = nn.GRU(self.m, self.hidR)
        self.residual_window = args.residual_window
        self.dropout = nn.Dropout(p=args.dropout)

        if args.tweets:
            self.linear1 = nn.Linear(self.hidR*2, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, self.P)
            self.residual = nn.Linear(self.residual_window, 1);
            
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x, z):
        # x: batch x window (self.P) x #signal (m)
        # r: window (self.P) x batch x #signal (m)
        r = x.permute(1, 0, 2).contiguous() # why permute is required?
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        s = z.permute(1, 0, 2).contiguous()
        _, s = self.GRU1(s)
        s = self.dropout(torch.squeeze(s, 0))

        t = torch.cat((r,s),-1)
        res = self.linear1(t)

        #residual
        if (self.residual_window > 0):
            z = x[:, -self.residual_window:, :]; # Take out the last 4 window in batch x = 73 x 16 x 10
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window);
            z = self.residual(z);
            z = z.view(-1,self.m);
            res = res * self.ratio + z;

        if self.output is not None:
            res = self.output(res).float()
        #print("res = {}".format(res.shape))
        return res