#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import glob
import numpy as np

## Find the best performance in sets of logs
# extract the values from log
def extract_tst_from_log(filename):
    # empty file
    lines = open(filename).readlines()
    if len(lines) < 1:
        return 1e10, -1
    line = lines[-1]
    # invalid or NaN
    if not line.startswith('test rse'):
        return 1e10, -1
    fields = line.split('|')
    tst_rse = float(fields[0].split()[2])
    tst_cor = float(fields[2].split()[2])
    return tst_rse, tst_cor


def format_logs(raw_expression):
    val_filenames = []
    for num in [1, 2, 4, 8]:
        expressions = raw_expression.format(num)
        filenames = glob.glob(expressions)
        tuple_list = [extract_tst_from_log(filename) for filename in filenames]
        if len(tuple_list) == 0:
            continue
        rse_list, cor_list = zip(*tuple_list)
        index = np.argmin(rse_list)
        print('horizon:{:2d}'.format(num), 'rmse: {:.4f}'.format(rse_list[index]), 'corr: {:.4f}'.format(cor_list[index]), 'best_model:', filenames[index])

if __name__ == '__main__':
    # rnncon_res
    print('*' * 40)
    format_logs('./log/rnncon_res/rnncon_res.hid-*.drop-*.w-*.h-{}.ratio-*.res-*.out')
    print('*' * 40)
    # cnnrnn
    format_logs('./log/rnn_res/rnn_reshid-*.drop-*.w-*.h-{}.out')
    print('*' * 40)
    # rnn
    format_logs('./log/rnn/rnn.hid-*.drop-*.w-*.h-{}.out')
    print('*' * 40)
    # gar
    format_logs('./log/gar/gar.d-*.w-*.h-{}.out')
    print('*' * 40)
    # ar
    format_logs('./log/ar/ar.d-*.w-*.h-{}.out')
    print('*' * 40)
    # var
    format_logs('./log/var/var.d-*.w-*.h-{}.out'
    print('*' * 40)