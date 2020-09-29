import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot 
import mpl_toolkits.axisartist as AA


def save(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    return 1

##-- Load obj from file    
def load(filename):
    with open(filename, 'rb') as input: 
        obj = pickle.load(input)
    return obj   

#def visual_plot(df1, df2, fig_save_dir):
def visual_plot(df1,  fig_save_dir): 
    plt.plot(df1, color = 'blue')
    plt.plot(df2, color = 'salmon')
    plt.savefig(fig_save_dir)
    #plt.clf()


if __name__ == "__main__":
    #1. USA nation- wide prediction comparison
    path = './figs/pickle/'
    #models = ['True value.pkl','RNN_Res.pkl', 'RNNCON_Res.pkl']
    #style = ['dimgrey', 'lightskyblue', 'blue']
    models = ['True value.pkl', 'AR.pkl', 'GAR.pkl', 'VAR.pkl', 'RNNCON_Res.pkl']
    style = ['dimgrey', 'y--', 'g--', 'b--', 'r--']
    for i in range(len(models)):
        df = load((path + models[i]))
        if models[i] == 'True value.pkl':
            plt.plot(df.iloc[41:], style[i], label = models[i][:-4])
            continue
        plt.plot(df, style[i], label = models[i][:-4])
    plt.legend(loc = 'upper left')
    plt.xlabel('Number of days since the first confirmed case')
    plt.ylabel('Daily confirmed cases')
    plt.savefig(path + 'usa_prediction.png')
    plt.clf()
    print("Complete country level predicted figure")

    #models = ['True value.pkl','RNN_Res.pkl', 'RNNCON_Res.pkl']
    #style = ['dimgrey', 'lightskyblue', 'blue']
    models = ['True value.pkl', 'AR.pkl', 'GAR.pkl', 'VAR.pkl', 'RNNCON_Res.pkl']
    style = ['dimgrey', 'y--', 'g--', 'b--', 'r--']
    #style = ['dimgrey', 'y--', 'g--', 'b--', 'r--']
    for state in ['New York', 'New Jersey', 'Connecticut', 'Illinois', 'Michigan', 'Alabama', 'California', 'Arizona', 'Utah', 'North Carolina']:
        for i in range(len(models)):
            save_dir = './figs/pickle/{}/{}'.format(state, models[i])
            df = load(save_dir)
            if models[i] == 'True value.pkl':
                plt.plot(df.iloc[41:], style[i], label = models[i][:-4])
                continue
            plt.plot(df, style[i], label = models[i][:-4])
        plt.legend(loc = 'upper left')
        plt.xlabel('Number of days since the first confirmed case')
        plt.ylabel('Daily confirmed cases')
        plt.title(state)
        fig_save_dir = './figs/pickle/{}/state prediction.png'.format(state)
        plt.savefig(fig_save_dir)
        plt.clf()
    print("Complete state level predicted figure")