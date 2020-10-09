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
        print(df)
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


    # Training cycles comparison
    fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize=(10,5))
    # h =1 
    models = ['GAR', 'VAR', 'AR', 'RNNCON-Res']
    epochs = [280, 1500, 200, 160]
    ax[0][0].bar(models, epochs, color = ['g', 'b', 'y', 'r'], width = 0.4)
    ax[0][0].set_title("Number of timesteps = 1", fontsize=12)
    ax[0][0].set_xlabel('model')
    ax[0][0].set_ylabel('epoch')
    ax[0][0].set_ylim(0, 1500)
    # h=4
    epochs = [330, 430, 300, 160]
    ax[0][1].bar(models, epochs, color = ['g', 'b', 'y', 'r'], width = 0.4)
    ax[0][1].set_title("Number of timesteps = 4", fontsize=12)
    ax[0][1].set_xlabel('model')
    ax[0][1].set_ylabel('epoch')
    ax[0][1].set_ylim(0, 1500)

    # h = 8
    epochs = [350, 450, 300, 160]
    ax[0][2].bar(models, epochs, color = ['g', 'b', 'y', 'r'], width = 0.4)
    ax[0][2].set_title("Number of timesteps = 8", fontsize=12)
    ax[0][2].set_xlabel('model')
    ax[0][2].set_ylabel('epoch')
    ax[0][2].set_ylim(0, 1500)

    # h = 12
    epochs = [195, 600, 120, 160]
    ax[1][0].bar(models, epochs, color = ['g', 'b', 'y', 'r'], width = 0.4)
    ax[1][0].set_title("Number of timesteps = 12", fontsize=12)
    ax[1][0].set_xlabel('model')
    ax[1][0].set_ylabel('epoch')
    ax[1][0].set_ylim(0, 1500)

    # h = 16
    epochs = [170, 600, 120, 179]
    ax[1][1].bar(models, epochs, color = ['g', 'b', 'y', 'r'], width = 0.4)
    ax[1][1].set_title("Number of timesteps = 16", fontsize=12)
    ax[1][1].set_xlabel('model')
    ax[1][1].set_ylabel('epoch')
    ax[1][1].set_ylim(0, 1500)

    # h = 20
    epochs = [300, 88, 400, 164]
    ax[1][2].bar(models, epochs, color = ['g', 'b', 'y', 'r'], width = 0.4)
    ax[1][2].set_title("Number of timesteps = 20", fontsize=12)
    ax[1][2].set_xlabel('model')
    ax[1][2].set_ylabel('epoch')
    ax[1][2].set_ylim(0, 1500)

    plt.subplots_adjust(wspace = 0.5, hspace = 0.7)
    plt.savefig('./figs/Number of epoch to criterion.png')

    # Visual ablation test
    fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize=(10,5))
    # h =1 
    models = ['RNN', 'RNN-Res', 'RNNCON-Res']
    epochs = [612.20, 197.20, 202.77]
    ax[0][0].bar(models, epochs, color = ['cornflowerblue', 'lightskyblue', 'blue'], width = 0.4)
    ax[0][0].set_title("Number of timesteps = 1", fontsize=12)
    ax[0][0].set_xlabel('model')
    ax[0][0].set_ylabel('RMSE loss')
    ax[0][0].set_ylim(0, 650)
    # h=4
    epochs = [611.66, 220, 209.16]
    ax[0][1].bar(models, epochs, color = ['cornflowerblue', 'lightskyblue', 'blue'], width = 0.4)
    ax[0][1].set_title("Number of timesteps = 4", fontsize=12)
    ax[0][1].set_xlabel('model')
    ax[0][1].set_ylabel('RMSE loss')
    ax[0][1].set_ylim(0, 650)

    # h = 8
    epochs = [611.64, 251.25, 269.70]
    ax[0][2].bar(models, epochs, color = ['cornflowerblue', 'lightskyblue', 'blue'], width = 0.4)
    ax[0][2].set_title("Number of timesteps = 8", fontsize=12)
    ax[0][2].set_xlabel('model')
    ax[0][2].set_ylabel('RMSE loss')
    ax[0][2].set_ylim(0, 650)

    # h = 12
    epochs = [519.11, 322.26, 307.92]
    ax[1][0].bar(models, epochs, color = ['cornflowerblue', 'lightskyblue', 'blue'], width = 0.4)
    ax[1][0].set_title("Number of timesteps = 12", fontsize=12)
    ax[1][0].set_xlabel('model')
    ax[1][0].set_ylabel('RMSE loss')
    ax[1][0].set_ylim(0, 650)

    # h = 16
    epochs = [510, 308.43, 322.48]
    ax[1][1].bar(models, epochs,color = ['cornflowerblue', 'lightskyblue', 'blue'], width = 0.4)
    ax[1][1].set_title("Number of timesteps = 16", fontsize=12)
    ax[1][1].set_xlabel('model')
    ax[1][1].set_ylabel('RMSE loss')
    ax[1][1].set_ylim(0, 650)

    # h = 20
    epochs = [500, 343.63, 329.67]
    ax[1][2].bar(models, epochs, color = ['cornflowerblue', 'lightskyblue', 'blue'], width = 0.4)
    ax[1][2].set_title("Number of timesteps = 20", fontsize=12)
    ax[1][2].set_xlabel('model')
    ax[1][2].set_ylabel('RMSE loss')
    ax[1][2].set_ylim(0, 650)

    plt.subplots_adjust(wspace = 0.5, hspace = 0.7)
    plt.savefig('./figs/Ablation tests.png')
