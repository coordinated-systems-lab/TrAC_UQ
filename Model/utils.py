import torch.nn as nn
import torch 
import numpy as np
import sys
import os 
from pathlib import Path
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MeanStdevFilter():
    def __init__(self, shape, clip=10.0):
        self.eps = 1e-12
        self.shape = shape
        self.clip = clip
        self._count = 0
        self._running_sum = np.zeros(shape)  # ob_dim or ac_dim
        self._running_sum_sq = np.zeros(shape) + self.eps
        self.mean = 0
        self.stdev = 1

    def update(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1,-1)   
        self._running_sum += np.sum(x, axis=0)
        self._running_sum_sq += np.sum(np.square(x), axis=0)
        # assume 2D data
        self._count += x.shape[0]
        self.mean = self._running_sum / self._count
        self.stdev = np.sqrt(
            np.maximum(
                self._running_sum_sq / self._count - self.mean**2,
                 self.eps
                 ))
        self.stdev[self.stdev <= self.eps] = 1.0

    def reset(self):
        self.__init__(self.shape, self.clip)

    def update_torch(self):
        self.torch_mean = torch.FloatTensor(self.mean).to(device)
        self.torch_stdev = torch.FloatTensor(self.stdev).to(device)

    def filter(self, x):
        return np.clip(((x - self.mean) / self.stdev), -self.clip, self.clip)

    def filter_torch(self, x: torch.Tensor):
        self.update_torch()
        return torch.clamp(((x - self.torch_mean) / self.torch_stdev), -self.clip, self.clip)

    def invert(self, x):
        return (x * self.stdev) + self.mean

    def invert_torch(self, x: torch.Tensor):
        self.update_torch()
        return (x * self.torch_stdev) + self.torch_mean

class GaussianMSELoss(nn.Module):

    def __init__(self):
        super(GaussianMSELoss, self).__init__()

    def forward(self, mu_logvar, target, logvar_loss = True):
        mu, logvar = mu_logvar.chunk(2, dim=1)
        inv_var = (-logvar).exp()
        if logvar_loss:
            return (logvar + (target - mu)**2 * inv_var).mean()
        else:
            return ((target - mu)**2).mean()

def prepare_data(input_data, input_filter):

    input_filtered = input_filter.filter(input_data)
    
    return input_filtered

def find_min_distances(val_data:np.ndarray, train_data:np.ndarray):
    """
    Find the nearest training example to each data point from validation dataset,
    calculate distance and return the indices of validation examples sorted based on 
    their distances from the nearest training examples 
    """ 
    print("Calculating distances...")
    min_distances = []
    for val_dat in val_data:
        
        distances = np.linalg.norm(train_data - val_dat, axis=1)
        min_distance = np.min(distances)
        min_distances.append(min_distance)

    return np.argsort(min_distances)

def min_max_norm(array: np.ndarray):
    """
    To split the data intentionally based on this instead of random 
    mixing to test the UQ abilities of deep ensembles. 
    """
    if array.shape[1] == 1:
        array = array.reshape(1,-1)    # reshape to find the min and max 
    min_value = np.min(array, axis=1)
    max_value = np.max(array, axis=1)
    norm_array = np.divide(np.subtract(array.T, min_value), np.subtract(max_value, min_value))

    if array.shape[0] == 1:
        return norm_array
    else:
        return norm_array.T       

def check_or_make_folder(folder_path):
    """
    Helper function that (safely) checks if a dir exists; if not, it creates it
    """
    
    folder_path = Path(folder_path)

    try:
        folder_path.resolve(strict=True)
    except FileNotFoundError:
        print("{} dir not found, creating it".format(folder_path))
        os.mkdir(folder_path)

def plot_many(mu: np.ndarray, upper_mu: np.ndarray, lower_mu: np.ndarray, ground_truth: np.ndarray, no_of_outputs:int, file_name:str, save_dir:str=None):

    no_of_inputs = np.linspace(0, stop=mu.shape[1], num=mu.shape[1])

    fig = plt.figure() 

    gs = fig.add_gridspec(no_of_outputs, hspace=0.15)
    ax = gs.subplots(sharex=True)

    fig.set_figheight(7*no_of_outputs)
    fig.set_figwidth(18)

    plt.rc('font', size=22)          # controls default text sizes
    plt.rc('axes', titlesize=22)     # fontsize of the axes title
    plt.rc('axes', labelsize=25)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=22)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=22)    # fontsize of the tick labels
    plt.rc('legend', fontsize=22)    # legend fontsize
    plt.rc('figure', titlesize=12)  # fontsize of the figure title

    for i in range(no_of_outputs):
        ax[i].fill_between(no_of_inputs, lower_mu[i,:].reshape(-1,), upper_mu[i,:].reshape(-1,), alpha=0.3)
        k1, = ax[i].plot(no_of_inputs, mu[i,:].reshape(-1,), "k*-")
        r1, = ax[i].plot(no_of_inputs, ground_truth[i,:].reshape(-1,), "r*-")
        ax[i].grid(True)
        ax[0].set_ylabel('Predictions')

    ax[no_of_outputs-1].set_xlabel('Inputs')
    ax[0].legend(('Confidence Interval', 'Predictions', 'Ground Truth'),\
                  bbox_to_anchor=(0,1.01,0.9,0.2), mode='expand', loc='lower center', ncol=4,\
                      borderaxespad=0, shadow=False)
    
    main_dir = "../Results/"
    if save_dir: 
        main_dir = main_dir + save_dir

    plt.savefig(main_dir+file_name)
    plt.show()

def plot_mse_var(mses: np.ndarray, aggr_var_dict: dict, file_name:str, save_dir:str=None):

    y_labels = [['Max Aleotoric', 'Ensemble Var'], ['Ensemble Std', 'LL Var']]
    fig = plt.figure() 

    gs = fig.add_gridspec(2, 2, hspace=0.15)
    ax = gs.subplots(sharex=True)

    fig.set_figheight(25)
    fig.set_figwidth(22)

    plt.rc('font', size=29)          # controls default text sizes
    plt.rc('axes', titlesize=29)     # fontsize of the axes title
    plt.rc('axes', labelsize=26)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=29)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=29)    # fontsize of the tick labels
    plt.rc('legend', fontsize=22)    # legend fontsize
    plt.rc('figure', titlesize=12)  # fontsize of the figure title

    i, j = 0, 0
    for key in aggr_var_dict.keys():
        ax[i, j].plot(mses, aggr_var_dict[key], "k*")
        ax[i, j].grid(True)
        ax[i, j].set_ylabel(y_labels[i][j])
        if j == 1:
            j = 0
            i += 1
            continue
        j += 1    

    ax[1,0].set_xlabel('Squared Error')
    ax[1,1].set_xlabel('Squared Error')
    #ax.legend(('Confidence Interval', 'Predictions', 'Ground Truth'),\
    #              bbox_to_anchor=(0,1.01,0.9,0.2), mode='expand', loc='lower center', ncol=4,\
    #                  borderaxespad=0, shadow=False)
    
    main_dir = "../Results/"
    if save_dir: 
        main_dir = main_dir + save_dir

    plt.savefig(main_dir+file_name)
    plt.show()

def plot_one(mu: np.ndarray, upper_mu: np.ndarray, lower_mu: np.ndarray, ground_truth: np.ndarray, file_name:str, save_dir:str=None):

    no_of_inputs = np.linspace(0, stop=mu.shape[0], num=mu.shape[0])

    fig = plt.figure() 

    gs = fig.add_gridspec(1, hspace=0.15)
    ax = gs.subplots(sharex=True)

    fig.set_figheight(7)
    fig.set_figwidth(18)

    plt.rc('font', size=22)          # controls default text sizes
    plt.rc('axes', titlesize=22)     # fontsize of the axes title
    plt.rc('axes', labelsize=25)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=22)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=22)    # fontsize of the tick labels
    plt.rc('legend', fontsize=25)    # legend fontsize
    plt.rc('figure', titlesize=12)  # fontsize of the figure title

    ax.fill_between(no_of_inputs, lower_mu.reshape(-1,), upper_mu.reshape(-1,), alpha=0.3)
    r1, = ax.plot(no_of_inputs, ground_truth.reshape(-1,), "r-")
    k1, = ax.plot(no_of_inputs, mu.reshape(-1,), "k-")

    ax.set_ylabel('Photosynthetic Rate')
    ax.set_xlabel('Inputs')
    ax.grid(True)
    ax.legend(('Confidence Interval', 'Ground Truth', 'Predictions'),\
                  bbox_to_anchor=(0,1.01,0.9,0.2), mode='expand', loc='lower center', ncol=4,\
                      borderaxespad=0, shadow=False)

    main_dir = "../Results/"
    if save_dir: 
        main_dir = main_dir + save_dir

    plt.savefig(main_dir+file_name)
    plt.show()    