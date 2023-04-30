import numpy as np
from numpy import genfromtxt
import torch
import argparse
import yaml
from model import Ensemble

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(params: dict):

    orig_data = genfromtxt(params['data_dir'], delimiter=',')
    params['input_data'] = np.array(orig_data[1:,1:5])
    params['output_data'] = np.array(orig_data[1:,5])

    params['no_of_inputs'] = params['input_data'].shape[1]
    params['no_of_outputs'] = 1

    ensemble_ins = Ensemble(params=params)
    ensemble_ins.calculate_mean_var()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--yaml_file', '-yml', type=str, default=None)
    parser.add_argument('--num_models', '-nm', type=int, default=7)
    parser.add_argument('--train_val_ratio', type=float, default=0.2)

    args = parser.parse_args()
    params = vars(args)

    if params['yaml_file']:
        with open(args.yaml_file, 'r') as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            for config in yaml_config['args']:
                if config in params:
                    params[config] = yaml_config['args'][config]

    train(params)                

if __name__ == '__main__':
    main()
