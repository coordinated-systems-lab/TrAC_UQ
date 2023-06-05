import numpy as np
from numpy import genfromtxt
import torch
import argparse
import yaml
from model import Ensemble
from utils import plot_one, find_min_distances, plot_mse_var
from sklearn.metrics import mean_squared_error
import random
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(params: dict):

    #np.random.seed(params['seed'])
    #random.seed(params['seed'])

    orig_data = genfromtxt(params['data_dir'], delimiter=',')
    params['input_data'] = np.array(orig_data[1:,1:5])
    params['output_data'] = np.array(orig_data[1:,5]).reshape(-1,1)

    params['no_of_inputs'] = params['input_data'].shape[1]
    params['no_of_outputs'] = 1


    ensemble_ins = Ensemble(params=params)
             
    if params['train_mode']:
        ensemble_ins.calculate_mean_var()
        ensemble_ins.set_loaders()
        ensemble_ins.train_model(params['model_epochs'], save_model=True)
    if params['test_mode']:
        if not params['saved_pred_csv']:
            ensemble_ins.load_model(params['load_model_dir'])
            sorted_val_indices = find_min_distances(ensemble_ins.rand_input_val, ensemble_ins.rand_input_train)
            sorted_rand_input_filtered_val = ensemble_ins.rand_input_filtered_val[sorted_val_indices]
            sorted_rand_output_filtered_val = ensemble_ins.rand_output_filtered_val[sorted_val_indices]
            ground_truth = ensemble_ins.output_filter.invert(sorted_rand_output_filtered_val)
            # for aggregated mean and variance 
            mu_unnorm_all = torch.zeros(params['num_models'], sorted_rand_input_filtered_val.shape[0], device=device)
            logvar_all = torch.zeros(params['num_models'], sorted_rand_input_filtered_val.shape[0], device=device)
            actual_mu_all = torch.zeros(params['num_models'], sorted_rand_input_filtered_val.shape[0], device=device)

            for model_no, model in ensemble_ins.models.items():
                mu, logvar =  model.get_next_state_reward(sorted_rand_input_filtered_val, \
                                                        deterministic=True, return_mean=True) # normalized validation data
                mu_unnorm, upper_mu_unnorm, lower_mu_unnorm =  ensemble_ins.calculate_bounds(mu[0], logvar)
                mu_unnorm_all[model_no,:] = torch.FloatTensor(mu_unnorm.reshape(-1,)).to(device)
                logvar_all[model_no,:] = logvar.reshape(-1,)
                actual_mu_all[model_no,:] = ensemble_ins.output_filter.invert_torch(mu[1].reshape(1,-1))
                if params['save_pred']:
                    save_pred = np.stack((ground_truth.reshape(-1,), mu_unnorm.reshape(-1,),
                                        upper_mu_unnorm.reshape(-1,), lower_mu_unnorm.reshape(-1,)), axis=1)
                    #np.savetxt(f"./../Results/pred_csv/preds_{params['load_model_dir'].split('/')[3]}_{model_no}_{params['seed']}.csv", save_pred, delimiter=',')
                plot_one(mu_unnorm[:100], upper_mu_unnorm[:100], lower_mu_unnorm[:100], ground_truth[:100], file_name=f"model_{model_no}_pred.png")
            mu_rand = ensemble_ins.random_selection(mu_unnorm_all.reshape(params['num_models'],-1,1), mu_unnorm_all.shape[1])
            mses = np.square(np.subtract(ground_truth, mu_rand.detach().cpu().numpy()))
            aggr_var_dict = ensemble_ins.aggr_var(['max_aleotoric', 'ensemble_var', 'ensemble_std', 'll_var'], mu_rand,\
                                                  mu_unnorm_all, logvar_all, actual_mu_all)
            plot_mse_var(mses, aggr_var_dict, file_name="aggr_var.png")
        else:
            start_idx = 0
            end_idx = start_idx + 100
            for model_no, model in ensemble_ins.models.items():         
                saved_data = genfromtxt(params['saved_pred_csv']+f"preds_{params['load_model_dir'].split('/')[3]}_{model_no}_{params['seed']}.csv", delimiter=',')
                ground_truth, mu_unnorm, upper_mu_unnorm, lower_mu_unnorm =\
                    saved_data[:,0], saved_data[:,1], saved_data[:,2], saved_data[:,3]
                plot_one(mu_unnorm[start_idx:end_idx], upper_mu_unnorm[start_idx:end_idx], lower_mu_unnorm[start_idx:end_idx],\
                          ground_truth[start_idx:end_idx], file_name=f"model_{model_no}_pred.png")
    return             

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-se', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--yaml_file', '-yml', type=str, default=None)
    parser.add_argument('--num_models', '-nm', type=int, default=7)
    parser.add_argument('--train_val_ratio', type=float, default=0.2)
    parser.add_argument('--model_epochs', '-me', type=int, default=200)
    parser.add_argument('--model_lr', type=float, default=0.001, help='lr for Transition Model')
    parser.add_argument('--l2_reg_multiplier', type=float, default=1.)
    parser.add_argument('--min_model_epochs', type=int, default=None)
    parser.add_argument('--train_mode', type=bool, default=False)
    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--load_model_dir', type=str, default=None)
    parser.add_argument('--split_type', type=str, default='random') # random | min_max
    parser.add_argument('--save_pred', type=bool, default=False)
    parser.add_argument('--saved_pred_csv', type=str, default=None)
    parser.add_argument('--swag_sample_select', type=str, default='random') 

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
