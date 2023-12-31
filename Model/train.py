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
    params['input_data'] = np.array(orig_data[1:,1:7])
    params['output_data'] = np.array(orig_data[1:,7]).reshape(-1,1)
    params['no_of_inputs'] = params['input_data'].shape[1]
    params['no_of_outputs'] = 1


    ensemble_ins = Ensemble(params=params)

    if not params['matlab_inference']:         
        if params['train_mode']:
            ensemble_ins.calculate_mean_var()
            ensemble_ins.set_loaders()
            ensemble_ins.train_model(params['model_epochs'], save_model=True)
        if params['test_mode']:
            if not params['saved_pred_csv'] and not params['saved_aggr_mu_csv']:

                start, end = 13000, 13100
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
                    mu, logvar =  model.get_next_state(sorted_rand_input_filtered_val, \
                                                            deterministic=True, return_mean=True) # normalized validation data
                    mu_unnorm, upper_mu_unnorm, lower_mu_unnorm =  ensemble_ins.calculate_bounds(mu[0], logvar)
                    mu_unnorm_all[model_no,:] = torch.FloatTensor(mu_unnorm.reshape(-1,)).to(device)
                    logvar_all[model_no,:] = logvar.reshape(-1,)
                    actual_mu_all[model_no,:] = ensemble_ins.output_filter.invert_torch(mu[1].reshape(1,-1))
                    if params['save_all_pred']:
                        save_pred = np.stack((ground_truth.reshape(-1,), mu_unnorm.reshape(-1,),
                                            upper_mu_unnorm.reshape(-1,), lower_mu_unnorm.reshape(-1,)), axis=1)
                        np.savetxt(f"./../Results/pred_csv/preds_{params['load_model_dir'].split('/')[3]}_{model_no}_{params['seed']}.csv", save_pred, delimiter=',')
                    plot_one(mu_unnorm[start:end], upper_mu_unnorm[start:end], lower_mu_unnorm[start:end], ground_truth[start:end], file_name=f"model_{model_no}_pred_{start}_{end}.png")

                if params['mu_aggr_type'] == 'random':
                    mu_aggr = ensemble_ins.random_selection(mu_unnorm_all.reshape(params['num_models'],-1,1), mu_unnorm_all.shape[1])
                elif params['mu_aggr_type'] == 'mean':
                    mu_aggr = ensemble_ins.mean_selection(mu_unnorm_all.reshape(params['num_models'],-1))
                else:
                    raise NotImplementedError    

                mses = np.square(np.subtract(ground_truth, mu_aggr.detach().cpu().numpy()))
                aggr_var_dict = ensemble_ins.aggr_var(['max_aleotoric', 'ensemble_var', 'ensemble_std', 'll_var'], mu_aggr,\
                                                    mu_unnorm_all, logvar_all, actual_mu_all)
                
                if params['plot_se_var']:
                    for i in range(2): # a weird hack to get bigger fonts on the second plot
                        plot_mse_var(mses, aggr_var_dict, file_name=f"aggr_var{i}.png")

                aggr_mu, aggr_ub, aggr_lb = ensemble_ins.aggr_mu_bounds(mu_aggr.detach().cpu().numpy(), aggr_var_dict,\
                                                                        params['var_aggr_type'])
                if params['save_aggr_pred']:
                    save_pred = np.stack((ground_truth.reshape(-1,), aggr_mu.reshape(-1,),
                                            aggr_ub.reshape(-1,), aggr_lb.reshape(-1,)), axis=1)
                    np.savetxt(f"./../Results/pred_csv_aggr_min_max_norm/preds_{params['load_model_dir'].split('/')[3]}_{params['mu_aggr_type']}Mu_{params['var_aggr_type']}_{params['seed']}.csv", save_pred, delimiter=',')

            else:
                if params['saved_pred_csv']:
                    start_idx = 0
                    end_idx = start_idx + 100
                    for model_no, model in ensemble_ins.models.items():         
                        saved_data = genfromtxt(params['saved_pred_csv']+f"preds_{params['load_model_dir'].split('/')[3]}_{model_no}_{params['seed']}.csv", delimiter=',')
                        ground_truth, mu_unnorm, upper_mu_unnorm, lower_mu_unnorm =\
                            saved_data[:,0], saved_data[:,1], saved_data[:,2], saved_data[:,3]
                        plot_one(mu_unnorm[start_idx:end_idx], upper_mu_unnorm[start_idx:end_idx], lower_mu_unnorm[start_idx:end_idx],\
                                ground_truth[start_idx:end_idx], file_name=f"model_{model_no}_pred_{start_idx}.png")

                if params['saved_aggr_mu_csv']:

                    read_file = params['saved_aggr_mu_csv']+f"preds_{params['load_model_dir'].split('/')[3]}_{params['mu_aggr_type']}Mu_{params['var_aggr_type']}_{params['seed']}.csv"
                    start_idx = 0
                    end_idx = start_idx + 100
                    saved_data = genfromtxt(read_file, delimiter=',')
                    ground_truth, aggr_mu, aggr_ub, aggr_lb =\
                            saved_data[:,0], saved_data[:,1], saved_data[:,2], saved_data[:,3]
                    for i in range(2):
                        plot_one(aggr_mu[start_idx:end_idx], aggr_ub[start_idx:end_idx], aggr_lb[start_idx:end_idx], ground_truth[start_idx:end_idx],\
                                    file_name=f"pred_csv_aggr_min_max_norm/aggr_{start_idx}to{end_idx}_{params['mu_aggr_type']}_{params['var_aggr_type']}_pred.png")            
    else:

        ensemble_ins.load_model(params['load_model_dir'])
        input_features = np.array(params['input_features'])
        if len(input_features.shape) == 1:
            input_features = input_features.reshape(1,-1)
        filtered_input = ensemble_ins.input_filter.filter(input_features)

        assert params['num_models'] > 0, "No. of models should at least be 1"

        mu_unnorm_all = torch.zeros(params['num_models'], input_features.shape[0], device=device)
        logvar_all = torch.zeros(params['num_models'], input_features.shape[0], device=device)
        actual_mu_all = torch.zeros(params['num_models'], input_features.shape[0], device=device)

        for model_no in range(params['num_models']):
            mu, logvar =  ensemble_ins.models[model_no].get_next_state(filtered_input, deterministic=True, return_mean=True)
            mu_unnorm, _, _ =  ensemble_ins.calculate_bounds(mu[0], logvar)
            mu_unnorm_all[model_no,:] = torch.FloatTensor(mu_unnorm.reshape(-1,)).to(device)
            logvar_all[model_no,:] = logvar.reshape(-1,)
            actual_mu_all[model_no,:] = ensemble_ins.output_filter.invert_torch(mu[1].reshape(1,-1))

        if params['mu_aggr_type'] == 'random':
            mu_aggr = ensemble_ins.random_selection(mu_unnorm_all.reshape(params['num_models'],-1,1), mu_unnorm_all.shape[1])
        elif params['mu_aggr_type'] == 'mean':
            mu_aggr = ensemble_ins.mean_selection(mu_unnorm_all.reshape(params['num_models'],-1)) 

        aggr_var_dict = ensemble_ins.aggr_var(['max_aleotoric', 'ensemble_var', 'ensemble_std', 'll_var'], mu_aggr,\
                                    mu_unnorm_all, logvar_all, actual_mu_all)       
        
        if params['num_models'] == 1:
            #print(mu_aggr.detach().cpu().numpy().reshape(-1,))
            #print(logvar_all.exp().detach().cpu().numpy().reshape(-1,))
            return mu_aggr.detach().cpu().numpy().reshape(-1,), logvar_all.exp().detach().cpu().numpy().reshape(-1,) # sending mu_aggr since mu[0] is unnormalized
        else:
            #print(mu_aggr.detach().cpu().numpy().reshape(-1,))
            #print(aggr_var_dict['ensemble_var'])
            return mu_aggr.detach().cpu().numpy().reshape(-1,), aggr_var_dict['ensemble_var']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-se', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--yaml_file', '-yml', type=str, default=None)
    parser.add_argument('--train_val_ratio', type=float, default=0.2)
    parser.add_argument('--model_epochs', '-me', type=int, default=200)
    parser.add_argument('--model_lr', type=float, default=0.001, help='lr for Transition Model')
    parser.add_argument('--l2_reg_multiplier', type=float, default=1.)
    parser.add_argument('--min_model_epochs', type=int, default=None)
    parser.add_argument('--train_mode', type=bool, default=False)
    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--load_model_dir', type=str, default=None)
    parser.add_argument('--split_type', type=str, default='random', choices=['random', 'min_max'])
    parser.add_argument('--mu_aggr_type', type=str, default='random', choices=['random', 'mean']) 
    parser.add_argument('--var_aggr_type', type=str, default='ensemble_std', choices=['ensemble_std', 'ensemble_var', 'll_var', 'max_aleotoric']) 
    parser.add_argument('--save_all_pred', type=bool, default=False)
    parser.add_argument('--save_aggr_pred', type=bool, default=False)
    parser.add_argument('--saved_pred_csv', type=str, default=None)
    parser.add_argument('--swag_sample_select', type=str, default='random') 
    parser.add_argument('--plot_se_var', type=bool, default=False, help='plot squared_error vs aggregated var')
    parser.add_argument('--saved_aggr_mu_csv', type=str, default=False, help='plot the aggregated mean and var')
    # parameters for inference with MATLAB
    parser.add_argument('--matlab_inference', type=bool, default=False, help='True while doing inference with matlab simulator')
    parser.add_argument('--input_features', type=list, default=None, help='input example to be predicted') # shape = [1,4]
    parser.add_argument('--num_models', type=int, default=7, help='how many nn models to use for ensemble')

    args = parser.parse_args()
    params = vars(args)

    if params['yaml_file']:
        with open(args.yaml_file, 'r') as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            for config in yaml_config['args']:
                if config in params:
                    params[config] = yaml_config['args'][config]

    if not params['matlab_inference']:
        train(params)
    else:    
        mu, var = train(params)               

if __name__ == '__main__':
    main()
