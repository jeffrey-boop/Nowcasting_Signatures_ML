import matplotlib as mpl
from pathlib import Path
import yaml

plot_config_path = Path(__file__).resolve().parent/'plot_configs.yaml'
with open(plot_config_path, 'r') as stream:
    rc_fonts = yaml.safe_load(stream)
rc_fonts['figure.figsize'] = tuple(rc_fonts['figure.figsize'])
        
mpl.rcParams.update(rc_fonts)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
def find_rmse(true_values, predictions):
    '''
    Finds the RMSE between two Series
    
    Parameters  true_values: pandas Series of true values 
                predictions: pandas Series of predicted values 
    Returns     rmse: float, the root mean square error 
    
    '''
    rss = np.sum((true_values-predictions)**2)
    rmse = np.sqrt(rss/(len(true_values)))
    
    return rmse


def plot_config(config_folder, experiment_list, start_date, end_date, save_dir=None, 
                filename=None, input_dir=None, results_dir=None):
    
    '''
    plots the results for a particular experiment_list
    
    Parameters  config_folder: string giving the folder name of the specific 
                               config was saved in
                experiment_list: list of tuples specifying each experiment_name and corresponding label
                start_date: pandas datetime object indicating the start date
                end_date: pandas datetime object indicating the end date
                save_dir: Path object indicating directory to save to, if left
                          as default None, then the plot is not saved, and the
                          filename is redundent
                filename: string specifying the filename 
                input_dir: Path giving the directory of the data, defaults to
                           a folder called "data"
                results_dir: Path of where the results are stored, defaults to
                             a folder called "results"
    Returns     df_rmse: a dataframe of rmse
    
    '''
    
    if not input_dir:
        input_dir = Path(__file__).resolve().parent.parent/'data'
    if not results_dir:
        results_dir = Path(__file__).resolve().parent.parent/'results'
    
    # Load results for the DFM baseline
    dfm_file = results_dir/'dfm'/'dfm_predictions.yaml'

    with open(dfm_file, 'r') as stream:
            dfm_results = yaml.safe_load(stream)
    df_dfm = pd.DataFrame(dfm_results.items(), columns=['date', 
                                                        'dfm_prediction'])
    df_dfm['date'] = pd.to_datetime(df_dfm['date'], format="%Y-%m-%d")
    df_dfm = df_dfm[(df_dfm['date'] >= start_date) & (df_dfm['date'] <= end_date)]
    
    all_dfs = []
    true_column = None

    for experiment, label in experiment_list:
        # Load results for each new model
        filtered_df = load_df(results_dir, experiment, config_folder, start_date, end_date)

        # Plot the true GDP values if it hasn't been plotted already
        if (true_column is None) and ('true_value' in filtered_df.columns):
            true_column = filtered_df['true_value']
            plot_df(filtered_df, label='True Value', pred_column = 'true_value', point_size = 8, line_width = 3, marker=None)

        # Plot each new model results and append results to list
        plot_df(filtered_df, label)
        all_dfs.append((filtered_df, label))

    # Plot DFM baseline results and append results to list
    all_dfs.append((df_dfm, 'Model 5 Baseline'))
    plot_df(df_dfm, 'Model 5 Baseline', 'dfm_prediction')  
    
    plt.xticks(rotation=30)
    plt.ylabel('GDP Growth (%)', fontsize='small')
    plt.legend(loc=2, fontsize=10)

    # Save plot
    if save_dir:
        if not filename:
            filename = 'USGDP_results'
        
        plt.savefig(f'{save_dir}/{filename}.pdf', bbox_inches='tight')
    
    plt.show()
    
    # Calculate RMSE for each model
    all_rmse = {}
    for df, label in all_dfs:
        pred_column = 'prediction' if 'prediction' in df.columns else 'dfm_prediction'
        rmse = find_rmse(true_column, df[pred_column])
        all_rmse[label] = rmse

    df_rmse = pd.DataFrame(all_rmse.items(), columns=['method', 'rmse'])
    
    return df_rmse

def load_df(results_dir, experiment, config_folder, start_date, end_date):
    '''
    Load and filter US GDP results based on the date range
    
    Parameters:
    - results_dir: Path of where the results are stored
    - experiment: string indicating experiment name of current data
    - config_folder: string giving the folder name of the specific 
                               config was saved in
    - start_date: pandas datetime object indicating the start date
    - end_date: pandas datetime object indicating the end date
    Returns     filtered_df: dataframe with the filtered date range
    '''

    current_df = pd.read_csv(results_dir/experiment/'pca'/'rerun'/config_folder\
                        /'all_predictions.csv')
    current_df['date'] = pd.to_datetime(current_df['date'], format="%Y_%m_%d")
    filtered_df = current_df[(current_df['date'] >= start_date) & (current_df['date'] <= end_date)]
    return filtered_df

def plot_df(filtered_df, label, pred_column = 'prediction', point_size = 8, line_width = 2, marker='.'):
    '''
    Plot US GDP results based on a pandas dataframe
    
    Parameters:
    - filtered_df: dataframe containing data to be plotted
    - label: string to be used as label in plot legend
    - pred_column: string indicating the column name of results in the dataframe
    - point_size: string indicating the size of points on the line in the plot
    - line_width: string indicating the width of the line
    - marker: string indicating the type of marker to use on the line
    '''
    plt.step(filtered_df['date'], filtered_df[pred_column],
        marker=marker, label=label, markersize=point_size, linewidth=line_width, where='post')