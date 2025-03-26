import os
import ast
import time
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor, ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


# Set a seed for reproducibility of random results
np.random.seed(0)

# This function performs 3-objective non-dominated sorting.
# It takes in 3 sets of values (accuracy, latency, and memory) and sorts them based on their dominance.
# 'values1', 'values2', 'values3' are the objectives that need to be minimized.
def non_dominated_sorting_algorithm_3d(values1, values2, values3):
    # Initialize S to store dominated sets and n to store the number of solutions dominating a given solution.
    S = [[] for _ in range(len(values1))]
    front = [[]]  # Initialize the first front (dominance level 0)
    n = [0 for _ in range(len(values1))]  # Counter for the number of dominating solutions
    rank = [0 for _ in range(len(values1))]  # Rank for each solution

    # Compare each solution 'p' with every other solution 'q' to determine dominance relationships
    for p in range(len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(len(values1)):
            # Check if solution p dominates solution q in all three objectives
            if (values1[p] < values1[q] and values2[p] < values2[q] and values3[p] < values3[q]) or \
               (values1[p] <= values1[q] and values2[p] < values2[q] and values3[p] < values3[q]) or \
               (values1[p] < values1[q] and values2[p] <= values2[q] and values3[p] < values3[q]) or \
               (values1[p] < values1[q] and values2[p] < values2[q] and values3[p] <= values3[q]):
                # If p dominates q, add q to the list of solutions dominated by p
                if q not in S[p]:
                    S[p].append(q)
            # Otherwise, check if solution q dominates p
            elif (values1[q] < values1[p] and values2[q] < values2[p] and values3[q] < values3[p]) or \
                 (values1[q] <= values1[p] and values2[q] < values2[p] and values3[q] < values3[p]) or \
                 (values1[q] < values1[p] and values2[q] <= values2[p] and values3[q] < values3[p]) or \
                 (values1[q] < values1[p] and values2[q] < values2[p] and values3[q] <= values3[q]):
                # If q dominates p, increment the domination count for p
                n[p] += 1
        
        # If no solution dominates p, add p to the first front (rank 0)
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    
    # Create subsequent fronts by updating ranks based on dominance relationships
    i = 0
    while front[i]:
        Q = []  # Store the next front
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1  # Decrease the domination count for q
                if n[q] == 0:
                    rank[q] = i + 1  # Set the rank for q
                    if q not in Q:
                        Q.append(q)
        i += 1
        front.append(Q)
    
    # Remove the last empty front and return the list of sorted fronts
    del front[-1]
    return front

# This function calculates the Approximate Distance to the Reference Set (ADRS) in 3D space.
# It compares the solution set (xs, ys, zs) with the reference set (xr, yr, zr) across three objectives.
def ADRS_3d(xr, yr, zr, xs, ys, zs):
    # Calculate weights for each objective based on the range of values in the reference set
    w_x = 1 / (np.max(xr) - np.min(xr))  # Weight for the x-axis objective
    w_y = 1 / (np.max(yr) - np.min(yr))  # Weight for the y-axis objective
    w_z = 1 / (np.max(zr) - np.min(zr))  # Weight for the z-axis objective

    dist = 0  # Initialize distance accumulator

    # Loop through each point in the reference set
    for i in range(len(xr)):
        # Calculate the weighted distance between each solution and the reference point
        c_x = w_x * abs(xs - xr[i])
        c_y = w_y * abs(ys - yr[i])
        c_z = w_z * abs(zs - zr[i])  # Include z-axis distance

        # Stack the distances along each axis and find the maximum for each solution
        zeros = np.zeros(len(xs))  # Used to initialize arrays
        stacked = np.stack((c_x, c_y, c_z))  # Stack the distances
        c_max = np.max(stacked, axis=0)  # Find the maximum distance along any axis

        # Find the minimum distance among all solutions and accumulate it
        dist += np.min(c_max)
    
    # Average the accumulated distances and return the result
    dist /= len(xr)
    return dist


# Set a seed for reproducibility of random results
np.random.seed(0)

# This function performs 3-objective non-dominated sorting.
# It takes in 3 sets of values (accuracy, latency, and memory) and sorts them based on their dominance.
# 'values1', 'values2', 'values3' are the objectives that need to be minimized.
def non_dominated_sorting_algorithm_3d(values1, values2, values3):
    # Initialize S to store dominated sets and n to store the number of solutions dominating a given solution.
    S = [[] for _ in range(len(values1))]
    front = [[]]  # Initialize the first front (dominance level 0)
    n = [0 for _ in range(len(values1))]  # Counter for the number of dominating solutions
    rank = [0 for _ in range(len(values1))]  # Rank for each solution

    # Compare each solution 'p' with every other solution 'q' to determine dominance relationships
    for p in range(len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(len(values1)):
            # Check if solution p dominates solution q in all three objectives
            if (values1[p] < values1[q] and values2[p] < values2[q] and values3[p] < values3[q]) or \
               (values1[p] <= values1[q] and values2[p] < values2[q] and values3[p] < values3[q]) or \
               (values1[p] < values1[q] and values2[p] <= values2[q] and values3[p] < values3[q]) or \
               (values1[p] < values1[q] and values2[p] < values2[q] and values3[p] <= values3[q]):
                # If p dominates q, add q to the list of solutions dominated by p
                if q not in S[p]:
                    S[p].append(q)
            # Otherwise, check if solution q dominates p
            elif (values1[q] < values1[p] and values2[q] < values2[p] and values3[q] < values3[p]) or \
                 (values1[q] <= values1[p] and values2[q] < values2[p] and values3[q] < values3[p]) or \
                 (values1[q] < values1[p] and values2[q] <= values2[p] and values3[q] < values3[p]) or \
                 (values1[q] < values1[p] and values2[q] < values2[p] and values3[q] <= values3[q]):
                # If q dominates p, increment the domination count for p
                n[p] += 1
        
        # If no solution dominates p, add p to the first front (rank 0)
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    
    # Create subsequent fronts by updating ranks based on dominance relationships
    i = 0
    while front[i]:
        Q = []  # Store the next front
        for p in front[i]:
            for q in S[p]:
                n[q] -= 1  # Decrease the domination count for q
                if n[q] == 0:
                    rank[q] = i + 1  # Set the rank for q
                    if q not in Q:
                        Q.append(q)
        i += 1
        front.append(Q)
    
    # Remove the last empty front and return the list of sorted fronts
    del front[-1]
    return front

# This function calculates the Approximate Distance to the Reference Set (ADRS) in 3D space.
# It compares the solution set (xs, ys, zs) with the reference set (xr, yr, zr) across three objectives.
def ADRS_3d(xr, yr, zr, xs, ys, zs):
    # Calculate weights for each objective based on the range of values in the reference set
    w_x = 1 / (np.max(xr) - np.min(xr))  # Weight for the x-axis objective
    w_y = 1 / (np.max(yr) - np.min(yr))  # Weight for the y-axis objective
    w_z = 1 / (np.max(zr) - np.min(zr))  # Weight for the z-axis objective

    dist = 0  # Initialize distance accumulator

    # Loop through each point in the reference set
    for i in range(len(xr)):
        # Calculate the weighted distance between each solution and the reference point
        c_x = w_x * abs(xs - xr[i])
        c_y = w_y * abs(ys - yr[i])
        c_z = w_z * abs(zs - zr[i])  # Include z-axis distance

        # Stack the distances along each axis and find the maximum for each solution
        zeros = np.zeros(len(xs))  # Used to initialize arrays
        stacked = np.stack((c_x, c_y, c_z))  # Stack the distances
        c_max = np.max(stacked, axis=0)  # Find the maximum distance along any axis

        # Find the minimum distance among all solutions and accumulate it
        dist += np.min(c_max)
    
    # Average the accumulated distances and return the result
    dist /= len(xr)
    return dist



from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import SGDRegressor

# This function generates different hyperparameters for GradientBoostingRegressor, trains the models,
# evaluates them on the validation set, and saves the prediction results to CSV files.
def gb_regressors(benchmark, X_train, X_val, y_train, y_val, 
                  losses=["squared_error", "huber", "quantile"], 
                  learning_rates=[0.1, 0.001, 0.0001], 
                  n_estimators_list=[10, 100, 1000]):

    # Calculate total number of combinations for progress bar
    total_combinations = len(losses) * len(learning_rates) * len(n_estimators_list)
    
    # Initialize progress bar
    with tqdm(total=total_combinations) as pbar:
        # Iterate through all combinations of hyperparameters
        for loss in losses:
            for learning_rate in learning_rates:
                for n_estimators in n_estimators_list:
                    # Initialize regressors for each label (accuracy, latency, memory)
                    regressors = [GradientBoostingRegressor(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators) for _ in range(3)]
                    
                    # Train each regressor for each output
                    for i in range(3):
                        regressors[i].fit(X_train, y_train[:, i])
                    
                    # Make predictions on the validation set
                    predictions = np.column_stack([regressor.predict(X_val) for regressor in regressors])
                    
                    # Create a DataFrame with the predictions
                    df_predictions = pd.DataFrame(predictions, columns=['accuracy', 'latency', 'memory'])
                    
                    # Create the filename based on the hyperparameters
                    filename = f"tuner/{benchmark}/GB_loss={loss}_lr={learning_rate}_n={n_estimators}.csv"
                    
                    # Save the predictions to a CSV file
                    df_predictions.to_csv(filename, index=False)
                    
                    # Update progress bar
                    pbar.set_description(f"GB: Loss={loss}, LR={learning_rate}, Est={n_estimators}")
                    pbar.update(1)

# This function generates different hyperparameters for RandomForestRegressor, trains the models,
# evaluates them on the validation set, and saves the prediction results to CSV files.
def rf_regressors(benchmark, X_train, X_val, y_train, y_val, 
                  n_estimators_list=[10, 100, 500, 1000], 
                  max_depth_list=['None', 4, 32, 256]):

    # Calculate total number of combinations for progress bar
    total_combinations = len(n_estimators_list) * len(max_depth_list)
    
    # Initialize progress bar
    with tqdm(total=total_combinations) as pbar:
        # Iterate through all combinations of hyperparameters
        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                # Convert 'None' string to actual None type for max_depth
                depth = None if max_depth == 'None' else max_depth
                
                # Initialize regressors for each label (accuracy, latency, memory)
                regressors = [RandomForestRegressor(n_estimators=n_estimators, max_depth=depth) for _ in range(3)]
                
                # Train each regressor for each output
                for i in range(3):
                    regressors[i].fit(X_train, y_train[:, i])
                
                # Make predictions on the validation set
                predictions = np.column_stack([regressor.predict(X_val) for regressor in regressors])
                
                # Create a DataFrame with the predictions
                df_predictions = pd.DataFrame(predictions, columns=['accuracy', 'latency', 'memory'])
                
                # Create the filename based on the hyperparameters
                filename = f"tuner/{benchmark}/RF_n={n_estimators}_depth={max_depth}.csv"
                
                # Save the predictions to a CSV file
                df_predictions.to_csv(filename, index=False)
                
                # Update progress bar
                pbar.set_description(f"RF: Est={n_estimators}, Depth={max_depth}")
                pbar.update(1)

# This function generates different hyperparameters for XGBRegressor, trains the models,
# evaluates them on the validation set, and saves the prediction results to CSV files.
def xgb_regressors(benchmark, X_train, X_val, y_train, y_val, 
                   n_estimators_list=[10, 100, 500, 1000], 
                   max_depth_list=['None', 4, 32, 256]):
    
    # Calculate total number of combinations for progress bar
    total_combinations = len(n_estimators_list) * len(max_depth_list)
    
    # Initialize progress bar
    with tqdm(total=total_combinations) as pbar:
        # Iterate through all combinations of hyperparameters
        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                # Convert 'None' string to actual None type for max_depth
                depth = None if max_depth == 'None' else max_depth
                
                # Initialize regressors for each label (accuracy, latency, memory)
                regressors = [XGBRegressor(n_estimators=n_estimators, max_depth=depth) for _ in range(3)]
                
                # Train each regressor for each output
                for i in range(3):
                    regressors[i].fit(X_train, y_train[:, i])
                
                # Make predictions on the validation set
                predictions = np.column_stack([regressor.predict(X_val) for regressor in regressors])
                
                # Create a DataFrame with the predictions
                df_predictions = pd.DataFrame(predictions, columns=['accuracy', 'latency', 'memory'])
                
                # Create the filename based on the hyperparameters
                filename = f"tuner/{benchmark}/XGB_n={n_estimators}_depth={max_depth}.csv"
                
                # Save the predictions to a CSV file
                df_predictions.to_csv(filename, index=False)
                
                # Update progress bar
                pbar.set_description(f"XGB: Est={n_estimators}, Depth={max_depth}")
                pbar.update(1)

# This function generates different hyperparameters for SGDRegressor, trains the models,
# evaluates them on the validation set, and saves the prediction results to CSV files.
def sgd_regressors(benchmark, X_train, X_val, y_train, y_val,
                   losses=["squared_error", "huber"],
                   penalties=['l2', 'l1', 'elasticnet', None]):
    
    # Calculate total number of combinations for progress bar
    total_combinations = len(losses) * len(penalties)
    
    # Initialize progress bar
    with tqdm(total=total_combinations) as pbar:
        # Iterate through all combinations of hyperparameters
        for loss in losses:
            for penalty in penalties:
                # Initialize regressors for each label (accuracy, latency, memory)
                regressors = [SGDRegressor(loss=loss, penalty=penalty, max_iter=5000) for _ in range(3)]
                
                # Train each regressor for each output
                for i in range(3):
                    regressors[i].fit(X_train, y_train[:, i])
                
                # Make predictions on the validation set
                predictions = np.column_stack([regressor.predict(X_val) for regressor in regressors])
                
                # Create a DataFrame with the predictions
                df_predictions = pd.DataFrame(predictions, columns=['accuracy', 'latency', 'memory'])
                
                # Create the filename based on the hyperparameters
                penalty_str = 'None' if penalty is None else penalty
                filename = f"tuner/{benchmark}/SGD_loss={loss}_penalty={penalty_str}.csv"
                
                # Save the predictions to a CSV file
                df_predictions.to_csv(filename, index=False)
                
                # Update progress bar
                pbar.set_description(f"SGD: Loss={loss}, Penalty={penalty_str}")
                pbar.update(1)





# This function performs a regression search over multiple models (GB, RF, XGBoost, and SGD) 
# with different hyperparameter combinations.
def reg_search(directory, X_train, X_val, y_train, y_val):
    print("Regressors tranining in progress...")
    # Define the hyperparameters for Gradient Boosting Regressors
    loss = ["squared_error", "huber", "quantile"]  # Loss functions for Gradient Boosting
    learning_rate = [0.1, 0.001, 0.0001]  # Learning rates to try
    n_estimators = [10, 100, 1000]  # Number of estimators (trees) in the ensemble
    
    # Train Gradient Boosting Regressors with different hyperparameter combinations
    gb_regressors(directory, X_train, X_val, y_train, y_val, loss, learning_rate, n_estimators)

    # Define the hyperparameters for Random Forest Regressors
    n_estimators = [10, 100, 500, 1000]  # Number of trees in the forest
    max_depth = ['None', 4, 32, 256]  # Maximum depth of the tree
    
    # Train Random Forest Regressors with different hyperparameter combinations
    rf_regressors(directory, X_train, X_val, y_train, y_val, n_estimators, max_depth)
    
    # Define the hyperparameters for XGBoost Regressors
    n_estimators = [10, 100, 500, 1000]  # Number of boosting rounds
    max_depth = ['None', 4, 32, 256]  # Maximum depth of a tree
    
    # Train XGBoost Regressors with different hyperparameter combinations
    xgb_regressors(directory, X_train, X_val, y_train, y_val, n_estimators, max_depth)

    # Define the hyperparameters for SGD Regressors
    loss = ["squared_error", "huber"]  # Loss functions for Stochastic Gradient Descent
    penalty = ['l2', 'l1', 'elasticnet', None]  # Penalty (regularization term) to try
    
    # Train SGD Regressors with different hyperparameter combinations
    sgd_regressors(directory, X_train, X_val, y_train, y_val, loss, penalty)



# This function reads the predictions from the corresponding CSV file and evaluates them based on MSE and MAPE.
def evaluate_predictions(directory, y_val):
    results = []
    
    # List all CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Read the predictions
            df_predictions = pd.read_csv(os.path.join(directory, filename))
            
            # Extract model and hyperparameters from filename
            params = filename.split('_')
            model_type = params[0]
            hyperparams = "_".join(params[1:]).replace('.csv', '')
            

            mse_accuracy = mean_squared_error(y_val[:, 0], df_predictions['accuracy'])
            mse_latency = mean_squared_error(y_val[:, 1], df_predictions['latency'])
            mse_memory = mean_squared_error(y_val[:, 2], df_predictions['memory'])
            
            mape_accuracy = mean_absolute_percentage_error(y_val[:, 0], df_predictions['accuracy'])
            mape_latency = mean_absolute_percentage_error(y_val[:, 1], df_predictions['latency'])
            mape_memory = mean_absolute_percentage_error(y_val[:, 2], df_predictions['memory'])
            
            # Append the evaluation results to the results list
            results.append({
                'model_type': model_type,
                'hyperparams': hyperparams,
                'mse_accuracy': mse_accuracy,
                'mse_latency': mse_latency,
                'mse_memory': mse_memory,
                'mape_accuracy': mape_accuracy,
                'mape_latency': mape_latency,
                'mape_memory': mape_memory
            })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df


# This function returns the top 5 predictions with the lowest MAPE for the specified metric (accuracy, latency, or memory).
def get_top_5_predictions(results_df, metric):
    return results_df.nsmallest(5, f'mape_{metric}')


# This function collects the predictions from the top models based on the DataFrame of top models.
def collect_predictions(directory, top_models):
    predictions = []
    # Iterate over the rows of the top_models DataFrame
    for _, row in top_models.iterrows():
        # Construct the filename based on the model type and hyperparameters
        filename = f"{row['model_type']}_{row['hyperparams']}.csv"
        filepath = os.path.join(directory, filename)
        
        # Read the predictions from the corresponding CSV file and append to predictions list
        df_predictions = pd.read_csv(filepath)
        predictions.append(df_predictions)
    return predictions


# Main function to evaluate predictions and identify the top triplets of models based on ADRS.
def evaluate_and_identify_top_triplets(directory, y_val, top_n=5):
    # Evaluate all predictions and get results as a DataFrame
    print("Sorting the triplets...")
    results_df = evaluate_predictions(directory, y_val)
    
    # Define the real values of the target metrics
    real_lat = y_val[:, 1]
    real_acc = -1 * y_val[:, 0]  # Inverting accuracy for non-dominated sorting
    real_mem = y_val[:, 2]
    
    # Extract the top 5 predictors based on MAPE for each metric (accuracy, latency, memory)
    top_accuracy = get_top_5_predictions(results_df, 'accuracy')
    top_latency = get_top_5_predictions(results_df, 'latency')
    top_memory = get_top_5_predictions(results_df, 'memory')
    
    # Collect predictions from the top models
    accuracy_predictions = collect_predictions(directory, top_accuracy)
    latency_predictions = collect_predictions(directory, top_latency)
    memory_predictions = collect_predictions(directory, top_memory)
    
    # List to store ADRS scores and corresponding models
    adrs_scores = []
    
    # Calculate the total number of combinations for the progress bar (5x5x5 = 125)
    total_combinations = len(accuracy_predictions) * len(latency_predictions) * len(memory_predictions)
    
    # Initialize progress bar
    with tqdm(total=total_combinations) as pbar:
        # Generate 125 distinct predictor triplet combinations (5x5x5)
        for acc_pred, acc_row in zip(accuracy_predictions, top_accuracy.iterrows()):
            for lat_pred, lat_row in zip(latency_predictions, top_latency.iterrows()):
                for mem_pred, mem_row in zip(memory_predictions, top_memory.iterrows()):

                    # Obtain the non-dominated front of real values using non-dominated sorting
                    idx = non_dominated_sorting_algorithm_3d(real_acc, real_lat, real_mem)
                    
                    # Obtain the non-dominated front of predicted values using non-dominated sorting
                    idx_pred = non_dominated_sorting_algorithm_3d(
                        acc_pred['accuracy'], lat_pred['latency'], mem_pred['memory']
                    )

                    # Calculate the ADRS score between the predicted front and the real front
                    adrs = ADRS_3d(
                        real_acc[idx[0]], real_lat[idx[0]], real_mem[idx[0]],
                        real_acc[idx_pred[0]], real_lat[idx_pred[0]], real_mem[idx_pred[0]]
                    )
                    
                    # Calculate the sum of squared MSE and MAPE across the triplet
                    mse = np.square(acc_row[1]['mse_accuracy']) + np.square(lat_row[1]['mse_latency']) + np.square(mem_row[1]['mse_memory'])
                    mape = np.square(acc_row[1]['mape_accuracy']) + np.square(lat_row[1]['mape_latency']) + np.square(mem_row[1]['mape_memory'])
                    
                    # Append the ADRS score and model details to the adrs_scores list
                    adrs_scores.append({
                        'adrs': adrs,
                        'front': idx_pred[0],
                        'accuracy_model': acc_row[1]['model_type'],
                        'accuracy_hyperparams': acc_row[1]['hyperparams'],
                        'latency_model': lat_row[1]['model_type'],
                        'latency_hyperparams': lat_row[1]['hyperparams'],
                        'memory_model': mem_row[1]['model_type'],
                        'memory_hyperparams': mem_row[1]['hyperparams'],
                        'mse': mse,
                        'mape': mape
                    })

                    # Update progress bar
                    pbar.update(1)
    
    # Convert the ADRS scores to a DataFrame
    adrs_df = pd.DataFrame(adrs_scores)

    # Sort the triplets based on ADRS, MSE, and MAPE (all in ascending order)
    adrs_df = adrs_df.sort_values(by=['adrs', 'mse', 'mape'], ascending=[True, True, True])
    
    # Save the sorted ADRS scores to a CSV file
    print("Top triplets list saved at "+ directory + "/adrs")
    adrs_df.to_csv(directory + "/adrs", index=False)
    
    # Return the top_n triplets with the best ADRS scores
    top_adrs = adrs_df.nsmallest(top_n, 'adrs')
    return top_adrs



def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run a regression search and evaluate top triplets based on a CSV dataset.')
    
    # Define arguments with defaults and help text
    parser.add_argument('--dataset', type=str, required=True, default='data/gpt_l_rtx3080.csv',
                        help='Path to the benchmark CSV containing the accuracy, latency and memory dataset. Default is data/gpt_l_rtx3080.csv.')
    
    parser.add_argument('--dir', type=str, required=True, default='GPT/gpt_l/rtx3080',
                        help='The path to the directory to save the ADRS tuner results. Default is GPT/gpt_l/rtx3080.')
    
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Ratio of the data to be used for training. Default is 0.1.')
    
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of the training data to be used for validation. Default is 0.1.')
    
    # Parse the arguments
    args = parser.parse_args()
    
    #  Load the data
    data = pd.read_csv(args.dataset)
    X_raw = data['vector']
    X = np.array([ast.literal_eval(x) for x in X_raw])
    memory = data['mem']
    latency = data['latency']
    accuracy = data['accuracy']
    
    # Prepare the feature matrix X and target matrix y
    y = np.vstack((accuracy, latency, memory)).T

    # Split the data into training/validation/test sets
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=1-args.train_ratio, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=args.val_ratio, random_state=0)

    # Run the regression search
    reg_search(args.dir, X_train, X_val, y_train, y_val)
    
    # Evaluate and identify the top triplets
    top_triplets = evaluate_and_identify_top_triplets("tuner/" + args.dir, y_val, top_n=5)#
    

if __name__ == "__main__":
    main()


