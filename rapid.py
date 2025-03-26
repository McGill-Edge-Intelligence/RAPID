import os
import sys
import numpy as np
import pandas as pd
import ast
import json
import argparse
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor, ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor

np.random.seed(0)



def parse_hyperparams(hyperparams):
    # Initialize an empty dictionary to store the parsed hyperparameters
    params = {}
    
    
    # Split the hyperparameters string by underscores to separate each key-value pair
    for param in hyperparams.split('_'):
        # Skip processing if the parameter string is 'error'
        if param == 'error':
            pass  # No action is taken for 'error'
        else:
            # Split each parameter by '=' to get the key and value
            k, v = param.split('=')

            # Handle special cases for parameter values
            if v == 'None':
                params[k] = None  # Assign None if the value is 'None'
            elif v == 'squared':
                params[k] = 'squared_error'  # Replace 'squared' with 'squared_error'
            elif k in ['n', 'depth']:
                params[k] = int(v)  # Convert to an integer if the key is 'n' or 'depth'
            else:
                try:
                    # Attempt to convert the value to a float
                    params[k] = float(v)
                except ValueError:
                    # If conversion to float fails, assign the value as a string
                    params[k] = v
    
    # Return the parsed hyperparameters dictionary
    return params

    


def initialize_model(model_type, hyperparams):
    # Parse the hyperparameters string into a dictionary of parameter values
    params = parse_hyperparams(hyperparams)

    # Check the model type and initialize the corresponding model with parsed hyperparameters
    if model_type == 'XGB':
        print(f"XGBRegressor(n_estimators={params.get('n')}, max_depth={params.get('depth')})")
        # Return an initialized XGBRegressor with n_estimators and max_depth
        return XGBRegressor(n_estimators=params.get('n'), max_depth=params.get('depth'))

    elif model_type == 'RF':
        print(f"RandomForestRegressor(n_estimators={params.get('n')}, max_depth={params.get('depth')})")
        # Return an initialized RandomForestRegressor with n_estimators and max_depth
        return RandomForestRegressor(n_estimators=params.get('n'), max_depth=params.get('depth'))

    elif model_type == 'GB':
        print(f"GradientBoostingRegressor(n_estimators={params.get('n')}, learning_rate={params.get('lr')}, loss={params.get('loss')})")
        # Return an initialized GradientBoostingRegressor with n_estimators, learning_rate, and loss
        return GradientBoostingRegressor(n_estimators=params.get('n'), learning_rate=params.get('lr'), loss=params.get('loss'))

    else:
        print(f"SGDRegressor(loss={params.get('loss')}, penalty={params.get('penalty')}, max_iter=5000)")
        # Return an initialized SGDRegressor with loss, penalty, and fixed max_iter
        return SGDRegressor(loss=params.get('loss'), penalty=params.get('penalty'), max_iter=5000)

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

def errorBand(percentage, y_test, pred):
    # Initialize an empty list to store whether each prediction is within the error band
    vec = []
    
    # Iterate over the true values (y_test) and corresponding predicted values (pred)
    for idx, v in enumerate(y_test):
        # Check if the prediction is within the acceptable error band (percentage)
        # The acceptable range is [v - percentage% of v, v + percentage% of v]
        if (pred[idx] < v + v * (percentage / 100)) and (pred[idx] > v - v * (percentage / 100)):
            # If the prediction is within the band, append 1 to the list
            vec.append(1)
        else:
            # If the prediction is outside the band, append 0 to the list
            vec.append(0)
    
    # Calculate the percentage of predictions within the error band
    # Count how many 1's are in vec (i.e., how many predictions were within the band)
    # Divide by the total number of predictions and multiply by 100 to get a percentage
    return vec.count(1) / len(pred) * 100

def predLatency(l_model, X, y, X_tv, y_tv):
    # Train the latency model using the training data (X_tv and y_tv[:,1])
    l_model.fit(X_tv, y_tv[:, 1])
    
    # Predict the latency values for the test set X
    pred_lat = l_model.predict(X)
    
    # Calculate and print the percentage of predictions within error bands of 1%, 5%, and 10%
    print("Error band 1% =", errorBand(1, y[:, 1], pred_lat))
    print("Error band 5% =", errorBand(5, y[:, 1], pred_lat))
    print("Error band 10% =", errorBand(10, y[:, 1], pred_lat))

def predAccuracy(a_model, X, y, X_tv, y_tv):
    # Train the accuracy model using the training data (X_tv and y_tv[:,0])
    a_model.fit(X_tv, y_tv[:, 0])
    
    # Predict the accuracy values for the test set X
    pred_acc = a_model.predict(X)
    
    # Calculate and print the percentage of predictions within error bands of 1%, 5%, and 10%
    print("Error band 1% =", errorBand(1, y[:, 0], pred_acc))
    print("Error band 5% =", errorBand(5, y[:, 0], pred_acc))
    print("Error band 10% =", errorBand(10, y[:, 0], pred_acc))

def predMemory(a_model, X, y, X_tv, y_tv):
    # Train the memory model using the training data (X_tv and y_tv[:,2])
    a_model.fit(X_tv, y_tv[:, 2])
    
    # Predict the memory values for the test set X
    pred_mem = a_model.predict(X)
    
    # Calculate and print the percentage of predictions within error bands of 1%, 5%, and 10%
    print("Error band 1% =", errorBand(1, y[:, 2], pred_mem))
    print("Error band 5% =", errorBand(5, y[:, 2], pred_mem))
    print("Error band 10% =", errorBand(10, y[:, 2], pred_mem))



def NAS_results(models_path, X,y, X_tv, X_test, y_tv, y_test):
    # Load the top triplets from the provided models_path CSV file
    top_triplets = pd.read_csv(models_path)

    for i in range(1):
        # Get the best triplet (only considering the first triplet in this case)
        best_triplet = top_triplets.iloc[i]

        # Initialize models for accuracy, latency, and memory using the best hyperparameters
        print("Selected accuracy regressor:")
        accuracy_model = initialize_model(best_triplet['accuracy_model'], best_triplet['accuracy_hyperparams'])
        print("Selected latency regressor:")
        latency_model = initialize_model(best_triplet['latency_model'], best_triplet['latency_hyperparams'])
        print("Selected memory regressor:")
        memory_model = initialize_model(best_triplet['memory_model'], best_triplet['memory_hyperparams'])
        print("#########################################################################################################################")

        # Train models and measure the time taken for training
        accuracy_model.fit(X_tv, y_tv[:, 0])  # Train the accuracy model
        latency_model.fit(X_tv, y_tv[:, 1])   # Train the latency model
        memory_model.fit(X_tv, y_tv[:, 2])    # Train the memory model


        # Prepare the real and predicted values for evaluation
        real_lat = y[:, 1]
        real_acc = y[:, 0]
        real_mem = y[:, 2]

        pred_acc = accuracy_model.predict(X)
        pred_lat = latency_model.predict(X)
        pred_mem = memory_model.predict(X) 

        # Print predictions for latency, accuracy, and memory
        print("Latency:")
        predLatency(latency_model, X, y, X_tv, y_tv)
        print("Accuracy:")
        predAccuracy(accuracy_model, X, y, X_tv, y_tv)
        print("Memory:")
        predMemory(memory_model, X, y, X_tv, y_tv)

        print("#########################################################################################################################")

        # Evaluate model performance using various metrics (Kendall Tau, Spearman, Pearson, MSE)
        import scipy.stats as stats
        tau, p_value = stats.kendalltau(real_acc, pred_acc)
        sp, p_value = stats.spearmanr(real_acc, pred_acc)
        pr, p_value = stats.pearsonr(real_acc, pred_acc)
        mse = mean_absolute_percentage_error(real_acc, pred_acc)
        print(f"Accuracy # Tau: {tau}, Spearman: {sp}, PearsonR: {pr}")
        print(f"Accuracy MSE: {mse}")

        tau, p_value = stats.kendalltau(real_lat, pred_lat)
        sp, p_value = stats.spearmanr(real_lat, pred_lat)
        pr, p_value = stats.pearsonr(real_lat, pred_lat)
        mse = mean_absolute_percentage_error(real_lat, pred_lat)
        print(f"Latency # Tau: {tau}, Spearman: {sp}, PearsonR: {pr}")
        print(f"Latency MSE: {mse}")

        tau, p_value = stats.kendalltau(real_mem, pred_mem)
        sp, p_value = stats.spearmanr(real_mem, pred_mem)
        pr, p_value = stats.pearsonr(real_mem, pred_mem)
        mse = mean_absolute_percentage_error(real_mem, pred_mem)
        print(f"Memory # Tau: {tau}, Spearman: {sp}, PearsonR: {pr}")
        print(f"Memory MSE: {mse}")

        print("#########################################################################################################################")

        # Evaluate the non-dominated sorting algorithm and calculate ADRS
        idx_3D = non_dominated_sorting_algorithm_3d(real_acc, real_lat, real_mem)
        idx_new_3D = non_dominated_sorting_algorithm_3d(pred_acc, pred_lat, pred_mem)

        # Calculate ADRS for the predicted vs real fronts
        adrs = ADRS_3d(real_acc[idx_3D[0]], real_lat[idx_3D[0]], real_mem[idx_3D[0]],
                       real_acc[idx_new_3D[0]], real_lat[idx_new_3D[0]], real_mem[idx_new_3D[0]])
        
        # Output ADRS result and indices of the fronts
        print("adrs=", adrs)
        # print(idx_3D[0])
        # print(idx_new_3D[0])

        # Prepare data for future use or export
        data = {
            'idx': idx_new_3D[0],
            'acc': real_acc[idx_new_3D[0]],
            'lat': real_lat[idx_new_3D[0]],
            'mem': real_mem[idx_new_3D[0]]
        }
        # Create a DataFrame
        pof_pred = pd.DataFrame(data)

        # Save DataFrame to a CSV file
        pof_pred.to_csv('NAS_POF.csv', index=False)








def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Run NAS with specified parameters.')
    parser.add_argument('--dataset', type=str, required=True, default='data/gpt_l_rtx3080.csv', help='Path to the dataset CSV file')
    parser.add_argument('--adrs_tuner', type=str, required=True, default='tuner/GPT/gpt_l/rtx3080/adrs', help='Path to the ADRS tuner results directory')
    parser.add_argument('--train_ratio', type=float, default=0.1, help='Ratio of training data (between 0 and 1)')

    args = parser.parse_args()
    # Use command-line arguments
    train_ratio = args.train_ratio
    dataset_path = args.dataset
    adrs_tuner_path = args.adrs_tuner


    # Load data
    data = pd.read_csv(dataset_path)

    # Process data
    X_raw = data['vector']
    X = np.array([ast.literal_eval(x) for x in X_raw])
    memory = data['mem']
    latency = data['latency']
    accuracy = data['accuracy']

    X = np.array(X)
    y = np.vstack((accuracy, latency, memory)).T

    # Split data
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=1 - train_ratio, random_state=24)

    # Run NAS results
    NAS_results(adrs_tuner_path, X,y, X_tv, X_test, y_tv, y_test)

if __name__ == "__main__":
    main()

