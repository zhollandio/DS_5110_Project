import math, json
import numpy as np

def err_calculate(prediction, z, execution_info, save_path):
    
    def r2_score(y_true, y_pred):
        # Calculate the residual sum of squares
        ss_res = np.sum((y_true - y_pred) ** 2)

        # Calculate the total sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        # Calculate R^2 score
        r2 = 1 - (ss_res / ss_tot)

        return r2
 
    prediction = np.array(prediction)
    z = np.array(z)
    
    #MAE
    mae = np.sum(abs(prediction - z)) / z.shape[0]
    
    #MSE
    mse = np.sum((prediction - z)**2) / z.shape[0]    
    
    #Delta
    deltaz = (prediction - z) / (1 + z)
    
    #Bias
    bias = np.sum(deltaz) / z.shape[0]
    
    #Precision
    nmad = 1.48 * np.median(abs(deltaz - np.median(deltaz)))
    
    #R^2 score
    r2 = r2_score(z, prediction)
    
    #All errors are stored in a json and saved in  save_path directory
    errs = {
    'total execution time': execution_info['total_time'],
    'throughput': execution_info['throughput_bps'],
    'samples per second': execution_info['sample_persec'],
    'average execution time (milliseconds) per batch': execution_info['execution_time'] * 1000,
    'batch size': execution_info['batch_size'],
    'number of batches': execution_info['num_batches'],
    'device': execution_info['device'],
    'MAE': mae,
    'MSE': mse,
    'Bias': bias,
    'Precision': nmad,
    'R2': r2
    }
    
    with open(save_path + 'Results.json', 'w') as file:
        json.dump(errs, file, indent=5)
