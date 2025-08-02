import sys
import argparse
import os
import torch
import time
import boto3
import json
import logging
import numpy as np
from torch.utils.data import DataLoader
from botocore.exceptions import ClientError

# Add temp directory to sys.path for importing Plot_Redshift
sys.path.insert(0, '/tmp')

# Attempt to import Plot_Redshift if it was downloaded
try:
    import Plot_Redshift as plt_rdshft
except ImportError:
    plt_rdshft = None

s3 = boto3.client('s3')
prj_dir = '/tmp/scripts/Anomaly Detection/'

def upload_file(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = os.path.basename(file_name)
    try:
        s3.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def environ_or_required(key, default=None, required=True):
    if default is None:
        return {'default': os.environ.get(key)} if os.environ.get(key) else {'required': required}
    else:
        return {'default': os.environ.get(key)} if os.environ.get(key) else {'default': default}

def load_data(data_path, device):
    return torch.load(data_path, map_location=device)

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    return model.module.eval()

def data_loader(data, batch_size):
    return DataLoader(data, batch_size=batch_size, drop_last=True)

def inference(model, dataloader, real_redshift, plot_to_save_path, device, batch_size):
    redshift_analysis = []
    total_time = 0.0
    num_batches = 0
    total_data_bits = 0

    for data in dataloader:
        image = data[0].to(device)
        magnitude = data[1].to(device)
        redshift = data[2].to(device)

        start_time = time.time()
        predict_redshift = model([image, magnitude])
        end_time = time.time()

        total_time += (end_time - start_time)
        redshift_analysis.append(predict_redshift.view(-1, 1))

        image_bits = image.element_size() * image.nelement() * 8
        magnitude_bits = magnitude.element_size() * magnitude.nelement() * 8
        total_data_bits += image_bits + magnitude_bits

        num_batches += 1

    num_samples = num_batches * batch_size
    redshift_analysis = torch.cat(redshift_analysis, dim=0).cpu().numpy().reshape(num_samples)
    real_redshift = real_redshift[:num_samples]

    execution_info = {
        'total_time': total_time,
        'execution_time': total_time / num_batches,
        'num_batches': num_batches,
        'batch_size': batch_size,
        'device': device,
        'throughput_bps': total_data_bits / total_time,
        'sample_persec': num_samples / total_time
    }

    if plt_rdshft:
        plt_rdshft.err_calculate(redshift_analysis, real_redshift, execution_info, plot_to_save_path)

    with open(f'{prj_dir}Plots/Results.json', 'w') as f:
        json.dump(execution_info, f, indent=2)

    upload_file(f'{prj_dir}Plots/Results.json', 'team2-cosmical-7078ea12', 'results/Results.json')

def engine(args):
    data = load_data(args.data_path, args.device)
    dataloader = data_loader(data, args.batch_size)
    model = load_model(args.model_path, args.device)
    real_redshift = data[:][2].to('cpu')
    inference(model, dataloader, real_redshift, args.plot_path, args.device, args.batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_path', type=str, default=f'{prj_dir}Inference/resized_inference.pt')
    parser.add_argument('--model_path', type=str, default=f'{prj_dir}Fine_Tune_Model/Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--plot_path', type=str, default=f'{prj_dir}Plots/')
    args = parser.parse_args()

    engine(args)
