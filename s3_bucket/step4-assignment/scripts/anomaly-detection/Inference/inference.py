import sys, argparse

sys.path.append('/tmp/scripts/Anomaly Detection/')  #adjust based on your system's directory
import torch, time
import numpy as np
import Plot_Redshift as plt_rdshft
from torch.utils.data import DataLoader
from blocks.model_vit_inception import ViT_Astro
import boto3
from botocore.exceptions import ClientError
import logging
import os
import fmi
from fmilib.fmi_operations import fmi_communicator
import json


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def environ_or_required(key, default = None, required=True):

    if default is None:
        return (
            {'default': os.environ.get(key)} if os.environ.get(key)
            else {'required': required}

        )
    else:
        return (
            {'default': os.environ.get(key)} if os.environ.get(key)
            else {'default': default}

        )
#Load Data
def load_data(data_path, device):
    return torch.load(data_path, map_location=device)


#Load Model
def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    return model.module.eval()


#Use DataLoader for iterating over batches
def data_loader(data, batch_size):
    return DataLoader(data, batch_size=batch_size, drop_last=True)  #Drop samples out of the batch size


#Iterate over data for predicting the redshift and invoke the evaluation modules
def inference(model, dataloader, real_redshift, plot_to_save_path, device, batch_size, rank, world_size):
    redshift_analysis = []
    total_time = 0.0  # Initialize total time for execution
    num_batches = 0  # Initialize number of batches
    total_data_bits = 0  # Initialize total data bits processed

    comm = fmi_communicator(world_size, rank)

    comm.hint(fmi.hints.fast)

    comm.barrier()

    for i, data in enumerate(dataloader):
        image = data[0].to(device)  #Image is permuted, cropped and moved to cuda
        magnitude = data[1].to(device)  #magnitude of of channels
        redshift = data[2].to(device)  #target, which is the redshift

        start_time = time.time()  #Put the start time of the execution

        predict_redshift = model(
            [image, magnitude])  #model predicts the redshft using two inputs (image and magnitudes)

        end_time = time.time()  #Put the end time of the execution

        total_time += (end_time - start_time)  # Accumulate time for this batch

        redshift_analysis.append(predict_redshift.view(-1, 1))

        # Calculate the size of the image and magnitude data in bits
        image_bits = image.element_size() * image.nelement() * 8  # Convert bytes to bits
        magnitude_bits = magnitude.element_size() * magnitude.nelement() * 8  # Convert bytes to bits
        total_data_bits += image_bits + magnitude_bits  # Add data bits for this batch

        num_batches += 1

    num_samples = num_batches * batch_size

    redshift_analysis = torch.cat(redshift_analysis, dim=0)

    redshift_analysis = redshift_analysis.cpu().detach().numpy().reshape(num_batches * batch_size, )

    real_redshift = real_redshift[:num_batches * batch_size]

    execution_info = {
        'total_time': total_time,
        'execution_time': total_time / num_batches,  # Average execution time per batch
        'num_batches': num_batches,  # Number of batches
        'batch_size': batch_size,  # Batch size
        'device': device,  # Selected device
        'throughput_bps': total_data_bits / total_time,  # Throughput in bits per second
        'sample_persec': num_samples / total_time
    }

    plt_rdshft.err_calculate(redshift_analysis, real_redshift, execution_info,
                             plot_to_save_path)  #invoke for calculating statistical prediction evaluation metrics


    reduce_res = comm.reduce(total_time, 0, fmi.func(fmi.op.sum), fmi.types(fmi.datatypes.double))

    if rank == 0:
        reduce_info = {
            'total_time': reduce_res
        }

        with open(f'{prj_dir}Plots/ResultsReduce.json', 'w', encoding='utf-8') as redInfo:
            json.dump(reduce_info, redInfo, ensure_ascii=False, indent=4)

        upload_file(file_name=f'{prj_dir}Plots/ResultsReduce.json',
                    bucket='cosmicai2',
                    object_name=f'results/{rank}/ResultsReduce.json')


    upload_file(file_name=f'{prj_dir}Plots/Results.json',
                bucket='cosmicai2',
                object_name=f'results/{rank}/Results.json')


#This is the engine module for invoking and calling various modules
def engine(args):
    data = load_data(args.data_path, args.device)
    dataloader = data_loader(data, args.batch_size)
    model = load_model(args.model_path, args.device)
    inference(model, dataloader, data[:][2].to('cpu'), args.plot_path, device=args.device, batch_size=args.batch_size,
              rank=args.rank, world_size=args.world_size)


# Pathes and other inference hyperparameters can be adjusted below
if __name__ == '__main__':
    prj_dir = '/tmp/scripts/Anomaly Detection/'  #adjust based on your system's directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_path', type=str, default=f'{prj_dir}Inference/resized_inference.pt')
    parser.add_argument(
        '--model_path',
        type=str,
        default=f'{prj_dir}Fine_Tune_Model/Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt',
    )
    parser.add_argument('--device', type=str, default='cpu')  # To run on GPU, put cuda, and on CPU put cpu

    parser.add_argument('--plot_path', type=str, default=f'{prj_dir}Plots/')

    parser.add_argument('--rank',  type=int, **environ_or_required('RANK', required=False))
    parser.add_argument('--world_size', type=int, **environ_or_required('WORLD_SIZE', required=False))
    args = parser.parse_args()

    engine(args)
