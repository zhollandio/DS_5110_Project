import sys, argparse

sys.path.append('/tmp/Anomaly Detection/')  # adjust based on your system's directory
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import boto3
from botocore.exceptions import ClientError
import logging
import os, gc
from torch.profiler import profile, ProfilerActivity
import json, platform, boto3, io
import numpy as np
import fmi
import time
from fmilib.fmi_operations import fmi_communicator
from cloudmesh.common.StopWatch import StopWatch
from cloudmesh.common.dotdict import dotdict
from cloudmesh.common.Shell import Shell
from cloudmesh.common.util import writefile
import pandas as pd


s3_client = boto3.client('s3')


def get_cpu_info():
    # CPU Information
    print("CPU Information:")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Machine type: {platform.machine()}")
    print(f"System: {platform.system()}")
    print(f"Platform: {platform.platform()}")

    return {
        'processor': platform.processor(),
        'architecture': platform.architecture(),
        'machine': platform.machine(),
        'system': platform.system(),
        'platform': platform.platform()
    }


# RAM Information
def get_ram_info():
    if hasattr(os, 'sysconf'):
        if 'SC_PAGE_SIZE' in os.sysconf_names and 'SC_PHYS_PAGES' in os.sysconf_names:
            page_size = os.sysconf('SC_PAGE_SIZE')  # in bytes
            total_pages = os.sysconf('SC_PHYS_PAGES')
            total_ram = page_size * total_pages  # in bytes
            total_ram_gb = total_ram / (1024 ** 3)  # convert to GB
            print(f"Total memory (GB): {total_ram_gb:.2f}")
            return total_ram_gb
    return None


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

    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def environ_or_required(key, default=None, required=True):
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


# Load Data
def load_data(data_path, device):
    return torch.load(data_path, map_location=device)


# Load Model
def load_model(model_path, device):
    model = torch.load(model_path, map_location=device, weights_only=False)
    return model.module.eval()


# Use DataLoader for iterating over batches
def data_loader(data, batch_size):
    return DataLoader(data, batch_size=batch_size)


# Iterate over data for predicting the redshift and invoke the evaluation modules
def inference(
        model, dataloader, device, batch_size,
        rank, result_path, data_path, args
):
    StopWatch.start(f"inference_total_lambda_{args.batch_size}_{args.world_size}")
    total_time = 0.0  # Initialize total time for execution
    num_batches = 0  # Initialize number of batches
    total_data_bits = 0  # Initialize total data bits processed
    logging.info(f'Rank: {rank}. Start Inference')
    num_samples = 0
    t1 = time.time()

    with profile(
            activities=[ProfilerActivity.CPU],
            profile_memory=True
    ) as prof:
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                image = data[0].to(device)  # Image is permuted, cropped and moved to cuda
                magnitude = data[1].to(device)  # magnitude of of channels

                _ = model([image, magnitude])  # Model inference

                # Calculate the size of the image and magnitude data in bits
                image_bits = image.element_size() * image.nelement() * 8  # Convert bytes to bits
                magnitude_bits = magnitude.element_size() * magnitude.nelement() * 8  # Convert bytes to bits
                total_data_bits += image_bits + magnitude_bits  # Add data bits for this batch

                num_batches += 1
                num_samples += len(image)
                gc.collect()

    # num_samples = num_batches * batch_size

    avg = prof.key_averages().total_average()
    # Extract total time and memory usage for CPU and GPU
    total_cpu_memory = avg.cpu_memory_usage / 1e6  # Convert bytes to MB
    # total_gpu_memory = prof.key_averages().total_average().cuda_memory_usage / 1e6  # Convert bytes to MB

    # Extract total CPU and GPU time
    total_time = avg.cpu_time_total / 1e6  # Convert from microseconds to seconds
    # total_gpu_time = prof.key_averages().total_average().cuda_time_total / 1e6  # Convert from microseconds to milliseconds
    avg_time_batch = total_time / (num_samples / batch_size)

    logging.info(f'Rank: {rank}. End Inference. Total time: {total_time}. Avg time per batch: {avg_time_batch}.')

    execution_info = {
        'total_cpu_time (seconds)': total_time,
        'total_cpu_memory (MB)': total_cpu_memory,
        'execution_time (seconds/batch)': avg_time_batch,  # Average execution time per batch
        'num_batches': num_batches,  # Number of batches
        'batch_size': batch_size,  # Batch size
        'device': device,  # Selected device
        'throughput_bps': total_data_bits / total_time,
        # Throughput in bits per second (using total_time for all batches)
        'sample_persec': num_samples / total_time,  # Number of samples processed per second
        'result_path': result_path,
        'data_path': data_path
    }

    comm = fmi_communicator(args.world_size, rank)

    comm.hint(fmi.hints.fast)

    comm.barrier()

    t2 = time.time()

    t = (t2 - t1) * 1000

    sum_t = comm.reduce(t, 0, fmi.func(fmi.op.sum), fmi.types(fmi.datatypes.double))

    if rank == 0:
        avg_t = sum_t / args.world_size
        timing = {'avg_t': [], 'world': [],
                  'total_cpu_time': [], 'total_cpu_memory': [], 'execution_time': [],
                  'num_batches': [], 'batch_size': [], 'device': [],
                  'throughput_bps': [], 'sample_persec': [],
                  'result_path': [], 'data_path': []
                  }

        timing['avg_t'].append(avg_t)
        timing['world'].append(args.world_size)
        timing['total_cpu_time'].append(total_time)
        timing['total_cpu_memory'].append(total_cpu_memory)
        timing['execution_time'].append(avg_time_batch)
        timing['num_batches'].append(num_batches)
        timing['batch_size'].append(batch_size)
        timing['device'].append(device)
        timing['throughput_bps'].append(total_data_bits / total_time)
        timing['sample_persec'].append(num_samples / total_time)
        timing['result_path'].append(result_path)
        timing['data_path'].append(data_path)
        StopWatch.stop(f"inference_total_lambda_{args.batch_size}_{args.world_size}")

        StopWatch.benchmark(tag=str(args), filename=f'{prj_dir}Plots/summary.txt')

        upload_file(file_name=f'{prj_dir}Plots/summary.txt',
                    bucket=args.data_bucket,
                    object_name=f'{result_path}/summary.txt')



        output_file = f'{prj_dir}Plots/ResultsReduce.csv'
        if os.path.exists(output_file):
            os.remove(output_file)

        pd.DataFrame(timing).to_csv(output_file, mode='w', index=False, header=True)

        upload_file(file_name=output_file,
                    bucket=args.data_bucket,
                    object_name=f'{result_path}/ResultsReduce.csv')

    s3_client.put_object(
        Bucket=args.data_bucket,
        Key=f'{result_path}/{rank}.json',
        Body=json.dumps(execution_info),
        ContentType="application/json"
    )



def concatenate_data(data_list):
    images = []
    magnitudes = []
    redshifts = []

    for chunk in data_list:
        # Split image, magnitude, and redshift from each chunk
        images.append(chunk[:][0])
        magnitudes.append(chunk[:][1])
        redshifts.append(chunk[:][2])

    # Concatenate image, magnitude and redshift in separate tensors
    images = torch.cat(images)
    magnitudes = torch.cat(magnitudes)
    redshifts = torch.cat(redshifts)

    # Store them as a dataset in save_cat_path
    return TensorDataset(images, magnitudes, redshifts)


def load_data(data_path, bucket):
    if type(data_path) == list:
        data_list = [load_data(data_path=path, bucket=bucket) for path in data_path]
        return concatenate_data(data_list)

    # Create a BytesIO buffer to hold the downloaded file
    buffer = io.BytesIO()
    # Download the .pt file from S3 to the buffer
    s3_client.download_fileobj(Bucket=bucket, Key=data_path, Fileobj=buffer)
    # Move to the beginning of the buffer to read
    buffer.seek(0)
    data = torch.load(buffer, weights_only=False)  # load_data(args.data_path, args.device)
    return data


def partition_data(data, rank, world_size):
    image, magnitude, redshift = data[:][0], data[:][1], data[:][2]
    total = len(data)
    rank, world_size = args.rank, args.world_size

    start = rank * total // world_size
    if rank == world_size - 1:
        end = total
    else:
        end = (rank + 1) * total // world_size  # int(splits[rank])

    logging.info(f'Rank: {rank}, Start: {start}, End: {end}')
    image = image[start:end]
    magnitude = magnitude[start:end]
    redshift = redshift[start:end]
    data = TensorDataset(image, magnitude, redshift)

    return data


# This is the engine module for invoking and calling various modules
def engine(args):
    data = load_data(
        data_path=args.data_path, bucket=args.data_bucket
    )

    dataloader = DataLoader(
        data, batch_size=args.batch_size  # , drop_last=True
    )  # data_loader(data, args.batch_size)
    model = load_model(args.model_path, args.device)

    inference(
        model, dataloader,
        device=args.device, batch_size=args.batch_size,
        rank=args.rank, result_path=args.result_path,
        data_path=args.data_path, args=args
    )


# Pathes and other inference hyperparameters can be adjusted below
if __name__ == '__main__':
    prj_dir = '/tmp/Anomaly Detection/'  # adjust based on your system's directory
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path', type=str,
        default=f'{prj_dir}Fine_Tune_Model/Mixed_Inception_z_VITAE_Base_Img_Full_New_Full.pt',
    )
    parser.add_argument('--device', type=str, default='cpu')  # To run on GPU, put cuda, and on CPU put cpu

    parser.add_argument('--plot_path', type=str, default=f'{prj_dir}Plots/')
    # parser.add_argument('--result_path', type=str, **environ_or_required('RESULT_PATH', required=False))

    parser.add_argument('--rank', type=int, **environ_or_required('RANK', required=False))
    parser.add_argument('--world_size', type=int, **environ_or_required('WORLD_SIZE', required=False))
    parser.add_argument('--batch_size', type=int, **environ_or_required('BATCH_SIZE', required=False))
    parser.add_argument('--data_bucket', type=str, **environ_or_required('DATA_BUCKET', required=False))
    parser.add_argument('--result_path', type=str, **environ_or_required('RESULT_PATH', required=False))
    parser.add_argument('--data_path', type=str, **environ_or_required('DATA_PATH', required=False))

    args = parser.parse_args()

    if args.data_path is None:
        print(f'Rank: {args.rank}. Data path is not specified. Exiting.')
        sys.exit(0)

    engine(args)
