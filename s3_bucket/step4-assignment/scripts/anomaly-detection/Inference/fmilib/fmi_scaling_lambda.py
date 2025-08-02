import time
import argparse


import pandas as pd
from numpy.random import default_rng

from cloudmesh.common.StopWatch import StopWatch

import boto3
from botocore.exceptions import ClientError
import os

import logging
import fmi

def environ_or_required(key):
    return (
        {'default': os.environ.get(key)} if os.environ.get(key)
        else {'required': True}
    )
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


def fmi_join(data=None):
    global ucc_config
    StopWatch.start(f"join_total_{data['host']}_{data['rows']}_{data['it']}")

    world_size = int(data["world_size"])
    rank = int(data["rank"])

    communicator = fmi.Communicator(rank, world_size, "fmi.json", "fmi_pair", 512)

    if communicator is None:
        print("unable to create FMI Communicator")
        return

    communicator.hint(fmi.hints.fast)

    print("retrieved fmi communicator")

    u = data['unique']

    if data['scaling'] == 'w':  # weak
        num_rows = data['rows']
        max_val = num_rows * world_size
    else:  # 's' strong
        max_val = data['rows']
        num_rows = int(data['rows'] / world_size)

    rng = default_rng(seed=rank)
    data1 = rng.integers(0, int(max_val * u), size=(num_rows, 2))
    data2 = rng.integers(0, int(max_val * u), size=(num_rows, 2))

    df1 = pd.DataFrame(data1).add_prefix("col")
    df2 = pd.DataFrame(data2).add_prefix("col")

    timing = {'scaling': [], 'world': [], 'rows': [], 'max_value': [], 'rank': [], 'avg_t': [], 'tot_l': []}

    print("iterating over range")
    for i in range(data['it']):
        communicator.barrier()
        StopWatch.start(f"join_{i}_{data['host']}_{data['rows']}_{data['it']}")
        t1 = time.time()
        df3 = pd.concat([df1, df2], axis=1)
        result_array = df3.to_numpy().flatten().tolist()

        communicator.barrier()
        t2 = time.time()
        t = (t2 - t1) * 1000
        # sum_t = comm.reduce(t)
        sum_t = communicator.allreduce(t, fmi.func(fmi.op.sum), fmi.types(fmi.datatypes.double))
        # tot_l = comm.reduce(len(df3))
        tot_l = communicator.allreduce(result_array, fmi.func(fmi.op.sum), fmi.types(fmi.datatypes.int_list, len(result_array)))

        if rank == 0:
            avg_t = sum_t / world_size
            print("### ", data['scaling'], world_size, num_rows, max_val, i, avg_t, len(tot_l))
            timing['scaling'].append(data['scaling'])
            timing['world'].append(world_size)
            timing['rows'].append(num_rows)
            timing['max_value'].append(max_val)
            timing['rank'].append(i)
            timing['avg_t'].append(avg_t)
            timing['tot_l'].append(len(tot_l))
            #print("### ", data['scaling'], env.world_size, num_rows, max_val, i, avg_t, tot_l, file=open(data['output_summary_filename'], 'a'))
            StopWatch.stop(f"join_{i}_{data['host']}_{data['rows']}_{data['it']}")

    StopWatch.stop(f"join_total_{data['host']}_{data['rows']}_{data['it']}")

    if rank == 0:
        StopWatch.benchmark(tag=str(data), filename=data['output_scaling_filename'])
        upload_file(file_name=data['output_scaling_filename'], bucket=data['s3_bucket'],
                    object_name=data['s3_stopwatch_object_name'])

        if os.path.exists(data['output_summary_filename']):
            pd.DataFrame(timing).to_csv(data['output_summary_filename'], mode='a', index=False, header=False)
        else:
            pd.DataFrame(timing).to_csv(data['output_summary_filename'], mode='w', index=False, header=True)

        upload_file(file_name=data['output_summary_filename'], bucket=data['s3_bucket'],
                    object_name=data['s3_summary_object_name'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fmi scaling")

    parser.add_argument('-r', dest='rank', type=int, **environ_or_required('RANK'))

    parser.add_argument('-n', dest='rows', type=int, **environ_or_required('ROWS'))

    parser.add_argument('-i', dest='it', type=int, **environ_or_required('PARTITIONS')) #10

    parser.add_argument('-u', dest='unique', type=float, **environ_or_required('UNIQUENESS'), help="unique factor") #0.9

    parser.add_argument('-s', dest='scaling', type=str, **environ_or_required('SCALING'), choices=['s', 'w'],
                        help="s=strong w=weak") #w

    parser.add_argument('-o', dest='operation', type=str, **environ_or_required('FMI_OPERATION'), choices=['join', 'sort', 'slice'],
                        help="join")  # w

    parser.add_argument('-w', dest='world_size', type=int, help="world size", **environ_or_required('WORLD_SIZE'))


    parser.add_argument('-f1', dest='output_scaling_filename', type=str, help="Output filename for scaling results",
                        **environ_or_required('OUTPUT_SCALING_FILENAME'))

    parser.add_argument('-f2', dest='output_summary_filename', type=str, help="Output filename for scaling summary results",
                        **environ_or_required('OUTPUT_SUMMARY_FILENAME'))

    parser.add_argument('-b', dest='s3_bucket', type=str, help="S3 Bucket Name", **environ_or_required('S3_BUCKET'))

    parser.add_argument('-o1', dest='s3_stopwatch_object_name', type=str, help="S3 Object Name", **environ_or_required('S3_STOPWATCH_OBJECT_NAME'))

    parser.add_argument('-o2', dest='s3_summary_object_name', type=str, help="S3 Object Name",
                        **environ_or_required('S3_SUMMARY_OBJECT_NAME'))

    args = vars(parser.parse_args())

    args['host'] = "aws"

    if args['operation'] == 'join':
        print("executing cylon join operation")
        fmi_join(args)


