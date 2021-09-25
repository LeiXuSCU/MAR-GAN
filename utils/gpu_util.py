import os

import pynvml


def get_device_id():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(device_count):
        available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus + str(g) + ','
    gpus = gpus[:-1]
    return gpus


def set_gpu(gpu_input):
    if gpu_input == 'all':
        gpus = get_device_id()
    else:
        gpus = gpu_input
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    return len(gpus.split(','))
