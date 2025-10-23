from torch import backends
from argparse import ArgumentParser
try:
    from utilities.util_ymlio import read_yaml
except ModuleNotFoundError:
    from util_ymlio import read_yaml

parser = ArgumentParser()
args = parser.parse_args()

def params_to_args(params):
    for key, value in params.items():
        if isinstance(value, dict):
            params_to_args(value)
        else:
            setattr(args, key, value)
    if hasattr(args, 'train_img_start') and hasattr(args, 'train_img_end'):
        args.images_to_train = range(args.train_img_start-1, args.train_img_end)
    if hasattr(args, 'test_img_start') and hasattr(args, 'test_img_end'):
        args.images_to_test = range(args.test_img_start-1, args.test_img_end)
    if hasattr(args, 'predict_img_start') and hasattr(args, 'predict_img_end'):
        args.images_to_predict = range(args.predict_img_start-1, args.predict_img_end)
    return args


def setupParams(ymlfilename,default=False):
    # Read in parameters from yaml file
    if default:
        params = read_yaml('L:/HRFLFMnet/dl_pytorch_100x_v2/paraymls/DefaultParams.yml')
    else:
        params = read_yaml(ymlfilename)
    # Convert parameters to arguments
    args = params_to_args(params)
    return args

def setupDevices(args,useBenchmark=True):
    from torch import set_num_threads, manual_seed, get_num_threads
    n_threads = 0
    if len(args.main_gpu)>0:
        device = "cuda:" + str(args.main_gpu[0])
        device_repro = "cuda:" + str(args.main_gpu[0]+1)
    else:
        device = "cpu"
        device_repro = "cuda:0"

    if len(args.gpu_repro)==0:
        device_repro = "cpu"
    else:
        device_repro = "cuda:" + str(args.gpu_repro[0])
    print("# Using device: ", device, " and ", device_repro)

    if n_threads!=0:
        set_num_threads(n_threads)
    manual_seed(3132219)
    print("# Using ", get_num_threads(), " threads")
    if useBenchmark:
        backends.cudnn.enabled = True
        backends.cudnn.benchmark = True
        print("# Using cudnn benchmark")
    return device, device_repro,n_threads

def setupTrainingID(args, XMLFILENAME, prefix=''):
    from datetime import datetime
    # Set up training ID
    training_id = "FLFMnet_train__" + datetime.now().strftime('%Y%m%d_%H%M%S') + "__" + \
                                  XMLFILENAME.split('/')[-1].split('.')[0] + "_" + prefix
    save_folder = args.train_ckpt_to + '/' + training_id
    print("# Training ID: ", training_id)
    print("# Save folder: ", save_folder)
    return training_id, save_folder



if __name__ == '__main__':
    from os import system, name
    from pprint import pprint

    system('cls' if name == 'nt' else 'clear')
    args = setupParams('L:/HRFLFMnet/dl_pytorch_100x_v2/paraymls/DefaultParams.yml')
    pprint(args)
    print(args.train_ckpt_from,type(args.train_ckpt_from))

    


