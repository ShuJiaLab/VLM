# Overall
Welcome to the short tutorial to VLM, where the implementation of the VLM can be found below. In general, the user would want to setup the V-Net and L-Net first, then use V-Net to reconstruct 3D low-resolution volume from 2D light-field recordings, lastly use L-net to reconstruct super-resolutiuon 3D volume from the previous low-res 3D voluem.

Last update: Sep, 2025

---
# 1. V-Net
The exact architecture of the V-Net can be found in the Supplementary Note 1 accompanied with the submitted manuscript.

## 1.1 Environment
The code is tested with `Python=3.11`, `PyTorch=2.0`, and `CUDA=12.1`. We recommend you to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) to make sure that all dependencies are in place. To create an conda environment, `cd` do your LNet folder and use the `.yml` file:
```bash
# create the conda environment
cd VNet
conda env create -f environment.yml
conda activate VNet
```
Depends on your system and network speed, the installation should taks about several miniutes.

## 1.2 Checkpoints and Setup
Please download the checkpoints file from our [Zenodo](https://doi.org/10.5281/zenodo.17064480) repository. Several files need to rearrange into your `VNet` root folder:
- Put the checkpoint file `model_microtubule` under the V-Net root folder `VNet/checkpoints/`
- Put the parameter setup file `STORM_microtubule.yml` under the V-Net root folder `VNet/paraymls/`
- Put the hybrid PSF file `PSF_microtubule_red_10um.mat` under the V-Net root folder `VNet/psf`

In the submitted Supplementary Software, we provided 20 frames of microtubulue dataset for testing under `VNet/date2predict/`. Additional 2000 frames of microtubulue dataset can be found in the Zenodo repository as `microtubule_2000fr.zip`.
 

## 1.3 Evaluation
To predict 3D volume output from 2D light-field recordings using V-Net, you will need to 
- Modify the `VNet/paraymls/STORM_microtubule.yml` to setup
- Run the python file `VNet/FLFMpredict_main.py` to run

### 1.3.1 Modify YML file
A typical `.yml` file should look like this
```bash
# basic setup
magnification: 100
psf_file: ./psf/PSF_microtubule_red_10um.mat
lenslet_file: ./lenslet_centers_python.txt
# train folder
train_folder_in: ./train/
train_folder_gt: ./train/
train_ckpt_from: ./checkpoints/
train_ckpt_to: ./checkpoints/
# test folder
test_folder_in: ./test/
test_folder_gt: ./test/
# predict flder
predict_folder_in: ./data2predict/
predict_ckpt_from: ./checkpoints/model_microtubule
# settings
files_to_store:
- FLFMtrain_main.py
- lenslet_centers_python.txt
- datapreproc/FLFMDataset.py
- utilities/util_camnoise.py
- utilities/util_centercrop.py
- utilities/util_checkdataset.py
- utilities/util_getMLAcenters.py
- utilities/util_imgRegister.py
- utilities/util_imVisualizeSave.py
- utilities/util_lossfunctions.py
- utilities/util_notice.py
- utilities/util_reprojection.py
- utilities/util_setupLosses.py
- utilities/util_setupPSFOTF.py
- utilities/util_setupSystem.py
- utilities/util_setupWriter.py
- utilities/util_str2url.py
- utilities/util_ymlio.py
- networks/resnet.py
- networks/unet.py
- networks/FLFMnet.py
data_rescale: 0.5
main_gpu:
- 0
gpu_repro:
- 0
# train parameters
train_params:
  train_img_start: 1
  train_img_end: 600
  test_img_start: 1
  test_img_end: 24
  img_depth: 64
  batch_size: 3
  max_epochs: 801
  val_split: 0.1
  val_every: 10
  shuffle: True
  learning_rate: 1e-4
  loss_type: l2
  add_noise: True
  signal_min: 100
  signal_max: 300
  norm_type: 1
  train_baseline_in: 10
  train_baseline_gt: 0
  unet_depth: 7
  unet_wf: 4
  unet_dropout: 0.3
# predict parameters
predict_params:
  predict_img_start: 1
  predict_img_end: 10
  predict_baseline_in: 10
  writeVolsToStack: True
  output_folder: ./output/
```
In this file, there are several paramters need to match so that you can proceed.
In the `# basic setup` section
- The PSF file needs to match, i.e. `psf_file: ./psf/PSF_microtubule_red_10um.mat`
- The microlens array file needs to match, i.e. `lenslet_file: ./lenslet_centers_python.txt`

In the `# predict flder` section
- Make sure the light-field recordings are stored in the correct location `predict_folder_in: ./data2predict/`
- Make sure the checkpoint are located in the correction location `predict_ckpt_from: ./checkpoints/model_microtubule`

In the `# predict parameters` section
- `predict_img_start: 1` and `predict_img_end: 10` tells you which file in the `./data2predict/` will be predict.
  To predict the full 20 frames of the example dataset, the `predict_img_end:` need to change to 20.
  Similarly, if predicting 2000 frames, the end index also need to match
- `output_folder: ./output/` tells you where the output 3D volumes will be stored. you can change the folder path as you wish.

### 1.3.2 Main Prediction File
The main file `FLFMpredict_main.py` evaluate the dataset and output 3D volumes. 
In line 25, `YMLFILENAME = './paraymls/STORM_microtubule.yml' # set up parameter file` you need to check if this `.yml` file match with the previously modified `.yml`. Once the file is saved, you can start prediction by running `FLFMpredict_main.py` file.


---
# 2. L-Net
The exact architecture of the L-Net can be found in the Supplementary Note 2 accomapnied with the submitted manuscript.

## 2.1 Environment
The code is tested with `Python=3.11`, `PyTorch=2.1`, and `CUDA=12.1`. We recommend you to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) to make sure that all dependencies are in place. To create an conda environment, `cd` do your LNet folder and use the `.yml` file:
```bash
# create the conda environment
cd LNet
conda env create -f environment.yml
conda activate LNet
```
Depends on your system and network speed, the installation should taks about several miniutes.

## 2.2 Checkpoints
Please download the checkpoints file from our Zenodo repository [Zenodo](https://doi.org/10.5281/zenodo.17064480). Put the checkpoints under the LNet folder `LNet/ckpt/` so that you have `LNet/ckpt/e08/340.ckpt` and `LNet/ckpt/e10/450.ckpt`. Note that `e08` and `e10` match config in `src/cfg/e.py` so that you can check the training configuration for each checkpoint.

## 2.3 Evaluation
You can check the parameters that must be specified for evaluation mode by:
```bash
python main.py evalu --help
```
usage:
```bash
python main.py evalu [-h] \
    [-s {4,8}] [-r RNG_SUB_USER [RNG_SUB_USER ...]] \
    -L FRAMES_LOAD_FOLD [-S DATA_SAVE_FOLD] [-C CKPT_LOAD_PATH] \
    [-T TEMP_SAVE_FOLD] [-stride STRIDE] [-window WINDOW] [-m {DCC,MCC,RCC}] \
    [-b BATCH_SIZE] [-w num_workers]
```
options:
-   `-s {4,8}`: Scale up factor, 4 or 8. Default: 4.
-   `-r RNG_SUB_USER`: Range of the sub-region of the frames to predict. Due to 
    limited memory, we cut whole frames into patches, i.e., sub-regions and 
    predict them separately. Please type six int separated by space as the 
    subframe start (inclusive) and end (exclusive) index for each dimension, 
    i.e., `-r 0 1 8 12 9 13`. If you not sure about the number of subframe for 
    each dimension you can select, do not specify this parameter; the code will 
    print the range you can select and ask you to type the range. Default: None.
-   `-L FRAMES_LOAD_FOLD`: Path to the frames load folder. Note that the code 
    will predict all the frames under this folder. Thus, if you want to predict 
    portion of the frames, please copy them to a new folder and specify this 
    parameter with that new folder.
-   `-S DATA_SAVE_FOLD`: Path to the data save folder. No need to specify when 
    stride or window is set as non-zero. Default: None.
-   `-C CKPT_LOAD_PATH`: Path to the checkpoint load file. Default: 
    `ckpt/e08/340.ckpt` or `ckpt/e10/450.ckpt` when scale up factor is 4 or 8.
-   `-T TEMP_SAVE_FOLD`: Path to the temporary save folder for drifting 
    analysis. Must be specified when drift correction will be performed. 
    Recomend to specify different path for different dataset.
    Default: `os.path.dirname(FRAMES_LOAD_FOLD)/temp/`.
-   `-stride STRIDE`: Step size of the drift corrector, unit frames. Window size
    must larger or equal to stride and divisible by stride. Should set with 
    window at the same time. Default: 0.
-   `-window WINDOW`: Number of frames in each window, unit frames. Window size 
    must larger or equal to stride and divisible by stride. Should set with 
    stride at the same time. Default: 0.
-   `-m {DCC,MCC,RCC}`: Drift correction method, DCC, MCC, or RCC. Must be set 
    if you want to evaluate with drift correction. DCC run very fast where MCC 
    and RCC is more accurate. We suggest to use DCC to test the window size 
    first and then use MCC or RCC to calculate the final drift. Default: None.
-   `-b BATCH_SIZE`: Batch size. Set this value according to your GPU memory. 
    Note that the product of rng_sub_user must divisible by batch_size. Default:
    1.
-   `-w num_workers`: Number of workers for dataloader. Set this value according
    to your CPU. Default: 1.


### 2.3.1 Pre-processing
Before we start any evaluation, we want to make sure that the input has isotropic pixel size of 130 nm/px. The V-Net direct output is 130 nm/px in the lateral direction and 65 nm/px in the axial direction. Therefore, we want to resize the output using provided toos from the first cell `frame preprocess` in `util.ipynb`, where 
```bash
frames_load_fold = ""
frames_save_fold = ""
```
Need to be filled with your folder where your data is stored. Usually we stored the data in the `LNet/data` for eazy operation. Make sure that save and load folder has different name.

### 2.3.2 Without drift correction
For example, for scale up by 4 (default) or 8 without drift correction, if you not sure about the number of subframe for each dimension you can select, run command below and follow the instruction of the code to type the range of sub-region you want to predict.
```bash
python main.py evalu -s 4 -L "data/frames/" -S "data/4-save/"
python main.py evalu -s 8 -L "data/frames/" -S "data/8-save/"
```

If you already know the sub-region you want to predict, for example, patch 
`[0, 1)` in Z, `[8, 12)` in Y, and `[9, 13)` in X, pass the range to 
`-r RNG_SUB_USER` as below.
```bash
python main.py evalu -s 4 -r 0 1 8 12 9 13 -L "data/frames/" -S "data/4-save/"
python main.py evalu -s 8 -r 0 1 8 12 9 13 -L "data/frames/" -S "data/8-save/"
```

### 2.3.3 With drift correction
To perform evaluation with drift correction, we split into two steps since temp results for calculating the drift and final prediction may use different region `-r RNG_SUB_USER` of the frames, i.e., a small region for getting temp result to reduce the time of calculating the drift and a large region for the final prediction. We skip examples of passing `-r RNG_SUB_USER` to the command below; it is the same as without drift correction.

#### 2.3.3.1 Step 1: Calculate the drift
First, we predict frames in `-L FRAMES_LOAD_FOLD` like without drift correction where the scale up factor `-s` must set to 4 (default). However, instead of saving the final prediction results in `-S DATA_SAVE_FOLD`, we stack and save `-stride STRIDE` number of prediction results to `-T TEMP_SAVE_FOLD` as temp result then reset. For example, if `-stride STRIDE` is 250, temp result `TEMP_SAVE_FOLD/00250.tif` will be the stack of frames 1-250 prediction results,`TEMP_SAVE_FOLD/00500.tif` will be the stack of frames 251-500 prediction results, and so on. As a comparison, when predicting without drift correction, `DATA_SAVE_FOLD/00500.tif` will be the stack of frames 1-500 prediction results. These temp results will be used to calculate the drift, and the final drift of each frames in all dimension will be saved in `TEMP_SAVE_FOLD/{DCC,MCC,RCC}.csv` depend on the method you choose. 

We highly rely on cached temp result and drift value here:
-   Delete `TEMP_SAVE_FOLD/{DCC,MCC,RCC}.csv` if you want to re-calculate the 
    drift for same dataset with new window size; 
-   In addition, for MCC and RCC, delete `TEMP_SAVE_FOLD/r.csv`, temp result 
    shared between MCC and RCC method. If you have run one of MCC or RCC method 
    and want to try the other, you can keep `TEMP_SAVE_FOLD/r.csv` to save time. 
-   Delete whole `-T TEMP_SAVE_FOLD` if you want to re-calculate the drift for 
    same dataset with new stride size.
-   Delete whole `-T TEMP_SAVE_FOLD` or specify a new path (recommend) if you 
    want to re-calculate the drift for different dataset.

Smaller stride size means more window number, leading to more accurate drift calculation but more time comsuming; big O of DCC is linear to number of windows and MCC and RCC are quadratic to number of windows. We suggest to use DCC to test the window size first and then use MCC or RCC to calculate the final drift.

For example, test the window size 1000, 2000, or 3000 with DCC method
```bash
python main.py evalu -L "data/frames/" -stride 250 -window 1000 -m DCC
python main.py evalu -L "data/frames/" -stride 250 -window 2000 -m DCC
python main.py evalu -L "data/frames/" -stride 250 -window 3000 -m DCC
```
and then use MCC or RCC to calculate the final drift with the best window size, 2000 as a example,
```bash
python main.py evalu -L "data/frames/" -stride 250 -window 2000 -m MCC
python main.py evalu -L "data/frames/" -stride 250 -window 2000 -m RCC
```
Note that we use default `-T TEMP_SAVE_FOLD` here, 
`os.path.dirname(FRAMES_LOAD_FOLD)/temp/`, i.e., `data/temp/`.

#### 2.3.3.2 Step 2: Perform drift correction
With cached drift value, perform drift correction while predicting the frames. Make sure that `-T TEMP_SAVE_FOLD` and `-m {DCC,MCC,RCC}` match the first step. For example, if use default `-T TEMP_SAVE_FOLD` and set `-m {DCC,MCC,RCC}` as RCC in the first step, perform drift correction by
```bash
python main.py evalu -s 4 -L "data/frames/" -S "data/4-RCC/" -m RCC
python main.py evalu -s 8 -L "data/frames/" -S "data/8-RCC/" -m RCC
```

### 2.3.4 Scaling Up
We provide the argument `-r RNG_SUB_USER` in purpose; if your whole frames patch into `(1, 32, 32)` sub-regions in `(Z, Y, X)` but your GPU memeory can only predict 4 sub-regions at a time, you can easily predict the whole frames by a loop script. Here is a simple python example that perform evaluation with drift correction, scale up by 8, 4 sub-regions at a time:
```python
import subprocess
for y in range(0, 32):          # 0, 1, 2, ..., 31
    for x in range(0, 32, 4):   # 0, 4, 8, ..., 28
        s =  "-s 8"
        r = f"-r 0 1 {y} {y+1} {x} {x+4}"
        L =  "-L data/frames/"
        S = f"-S data/8-({y:02d}-{y+1:02d}-{x:02d}-{x+4:02d})-RCC/"
        m =  "-m RCC"
        subprocess.run(
            f"python main.py evalu {s} {r} {L} {S} {m}", 
        check=True, shell=True)
```
Remember to provide unique `-S DATA_SAVE_FOLD` for each loop, like the example we show above, to avoid overwriting the results.

Then, you can concatenate results together to get the prediction for whole frames. We provide a simple script to do this; please check code cell "concatenate two or more 3D subframes into a 3D frame" in `util.ipynb` for more detail.
