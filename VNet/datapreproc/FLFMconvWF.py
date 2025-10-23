from glob import glob
from sys import path as sys_path
from os.path import basename
from imageio.v3 import imread, imwrite
from scipy.io import loadmat
import numpy as np
# import matplotlib.pyplot as plt
from cupyx.scipy.ndimage import zoom, convolve
import cupy as cp
# from rich.progress import Progress, track
# from cupy.fft import rfft2, irfft2, ifftshift

def loadWFPSF(psffolder,windowsize):
    psfmat = loadmat(psffolder)
    psf = cp.asarray(psfmat['FLFPSF']).astype(cp.float32).transpose(2,0,1)
    if psf.shape[1]//2!=psf.shape[1]/2:
        psf_iso = zoom(psf[1:,0:-1,0:-1],(65.0/65,1,1))
    else:
        psf_iso = zoom(psf[1:,:,:],(65.0/65,1,1))
    psfIn_midindex_0 = int(psf_iso.shape[0]/2)
    psfIn_midindex_1 = int(psf_iso.shape[1]/2)
    psfIn_midindex_2 = int(psf_iso.shape[2]/2)
    windowsize = 16
    psfIn_center = psf_iso[psfIn_midindex_0-windowsize:psfIn_midindex_0+windowsize,
                           psfIn_midindex_1-windowsize:psfIn_midindex_1+windowsize,
                           psfIn_midindex_2-windowsize:psfIn_midindex_2+windowsize]
    print(">>> psf loaded in shape:", psfIn_center.shape)
    return psfIn_center

def addAperture(vol,radius,depth):
    vol_midindex_0 = int(vol.shape[0]/2)
    vol_midindex_1 = int(vol.shape[1]/2)
    vol_midindex_2 = int(vol.shape[2]/2)
    circle_aperture = cp.zeros_like(vol[0,:,:])
    X,Y = cp.meshgrid(cp.arange(-vol_midindex_1,vol_midindex_1),cp.arange(-vol_midindex_2,vol_midindex_2))
    circle_aperture[cp.sqrt(X**2+Y**2)<radius] = 1.0
    depth = int(depth/2)
    volout = cp.zeros_like(vol)
    volout[vol_midindex_0-depth:vol_midindex_0+depth,:,:] = vol[vol_midindex_0-depth:vol_midindex_0+depth,:,:]*circle_aperture[None,:,:]
    return volout

def convolveWFM(currVol,psfIn):
    psfIn_midindex_0 = int(psfIn.shape[0]/2)
    psfIn_midindex_1 = int(psfIn.shape[1]/2)
    psfIn_midindex_2 = int(psfIn.shape[2]/2)
    windowsize = 16
    psfIn_center = psfIn[psfIn_midindex_0-windowsize:psfIn_midindex_0+windowsize,
                         psfIn_midindex_1-windowsize:psfIn_midindex_1+windowsize,
                         psfIn_midindex_2-windowsize:psfIn_midindex_2+windowsize]
    currVol_conv = convolve(currVol,psfIn_center)
    return currVol_conv


if __name__=="__main__":
    from os import system
    from os.path import basename, exists
    system('cls')

    # load widefield PSF
    psffolder = "Z:\\Xuanwen\\DLFLFM\\expdata100x\\Simu20231212Wv445\\PSFGAUint_20231215_Purple_gly_10um.mat"
    windowsize = 16
    psfIn_center = loadWFPSF(psffolder,windowsize)

    # convolve gtc to cfc
    gtcfolder = "Z:\\Xuanwen\\DLFLFM\\dldatasets\\RawDatasets\\syn_gtc_norm10_aug\\"
    cfcfolder = "Z:\\Xuanwen\\DLFLFM\\dldatasets\\RawDatasets\\syn_cfc_norm10_aug\\"
    gtclist = glob(gtcfolder+"*.tif")
    isreplace = False
    for ii in range(0,len(gtclist)):
        filename = basename(gtclist[ii])
        print(">>> Processing:", filename)
        if not(isreplace) and exists(cfcfolder+"\\"+filename):
            print("--- File already exists, skip:", filename)
        else:
            vol = cp.asarray(imread(gtclist[ii])).astype(cp.float32)
            if vol.shape[1]//2!=vol.shape[1]/2:
                vol = vol[:,0:-1,0:-1]
            # vol = addAperture(vol,500,100)
            cfc = convolveWFM(vol,psfIn_center).get()
            imwrite(cfcfolder+"\\"+filename, (cfc/np.max(cfc)*60000).astype(np.uint16))
            print("*** Saved:", filename)
