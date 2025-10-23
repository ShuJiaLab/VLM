import cupy as cp
from cupyx.scipy.ndimage import zoom
from cucim.skimage.restoration import richardson_lucy
from imageio.v3 import imread, imwrite
from os import system
system("cls")
import time
from glob import glob



# imstack = imread("Z:\\Xuanwen\\DLFLFM\\dldatasets\\RawDatasets\\syn_cfc_norm10_aug\\wf_0000.tif")
# psfstack = imread("Z:\\Xuanwen\\DLFLFM\\expdata100x\\Simu20231212Wv445\\PSFGAUint_20231215_Purple_gly_10um.tif")
# print('imstack shape: ', imstack.shape, 'psfstack shape: ', psfstack.shape)

# imstack = cp.asarray(imstack).astype(cp.float32)
# psfstack = cp.asarray(psfstack).astype(cp.float32)

# imstack = imstack/cp.max(imstack)
# psfstack = psfstack/cp.max(psfstack)

# # imstack = zoom(imstack, (1, 0.5, 0.5))
# # psfstack = zoom(psfstack, (1, 0.5, 0.5))


# time_start = time.time()
# imdeconv = richardson_lucy(imstack, psfstack, num_iter=50, clip=False)
# time_end = time.time()
# print('time cost: ', time_end - time_start, 's')

'''Load PSF into GPU'''
psfstack = imread("Z:\\Xuanwen\\DLFLFM\\expdata100x\\Simu20231212Wv445\\PSFGAUint_20231215_Purple_gly_10um.tif")
psfstack = cp.asarray(psfstack).astype(cp.float32)
psfstack = psfstack/cp.max(psfstack)
print('PSF loaded with shape: ', psfstack.shape)

imstacklist = glob("Z:\\Xuanwen\\DLFLFM\\dldatasets\\RawDatasets\\syn_cfc_norm10_aug\\*.tif")

for i in range(len(imstacklist)):
    imstack = imread(imstacklist[i])
    imstack = cp.asarray(imstack).astype(cp.float32)
    imstackmaxv = cp.max(imstack)
    imstack = imstack/imstackmaxv
    imdeconv = richardson_lucy(imstack, psfstack, num_iter=50, clip=False)
    imdeconv = imdeconv/cp.max(imdeconv)
    imdeconv = cp.around(imdeconv*imstackmaxv).astype(cp.uint16)
    imwrite("Z:\\Xuanwen\\DLFLFM\\dldatasets\\RawDatasets\\syn_cfc_norm10rld_aug\\wf_" + str(i).zfill(4) + ".tif", imdeconv)
    print('Image ' + str(i) + ' deconvolved!')
