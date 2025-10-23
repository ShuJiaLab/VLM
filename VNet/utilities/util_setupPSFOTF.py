import time
import gc
import findpeaks
from numpy import array,ndarray,savetxt
from scipy.io import loadmat
from torch.nn.functional import pad as Fpad
import torch
from torchvision.transforms.functional import resize
from torch.cuda import empty_cache
from utilities.util_getMLAcenters import get_lenslet_centers

def load_PSF(filename, n_depths=120,data_scale=1):
    # Load PSF
    try:
        # Check permute
        psfIn = torch.from_numpy(loadmat(filename)['FLFPSF']).permute(2,0,1).unsqueeze(0)
        if psfIn.shape[2]==1025: psfIn = psfIn[:,:,1:,1:]
        psfIn = resize(psfIn, (round(psfIn.shape[-2]*data_scale),
                               round(psfIn.shape[-1]*data_scale)),antialias=True)
    except:
        import h5py
        psfFile = h5py.File(filename,'r') 
        psfIn = torch.from_numpy(psfFile.get('FLFPSF')[:]).permute(2,0,1).unsqueeze(0)

    # Grab only needed depths
    if n_depths < psfIn.shape[1]:
        psfIn = psfIn[:, psfIn.shape[1]//2- n_depths//2+1 : psfIn.shape[1]//2+n_depths//2+1, ...]
    # Normalize psfIn such that each depth sum is equal to 1
    for nD in range(psfIn.shape[1]):
        psfIn[:,nD,...] = psfIn[:,nD,...] / psfIn[:,nD,...].sum()    
    return psfIn

def load_PSF_OTF(filename, vol_size, n_split=1, n_depths=120, data_scale=1, device="cpu",
                 calc_max=False, psfIn=None, compute_transpose=False,
                 n_lenslets=3, lenslet_centers_file_out='lenslet_centers_python.txt',
                 recalc_lenslet_centers=False):
    # Load PSF
    if psfIn is None:
        psfIn = load_PSF(filename, n_depths,data_scale)
        print(">>> PSF loaded", psfIn.shape)

    if len(lenslet_centers_file_out)>0 and recalc_lenslet_centers:
        print(">>> Recalculating lenslet centers, original coordinates: ",get_lenslet_centers(lenslet_centers_file_out))
        find_lenslet_centers(psfIn[0,n_depths//2,...].numpy(), n_lenslets=n_lenslets, file_out_name=lenslet_centers_file_out)
    lenscoords = get_lenslet_centers(lenslet_centers_file_out)
    print(">>> Using", lenscoords.shape[0], "lenslets, Current coords:", lenscoords)       

    psf_shape = torch.tensor(psfIn.shape[2:])
    print(">>> PSF shape: " + str(psf_shape))
    vol = torch.rand(1,psfIn.shape[1], vol_size[0], vol_size[1], device=device)
    print(">>> Vol rand shape: " + str(vol.shape))
    img, OTF = fft_conv_split(vol, psfIn.float().detach().to(device), psf_shape, n_split=n_split, B_precomputed=False, device=device)
    OTF = OTF.detach()
    if compute_transpose:
        OTFt = torch.real(OTF) - 1j * torch.imag(OTF)
        OTF = torch.cat((OTF.unsqueeze(-1), OTFt.unsqueeze(-1)), 4)
    if calc_max:
        psfMaxCoeffs = torch.amax(psfIn, dim=[0,2,3])
        return OTF, psf_shape, psfMaxCoeffs
    else:
        return OTF,psf_shape
    
def fft_conv_split(A, B, psf_shape, n_split, B_precomputed=False, device = "cpu"):
    n_depths = A.shape[1]   
    split_conv = n_depths//n_split
    depths = list(range(n_depths))
    depths = [depths[i:i + split_conv] for i in range(0, n_depths, split_conv)]

    fullSize = torch.tensor(A.shape[2:]) + psf_shape
    
    crop_pad = [(psf_shape[i] - fullSize[i])//2 for i in range(0,2)]
    crop_pad = (crop_pad[1], (psf_shape[-1]- fullSize[-1])-crop_pad[1], crop_pad[0], (psf_shape[-2] - fullSize[-2])-crop_pad[0])
    # Crop convolved image to match size of PSF
    img_new = torch.zeros(A.shape[0], 1, psf_shape[0], psf_shape[1], device=device)
    if B_precomputed==False:
        OTF_out = torch.zeros(1, n_depths, fullSize[0], fullSize[1]//2+1, requires_grad=False, dtype=torch.complex64, device=device)
    for n in range(n_split):
        curr_psf = B[:,depths[n],...].to(device)
        img_curr = fft_conv(A[:,depths[n],...].to(device), curr_psf, fullSize, B_precomputed)
        if B_precomputed == False:
            OTF_out[:,depths[n],...] = img_curr[1]
            img_curr = img_curr[0]
        img_curr = Fpad(img_curr, crop_pad)
        img_new += img_curr[:,:,:psf_shape[0],:psf_shape[1]].sum(1).unsqueeze(1).abs()   
    
    if B_precomputed == False:
        return img_new, OTF_out
    return img_new

# FFT convolution, the kernel fft can be precomputed
def fft_conv(A,B, fullSize,B_precomputed=False):
    nDims = A.ndim-2
    padSizeA = (fullSize - torch.tensor(A.shape[2:]))
    padSizesA = torch.zeros(2*nDims,dtype=int)
    padSizesA[0::2] = torch.floor(padSizeA/2.0)
    padSizesA[1::2] = torch.ceil(padSizeA/2.0)
    padSizesA = list(padSizesA.numpy()[::-1])
    A_padded = Fpad(A,padSizesA)
    Afft = torch.fft.rfft2(A_padded)
    if B_precomputed:
        return batch_fftshift2d_real(torch.fft.irfft2( Afft * B.detach()))
    else:
        padSizeB = (fullSize - torch.tensor(B.shape[2:]))
        padSizesB = torch.zeros(2*nDims,dtype=int)
        padSizesB[0::2] = torch.floor(padSizeB/2.0)
        padSizesB[1::2] = torch.ceil(padSizeB/2.0)
        padSizesB = list(padSizesB.numpy()[::-1])
        B_padded = Fpad(B,padSizesB)
        Bfft = torch.fft.rfft2(B_padded)
        return batch_fftshift2d_real(torch.fft.irfft2( Afft * Bfft.detach())), Bfft.detach()

# Aid functions for shiftfft2
def batch_fftshift2d_real(x):
    out = x
    for dim in range(2, len(out.size())):
        n_shift = x.size(dim)//2
        if x.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images

        f_idx = tuple(slice(None, None, None) if i != dim else slice(0, n_shift, None) for i in range(out.dim()))
        b_idx = tuple(slice(None, None, None) if i != dim else slice(n_shift, None, None) for i in range(out.dim()))
        front = out[f_idx]
        back = out[b_idx]
        out = torch.cat([back, front], dim)
    return out  

def find_lenslet_centers(img, n_lenslets=29, file_out_name='lenslet_centers_python.txt'):
    fp2 = findpeaks.findpeaks(method='topology',whitelist=['peak'])
    image_divisor = 2 # To find the centers faster
    img = findpeaks.stats.resize(img, size=(int(img.shape[0]*1.0/image_divisor),
                                            int(img.shape[1]*1.0/image_divisor)))
    fp2.fit(img)
    limit_min = fp2.results['persistence'][0:n_lenslets+1]['score'].min()
    # Initialize topology
    fp = findpeaks.findpeaks(method='topology',whitelist=['peak'], limit=limit_min)
    results = fp.fit(img)
    results = ndarray([n_lenslets,2], dtype=int)
    # fp.plot_persistence()
    for ix,data in enumerate(fp.results['groups0'][0:n_lenslets]):
        results[ix] = array(data[0], dtype=int) * image_divisor
    if len(file_out_name) > 0:
        print(">>> Fitted lenslet centers to " + file_out_name)
        savetxt(file_out_name, results, fmt='%d', delimiter='\t')
    return results

def setupPSFOTF(args, device, n_depths,recalcPSFcenters=False):
    if len(args.gpu_repro)>0:
        S = time.time()
        # Load PSF and compute OTF
        # n_split = args.n_split
        OTF,psf_shape = load_PSF_OTF(args.psf_file, args.output_shape, n_depths=n_depths,
                                    data_scale=args.data_rescale,n_lenslets=3, device="cpu",
                                    lenslet_centers_file_out=args.lenslet_file,
                                    recalc_lenslet_centers=recalcPSFcenters)
        OTF = OTF.to(device)
        gc.collect()
        empty_cache()
        E = time.time()
        print("PSF OTF loading time: ",round(E - S,2),"s. PSF shape: ",psf_shape," OTF shape: ",OTF.shape)
        gc.collect()
        empty_cache()
        return OTF, psf_shape