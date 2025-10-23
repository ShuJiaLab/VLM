from sys import exit
from utilities.util_imgRegister import imRegister,imshowfalsecolor
from matplotlib import pyplot as plt
from torch import tensor

def checkDataset(dataset):
    extracted_views = dataset.extract_views(dataset.stacked_views.unsqueeze(1), dataset.vols,dataset.lenslet_coords,dataset.subimage_shape)[:,0,...]
    showidx = 5
    trainvolshow = dataset.vols[showidx,:,:,:].max(dim=0).values
    plt.figure(3)
    plt.subplot(231)
    plt.imshow(imshowfalsecolor(trainvolshow,extracted_views[showidx,0,:,:],k=0.5))
    plt.subplot(232)
    plt.imshow(imshowfalsecolor(trainvolshow,extracted_views[showidx,1,:,:],k=0.5))
    plt.subplot(233)
    plt.imshow(imshowfalsecolor(trainvolshow,extracted_views[showidx,2,:,:],k=0.5))
    plt.subplot(234)
    plt.imshow(trainvolshow*(trainvolshow>0.0*trainvolshow.max()))

    img_transformed,outTx = imRegister(trainvolshow, extracted_views[showidx,2,:,:])
    plt.subplot(235)
    plt.imshow(imshowfalsecolor(trainvolshow,extracted_views[showidx,2,:,:],k=0.5))
    plt.subplot(236)
    plt.imshow(imshowfalsecolor(trainvolshow,tensor(img_transformed).half(),k=0.5))
    plt.show()
    exit()