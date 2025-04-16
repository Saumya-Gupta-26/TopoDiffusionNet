'''
image_sample.py generates an .npz file. This script extracts the images from the npz, when `save_intermediate_xstarts = True` in image_sample.py
'''

import numpy as np 
import pdb, os, sys, glob
from PIL import Image

# NEED TO BE FILLED BY YOU
srcdir="" # path to directory where image_sample.py stored predictions
ch = 3 # Number of channels in the image; 1 or 3
totalsteps = 999  # Highest timestep in diffusion process

filepaths = glob.glob(srcdir + "/*xstart*steps.npz")

for fp in filepaths:
    print("Considering {}".format(fp))
    npzdata = np.load(fp)['arr_0'] #(num_samples, image_size, image_size, 3)
    savedir = fp.replace(".npz", "")
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for i in range(npzdata.shape[0]):
        if ch == 1:
            img =  Image.fromarray(npzdata[i][:,:,0]) # take 1 channel
        elif ch == 3:
            img =  Image.fromarray(npzdata[i]) # take 3 channels

        savename = "timestep_{}_{}ch_x0.png".format(str(totalsteps - i).zfill(4), ch)
        img.save(os.path.join(savedir, savename))

print("Done!")

filepaths = glob.glob(srcdir + "/*epsilon*steps.npz")

for fp in filepaths:
    print("Considering {}".format(fp))
    npzdata = np.load(fp)['arr_0'] #(num_samples, image_size, image_size, 3)
    savedir = fp.replace(".npz", "")
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    #pdb.set_trace()
    for i in range(npzdata.shape[0]):
        if ch == 1:
            img =  Image.fromarray(npzdata[i][:,:,0]) # take 1 channel
        elif ch == 3:
            img =  Image.fromarray(npzdata[i]) # take 3 channels

        savename = "timestep_{}_{}ch_eps.png".format(str(totalsteps - i).zfill(4), ch)
        img.save(os.path.join(savedir, savename))

print("Done!")


filepaths = glob.glob(srcdir + "/*xt*steps.npz")

for fp in filepaths:
    print("Considering {}".format(fp))
    npzdata = np.load(fp)['arr_0'] #(num_samples, image_size, image_size, 3)
    savedir = fp.replace(".npz", "")
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for i in range(npzdata.shape[0]):
        if ch == 1:
            img =  Image.fromarray(npzdata[i][:,:,0]) # take 1 channel
        elif ch == 3:
            img =  Image.fromarray(npzdata[i]) # take 3 channels

        savename = "timestep_{}_{}ch_xt.png".format(str(totalsteps - i).zfill(4), ch)
        img.save(os.path.join(savedir, savename))

print("Done!")