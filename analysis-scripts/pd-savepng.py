'''
Generates persistence diagram for an image
'''

import numpy as np
import matplotlib.pyplot as plt
import cripser as cr
import persim
#import cv2
from PIL import Image
import warnings
import os, glob, sys, pdb
import seaborn as sns

plt.rcParams["figure.figsize"] = (5,5)

warnings.filterwarnings("ignore")

# NEED TO BE FILLED BY YOU
srcpath = "" # path to image. Eg: dataset-format/shapes/0-dim/6_xxx.png
savename = srcpath.replace(".png", "_pd.png")
topo_dim = 0 # 0 or 1; 0-dim topology (connected components) or 1-dim toplogy (holes)
isbin = True # whether the image is binary or not. If not, it will be thresholded

def main():

    likelihood = np.array(Image.open(srcpath)) / 255.
    if isbin:
        likelihood[likelihood <= 0.5] = 0.0
        likelihood[likelihood > 0.5] = 1.0

    print("Likelihood range: ", np.min(likelihood), np.max(likelihood))

    pd_lh_0, pd_lh_1 = getCriticalPoints_cr(likelihood, threshold=0.)

    print("Num dots: ", len(pd_lh_1))

    if topo_dim == 0:
        pd_lh = pd_lh_0
    elif topo_dim == 1:
        pd_lh = pd_lh_1

    sns.set_style("dark")
    sns.scatterplot(x=pd_lh[:, 0], y=pd_lh[:, 1], hue=pd_lh[:, 1] - pd_lh[:, 0], palette="flare", edgecolor='none')
    plt.plot([0, 1], [0, 1], ls="dashed", color="black")
    plt.legend([],[], frameon=False)
    plt.savefig(savename)



def getCriticalPoints_cr(likelihood, threshold):
    lh = 1- likelihood #lh = likelihood
    pd = cr.computePH(lh, maxdim=1, location="birth")
    
    print("computePH output: Num rows: {}\ndim birth death x1  y1  z1  x2  y2  z2\n{}".format(pd.shape[0],pd))

    pd_arr_lh = pd[pd[:, 0] == 0] # 0-dim topological features ; for 1-dim topo features, use: pd_arr_lh = pd[pd[:, 0] == 1]
    pd_lh_dim0 = pd_arr_lh[:, 1:3] # birth time and death time

    pd_arr_lh = pd[pd[:, 0] == 1] # 1-dim topological features 
    pd_lh_dim1 = pd_arr_lh[:, 1:3] # birth time and death time

    # if the death time is inf, set it to 1.0
    for i in pd_lh_dim0:
        if i[1] > 1.0:
            i[1] = 1.0

    for i in pd_lh_dim1:
        if i[1] > 1.0:
            i[1] = 1.0

    return pd_lh_dim0, pd_lh_dim1

if __name__ == "__main__":
    main()