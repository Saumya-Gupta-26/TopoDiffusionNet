# Compute classification metrics for the evaluation : Accuracy, Precision, Recall, F1 score
# Call this after extract-npz.py
# This script will write results to file metrics.txt in the pred_dir folder

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os, glob, sys 
from PIL import Image
from scipy import ndimage # for connected components
import cripser as cr

# NEED TO BE FILLED BY YOU
pred_dir = "" # no trailing / ; directory containing the predicted images.
ch = 3 # Number of channels in the image; 1 or 3
classes = [1,2,3,4,5] # Set the list of topological constraints to evaluate on. 
topo_dim = 0 # 0 or 1; 0-dim topology (connected components) or 1-dim toplogy (holes)


def getCriticalPoints_cr(likelihood, threshold = 0.0):
    lh = 1 - likelihood
    pd = cr.computePH(lh, maxdim=1) # dim birth death x1 y1 z1 x2 y2 z2
    pd_arr_lh = pd[pd[:, 0] == topo_dim] # 1-dim or 0-dim topological features
    pd_lh = pd_arr_lh[:, 1:]

    # if the death time is inf, set it to 1.0
    for i in pd_lh:
        if i[1] > 1.0:
            i[1] = 1.0
    
    pd_pers = abs(pd_lh[:, 1] - pd_lh[:, 0])
    valid_idx = np.where(pd_pers > threshold)[0]

    pd_lh_filtered = pd_lh[valid_idx]

    return pd_lh_filtered # birth death bx by bz dx dy dz

def compute_average(y_true, y_pred):
    denom = len(y_true) / len(classes)
    print("num samples per class: {}".format(denom))

    acc = {}

    for idx, gt in enumerate(y_true):
        ans = int(y_pred[idx] == gt)
        if gt not in acc:
            acc[gt] = 0
        acc[gt] += ans
    
    ans = []
    for c in classes:
        ans.append(acc[c] / denom)
    return ans


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0, labels=classes) # None returns scores for all classes
    recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=classes)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=classes)
    return accuracy, precision, recall, f1

# for 0-dimensional topological features
def main_dim0():
    filepaths = glob.glob(pred_dir + "/*.png")
    denom = len(classes)
    
    with open(os.path.join(pred_dir, "metrics.txt"), "w") as wfile:
        wfile.write("Pred dir: {}\n\n".format(pred_dir))
        wfile.write("Num samples: {}".format(len(filepaths)))
        
        # need to populate these first
        y_true = []
        y_pred_nodiag = [] # Assuming 4-connectivity within the mask
        y_pred_diag = [] # Assuming 8-connectivity within the mask 

        for fp in filepaths:
            label = int(fp.split("label")[0].split("_")[-1].split('.')[0])

            y_true.append(label)
            img = np.array(Image.open(fp))
            if ch == 3:
                img = img[:,:,0] # take 1 channel
            img[img > 128] = 255
            img[img <= 128] = 0

            _, num_objs = ndimage.label(img)
            y_pred_nodiag.append(num_objs)

            _, num_objs = ndimage.label(img, structure=ndimage.generate_binary_structure(2,2))
            y_pred_diag.append(num_objs)

        y_true = np.array(y_true)
        y_pred_nodiag = np.array(y_pred_nodiag)
        y_pred_diag = np.array(y_pred_diag)
        
        wfile.write("\n\ny_true: {}".format(np.unique(y_true, return_counts=True)))
        wfile.write("\n\ny_pred_nodiag: {}".format(np.unique(y_pred_nodiag, return_counts=True)))
        wfile.write("\n\ny_pred_diag: {}".format(np.unique(y_pred_diag, return_counts=True)))          

        acc_avg, p, r, f1 = compute_metrics(y_true, y_pred_nodiag)
        acc_multiclass = compute_average(y_true, y_pred_nodiag)
        wfile.write("\n\n4-connectivity: Diagonal touch => Not connected\n")
        wfile.write("Accuracy: {} +- {}\n{}\n".format(acc_avg, np.std(acc_multiclass), acc_multiclass))
        wfile.write("\nPrecision: {} +- {}\n{}\n".format(sum(p)/denom, np.std(p), p))
        wfile.write("\nRecall: {} +- {}\n{}\n".format(sum(r)/denom, np.std(r),r))
        wfile.write("\nF1: {} +- {}\n{}\n".format(sum(f1)/denom, np.std(f1), f1))
        mae_list = np.abs(y_true - y_pred_nodiag)
        wfile.write("\nMAE: {} +- {}\n".format(np.mean(mae_list), np.std(mae_list)))

        acc_avg, p, r, f1 = compute_metrics(y_true, y_pred_diag)
        acc_multiclass = compute_average(y_true, y_pred_diag)
        wfile.write("\n\n8-connectivity: Diagonal touch => Connected\n")
        wfile.write("Accuracy: {} +- {}\n{}\n".format(acc_avg, np.std(acc_multiclass), acc_multiclass))
        wfile.write("\nPrecision: {} +- {}\n{}\n".format(sum(p)/denom, np.std(p), p))
        wfile.write("\nRecall: {} +- {}\n{}\n".format(sum(r)/denom, np.std(r),r))
        wfile.write("\nF1: {} +- {}\n{}\n".format(sum(f1)/denom, np.std(f1), f1))
        mae_list = np.abs(y_true - y_pred_diag)
        wfile.write("\nMAE: {} +- {}\n".format(np.mean(mae_list), np.std(mae_list)))

# for 1-dimensional topological features
def main_dim1():
    filepaths = glob.glob(pred_dir + "/*.png")
    denom = len(classes)
    
    with open(os.path.join(pred_dir, "metrics.txt"), "w") as wfile:
        wfile.write("Pred dir: {}\n\n".format(pred_dir))
        wfile.write("Num samples: {}".format(len(filepaths)))
        
        # need to populate these first
        y_true = []
        y_pred = []

        for fp in filepaths:
            label = int(fp.split("_")[-1].split("label")[0].split('.')[0])

            y_true.append(label)
            img = np.array(Image.open(fp))
            if ch == 3:
                img = img[:,:,0] # take 1 channel
            img[img > 128] = 255
            img[img <= 128] = 0

            img = img / 255.
            pd_lh = getCriticalPoints_cr(img)
            num_holes = len(pd_lh)

            y_pred.append(num_holes)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        wfile.write("\n\ny_true: {}".format(np.unique(y_true, return_counts=True)))
        wfile.write("\n\ny_pred: {}".format(np.unique(y_pred, return_counts=True)))

        acc_avg, p, r, f1 = compute_metrics(y_true, y_pred)
        acc_multiclass = compute_average(y_true, y_pred)
        wfile.write("\n\nTrue metrics\n")
        wfile.write("Accuracy: {} +- {}\n{}\n".format(acc_avg, np.std(acc_multiclass), acc_multiclass))
        wfile.write("\nPrecision: {} +- {}\n{}\n".format(sum(p)/denom, np.std(p), p))
        wfile.write("\nRecall: {} +- {}\n{}\n".format(sum(r)/denom, np.std(r),r))
        wfile.write("\nF1: {} +- {}\n{}\n".format(sum(f1)/denom, np.std(f1), f1))
        mae_list = np.abs(y_true - y_pred)
        wfile.write("\nMAE: {} +- {}\n".format(np.mean(mae_list), np.std(mae_list)))



if __name__ == "__main__":
    if topo_dim == 0:
        main_dim0()
    elif topo_dim == 1:
        main_dim1()
    else:
        print("Invalid topo_dim")
