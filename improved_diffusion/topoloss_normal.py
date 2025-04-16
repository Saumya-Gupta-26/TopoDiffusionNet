# Topoloss to train the diffusion model

from __future__ import print_function, division

import torch
#import time
import numpy as np
#from pylab import *
import torch.nn.functional as F
from gudhi.wasserstein import wasserstein_distance
import sys
import cripser as cr
import pdb

# debug flag. If set to True, prints additional information useful for debugging.
printstuff = False

class SimulTopoLossMSE2D(torch.nn.Module):
    """calculates 0dim and 1dim topoloss simultaneously
    """
    def __init__(self, topo_birth, topo_death, topo_noisy):
        super().__init__()
        print("[TDN] Applying topoloss simultaneously to 0dim and 1dim")
        print("[TDN] Topo birth/death/noisy: {}, {}, {}".format(topo_birth, topo_death, topo_noisy))
        self.topo1 = TopoLossMSE2D(topo_birth, topo_death, topo_noisy, 0)
        self.topo2 = TopoLossMSE2D(topo_birth, topo_death, topo_noisy, 1)

    def forward(self, pred, target):
        ans = self.topo1(pred, target) + self.topo2(pred, target)
        return ans


class TopoLossMSE2D(torch.nn.Module):
    """Weighted Topological loss
    """

    def __init__(self, topo_birth, topo_death, topo_noisy, topo_dim):
        super().__init__()
        print("[TDN] Topo dim: {}".format(topo_dim))
        print("[TDN] Topo birth/death/noisy: {}, {}, {}".format(topo_birth, topo_death, topo_noisy))
        self.topo_birth = topo_birth # penalize only births
        self.topo_death = topo_death # penalize only deaths
        self.topo_noisy = topo_noisy # penalize noisy too
        self.topo_dim = topo_dim # dimension of the topological features


    def forward(self, pred, target):
        loss_val = [0.] * pred.size()[0]

        for idx in range(pred.size()[0]): # batchsize=N
            for ch in range(pred.size()[1]): # n_channel=C 
                loss_val[idx] += getTopoLoss2d(pred[idx, ch, :, : ], target[idx, ch, :, : ], self.topo_birth, self.topo_death, self.topo_noisy, self.topo_dim)
            loss_val[idx] /= pred.size()[1] # divide by the number of channels
            if not torch.is_tensor(loss_val[idx]):
                loss_val[idx] = torch.tensor(loss_val[idx]).cuda()
            loss_val[idx] = torch.unsqueeze(loss_val[idx], 0)

        return torch.cat(loss_val) # list of values for each sample in the batch


def compute_dgm_force(stu_lh_dgm, tea_lh_dgm):
    """
    Compute the persistent diagram of the image

    Args:
        stu_lh_dgm: likelihood persistent diagram of student model.
        tea_lh_dgm: likelihood persistent diagram of teacher model.

    Returns:
        idx_holes_to_remove: The index of student persistent points that require to remove for the following training process [aka stu dots matched to diagonal]
        off_diagonal_match: The index pairs of persistent points that requires to fix in the following training process [aka stu dots matched to tea]
    
    """
    if stu_lh_dgm.shape[0] == 0:
        idx_holes_to_remove, off_diagonal_match = np.zeros((0,2)), np.zeros((0,2))
        return idx_holes_to_remove, off_diagonal_match
    
    if (tea_lh_dgm.shape[0] == 0):
        tea_pers = None
        tea_n_holes = 0
    else:
        tea_pers = abs(tea_lh_dgm[:, 1] - tea_lh_dgm[:, 0])
        tea_n_holes = tea_pers.size

    if (tea_pers is None or tea_n_holes == 0):
        idx_holes_to_remove = list(set(range(stu_lh_dgm.shape[0])))
        off_diagonal_match = list()
    else:
        idx_holes_to_remove, off_diagonal_match = get_matchings(stu_lh_dgm, tea_lh_dgm)
    
    return idx_holes_to_remove, off_diagonal_match


def getCriticalPoints_cr(likelihood, topo_dim, threshold):
        
    lh = 1 - likelihood
    pd = cr.computePH(lh, maxdim=1, location="birth") # dim birth death x1  y1  z1  x2  y2  z2
    pd_arr_lh = pd[pd[:, 0] == topo_dim] # 0 or 1-dim topological features
    pd_lh = pd_arr_lh[:, 1:3] # birth time and death time
    # birth critical points
    bcp_lh = pd_arr_lh[:, 3:5]
    # death critical points
    dcp_lh = pd_arr_lh[:, 6:8]
    pairs_lh_pa = pd_arr_lh.shape[0] != 0 and pd_arr_lh is not None

    # if the death time is inf, set it to 1.0
    for i in pd_lh:
        if i[1] > 1.0:
            i[1] = 1.0
    
    pd_pers = abs(pd_lh[:, 1] - pd_lh[:, 0])
    valid_idx = np.where(pd_pers > threshold)[0]
    noisy_idx = np.where(pd_pers <= threshold)[0]

    #return pd_lh_filtered, bcp_lh_filtered, dcp_lh_filtered, pairs_lh_pa
    return pd_lh, bcp_lh, dcp_lh, pairs_lh_pa, valid_idx, noisy_idx

def get_matchings(lh_stu, lh_tea):
    
    cost, matchings = wasserstein_distance(lh_stu, lh_tea, matching=True)

    dgm1_to_diagonal = matchings[matchings[:,1] == -1, 0] # dots in stu that matched to diagonal
    off_diagonal_match = np.delete(matchings, np.where(matchings == -1)[0], axis=0) # remove any diagonal dots in stu (tea had extra dots which mapped to stu diagonal)

    return dgm1_to_diagonal, off_diagonal_match

def interpolate(nparr, omin = 0., omax = 1.):
    imin  = np.min(nparr)
    imax = np.max(nparr)

    denom = imax - imin 
    if denom == 0:
        denom = 1
    return (nparr-imin)*(omax-omin)/denom + omin

def getTopoLoss2d(inter_tensor, xstart_tensor, topo_birth, topo_death, topo_noisy, topo_dim, pd_threshold = 0.):
    if inter_tensor.ndim != 2:
        print("incorrct dimension")
    
    inter = inter_tensor.clone()
    xstart = xstart_tensor.clone()

    inter = torch.squeeze(inter).cpu().detach().numpy()
    xstart = torch.squeeze(xstart).cpu().detach().numpy()

    # inter_tensor, xstart_tensor are in [-1, 1] range
    # do normalization to [0,1] so that the PD calculations are correct. But when actual topoloss is computed, use the original [-1,1] itself

    likelihood = interpolate(inter)
    gt = interpolate(xstart)

    topo_cp_weight_map = np.zeros(likelihood.shape)
    topo_cp_ref_map = np.zeros(likelihood.shape)

    if(np.min(likelihood) == 1 or np.max(likelihood) == 0): return 0.
    if(np.min(gt) == 1 or np.max(gt) == 0): return 0.
    
    # Get the critical points of predictions and ground truth
    pd_lh, bcp_lh, dcp_lh, pairs_lh_pa, valid_idx_lh, noisy_idx_lh = getCriticalPoints_cr(likelihood, topo_dim, threshold=pd_threshold)
    pd_gt, bcp_gt, dcp_gt, pairs_lh_gt, valid_idx_gt, noisy_idx_gt = getCriticalPoints_cr(gt, topo_dim, threshold=0.)

    # select pd with high threshold to match
    pd_lh_for_matching = pd_lh[valid_idx_lh]
    pd_gt_for_matching = pd_gt[valid_idx_gt]

    # If the pairs not exist, continue for the next loop
    if not(pairs_lh_pa): return 0.
    if not(pairs_lh_gt): return 0.

    idx_holes_to_remove_for_matching, off_diagonal_for_matching = compute_dgm_force(pd_lh_for_matching, pd_gt_for_matching)

    idx_holes_to_remove = []
    off_diagonal_match = []

    if (len(idx_holes_to_remove_for_matching) > 0):
        for i in idx_holes_to_remove_for_matching:
            index_pd_lh_removed = np.where(np.all(pd_lh == pd_lh_for_matching[i], axis=1))[0][0]
            idx_holes_to_remove.append(index_pd_lh_removed)
    
    for k in noisy_idx_lh:
        idx_holes_to_remove.append(k)
    
    if len(off_diagonal_for_matching) > 0:
        for idx, (i, j) in enumerate(off_diagonal_for_matching):
            index_pd_lh = np.where(np.all(pd_lh == pd_lh_for_matching[i], axis=1))[0][0]
            index_pd_gt = np.where(np.all(pd_gt == pd_gt_for_matching[j], axis=1))[0][0]
            off_diagonal_match.append((index_pd_lh, index_pd_gt))

    if (len(off_diagonal_match) > 0 or len(idx_holes_to_remove) > 0):
        for (idx, (hole_indx, j)) in enumerate(off_diagonal_match):
            if topo_birth and (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] and int(
                    bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]):
                topo_cp_weight_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] = 1 # push birth to the corresponding teacher birth i.e. min birth prob or likelihood
                topo_cp_ref_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] = xstart[int(bcp_gt[j][0]), int(bcp_gt[j][1])] #pd_gt[j][0]
            
            if topo_death and (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                    likelihood.shape[1]):
                topo_cp_weight_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] = 1  # push death to the corresponding teacher death i.e. max death prob or likelihood
                topo_cp_ref_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] = xstart[int(dcp_gt[j][0]), int(dcp_gt[j][1])] #pd_gt[j][1]
        
        if topo_noisy:
            for hole_indx in idx_holes_to_remove:
                if topo_birth and (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                    0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                        likelihood.shape[1]):
                    topo_cp_weight_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] = 1  # push to diagonal
                    
                    if (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_ref_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] = \
                            inter[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] # lh_patch[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]
                    else:
                        topo_cp_ref_map[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] = 1
                
                if topo_death and (int(dcp_lh[hole_indx][0]) >= 0 and int(dcp_lh[hole_indx][0]) < likelihood.shape[
                    0] and int(dcp_lh[hole_indx][1]) >= 0 and int(dcp_lh[hole_indx][1]) <
                        likelihood.shape[1]):
                    topo_cp_weight_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] = 1  # push to diagonal
                    
                    if (int(bcp_lh[hole_indx][0]) >= 0 and int(bcp_lh[hole_indx][0]) < likelihood.shape[
                        0] and int(bcp_lh[hole_indx][1]) >= 0 and int(bcp_lh[hole_indx][1]) <
                            likelihood.shape[1]):
                        topo_cp_ref_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] = \
                            inter[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])] # lh_patch[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]
                    else:
                        topo_cp_ref_map[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])] = 0

    topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).cuda()
    topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).cuda()

    # Measuring the MSE loss between predicted critical points and reference critical points
    loss_topo = (((inter_tensor * topo_cp_weight_map) - topo_cp_ref_map) ** 2).sum()

    if printstuff:
        print("\nTopoloss AKA Image [-1,1] Intensity: {}".format(loss_topo.item()))

    return loss_topo