""" subJob main.py --long --env light2 --memory 180 --gpu_type 'a100-pcie-80gb' """
import pandas as pd
import os
import sys
sys.path.append('../')
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging, LearningRateMonitor
from utils_entropy import read_yaml
from PredEntropyData import build_pred_data
from lightning.pytorch.callbacks import TQDMProgressBar
from model import MAE_Module
import torch
import torchmetrics

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import DataLoader


cfg = read_yaml('../config.yaml')

pl.seed_everything(123456)
print('Torch version:', torch.__version__ )

# Load data
data = build_pred_data(cfg['DATA']['training_files'], cfg['MODEL']['img_size'], cfg['MODEL']['img_stride'], mito_perc_thresh=cfg['DATA']['mito_perc_thresh'])
train_loader = DataLoader(data, batch_size=cfg['TRAINING']['batch_size'], shuffle=True, num_workers=8)

# Load model
model = MAE_Module(cfg)

model_path = '/well/rittscher/users/jyo949/SegmentationChecker/results/SmartMasking_128_Patch4_Emb128_Weights2/version_0/'
ckpt_file = 'checkpoints/epoch=199-step=157000.ckpt'
cfg = read_yaml(os.path.join(model_path, 'hparams.yaml'))['cfg']
model.load_state_dict(torch.load(os.path.join(model_path,ckpt_file))['state_dict'])

model = torch.compile(model)


model = model.to('cuda')


# # Get mean entropy per patch

def adjust_masks(m, dist, ent=None, ent_thres=0.4, mask=None):
    m[dist > 10] = 5
    if ent is not None:
        m[ent > ent_thres] = 5
    m[mask==0] = 5
    return m

def adjust_masks_unc(m, dist, ent=None, ent_thres=0.4, mask=None):
    m[dist > 10] = 5
    if ent is not None:
        m[ent < ent_thres] = 5
    m[mask==0] = 5
    return m


results = {}
results_uncertainty = {}
uncertaint_threshold = [0.1, 0.2, 0.4]
for perc in [0.1, 0.25, 0.5, 0.75, 0.9]:
    it = iter(train_loader)
    print(perc)
    dice_dict, dice_dict_unc = {uc: [] for uc in uncertaint_threshold}, {uc: [] for uc in uncertaint_threshold}
    for i, (mito_p, org_ent, dist)  in enumerate(it):
        ent = torch.rand_like(dist)
        ent_patches = model.model.patchify(ent)
        dist_patches = model.model.patchify(dist)
        ent_mean_patches = ent_patches.mean(dim=-1)
        dist_mean_patches = dist_patches.mean(dim=-1)
        ent_means = model.model.unpatchify(ent_mean_patches[..., None].repeat((1,1,ent_patches.shape[-1])))

        # perc = 0.99
        with torch.no_grad():
            # preds, masks = model.model.forward_full_pred(img=mito_p, 
            #                                     entropy=ent,
            #                                     dist=dist.clone(),
            #                                     perc_masked=perc,
            #                                     )
            preds, masks = model.model.forward_full_pred(img=mito_p.to('cuda'), 
                                                entropy=ent.to('cuda'),
                                                dist=dist.to('cuda').clone(),
                                                perc_masked=perc,
                                                )

            preds, masks = preds[0], masks[0]
            

            target = mito_p.clone()

            for thres in uncertaint_threshold:
                target_unc = adjust_masks_unc(target.clone(), dist, org_ent, thres, masks)
                preds_unc = adjust_masks_unc(preds.clone(), dist, org_ent, thres, masks)
                dice_dict_unc[thres].append(torchmetrics.functional.dice(preds_unc.to('cuda'), target_unc.long().to('cuda'), 
                                                ignore_index=5, average='micro', mdmc_average='samplewise').cpu().item())

            for thres in uncertaint_threshold:
                target_c = adjust_masks(target.clone(), dist, org_ent, thres, masks)
                preds_c = adjust_masks(preds.clone(), dist, org_ent, thres, masks)
                dice_dict[thres].append(torchmetrics.functional.dice(preds_c.to('cuda'), target_c.long().to('cuda'), 
                                                             ignore_index=5, average='micro', mdmc_average='samplewise').cpu().item())
            # print(dice_arr, dice_arr_unc)

    results[perc] = dice_dict 
    results_uncertainty[perc] = dice_dict_unc



for perc, dic in results.items():
    for k, v in dic.items():
        results[perc][k] = np.mean(v)
        results_uncertainty[perc][k] = np.mean(results_uncertainty[perc][k])
pd.DataFrame(results).to_csv('results_certain_pixel.csv')
pd.DataFrame(results_uncertainty).to_csv('results_uncertain_pixel.csv')