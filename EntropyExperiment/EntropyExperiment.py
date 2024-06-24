import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import tifffile
import torch
from EnsembleModel import ModelEnsemble
from scipy.ndimage import distance_transform_edt
from skimage.draw import ellipsoid, rectangle


def get_sample(model=None, patch_size=[24, 256, 256]):
    """Adjust this function according to YOUR DATA"""
    id = PARAMETER.get_random_id()
    print(f'Random ID: {id}')
    drug, batch, cell, label = identify_ID(id)
    if cell == 0:
        raise ValueError('Cell 0 not found, skipping...')
    print(f'Drug: {drug}, Batch: {batch}, Cell: {cell}, Label: {label}')
    cur_file_base = loader.file_base(drug, batch, cell)
    param = pd.read_csv(os.path.join(PATH_BASE, loader.dir_batch(batch, drug), cur_file_base + '.csv'))
    param['cristae_perc'] = param['cristae_volume'] / param['volume']
    param['ambig_perc'] = param['ambiguous_volume'] / param['volume']
    pred = loader.img_reader(os.path.join(PATH_BASE_PRED, loader.dir_batch(batch), cur_file_base + '.tif'))
    entropy = loader.img_reader(os.path.join(PATH_BASE_ENTROPY, f'Entropy_DMSO_1st_{cell}.tif'))
    entropy = entropy.astype('float') / 255 * np.log(5)

    if model is None:
        nhs = loader.img_reader(
            os.path.join(PATH_NHS, loader.drug_dir_nhs(batch), loader.find_nhs_file(drug, batch, cell))
        )[:, 2]
    else:
        nhs, drug_info = model.extract_data(
            os.path.join(PATH_NHS, loader.drug_dir_nhs(batch), loader.find_nhs_file(drug, batch, cell)),
            extract_channel=2,
        )

    mitos, _ = loader.get_mitos(drug, batch, cell)
    skeleton, _ = loader.get_skel(drug, batch, cell)

    # Need to create a copy to avoid modifying the original
    mitos = mitos.copy()
    skeleton = skeleton.copy()

    # Find a random point on the skeleton
    skeleton[mitos != label] = 0
    skel_coords = np.where(skeleton > 0)
    idx = np.random.randint(len(skel_coords[0]))
    selected_coord = (skel_coords[0][idx], skel_coords[1][idx], skel_coords[2][idx])

    # Select patch around this point with a size
    bound = [
        (max(0, selected_coord[i] - patch_size[i] // 2), min(skeleton.shape[i], selected_coord[i] + patch_size[i] // 2))
        for i in range(3)
    ]

    nhs_patch = loader.extr_patch(nhs, bound, padding=0)
    pred_patch = loader.extr_patch(pred, bound, padding=0)
    entropy_patch = loader.extr_patch(entropy, bound, padding=0)
    mitos_patch = (loader.extr_patch(mitos, bound, padding=0) == label).astype(int)
    skeleton_patch = loader.extr_patch(skeleton, bound, padding=0)
    skeleton_patch[np.logical_not(mitos_patch)] = 0
    pred_patch[np.logical_not(mitos_patch)] = 0
    mitos_patch += skeleton_patch

    return nhs_patch, pred_patch, mitos, skeleton_patch, entropy_patch, drug_info, id


def generate_modification(nhs, pred=None, type='background'):
    nhs = nhs.clone()
    # Generate a level set about zero of two identical ellipsoids in 3D
    r = np.random.randint(4, 10) * 2
    z, y, x = z, y, x = np.array(nhs.shape) // 2

    m = np.zeros_like(nhs)
    ellip_base = ellipsoid(r // 2, r, r)[:r, : 2 * r, : 2 * r]

    m[z - r // 2 : z + r // 2, y - r : y + r, x - r : x + r] = ellip_base

    d = distance_transform_edt(m)
    d = d / d.max()
    d = torch.Tensor(np.sqrt(d))

    if type == 'background':
        t = 0.8
        mod = d * t * bg[: nhs.shape[0]] + (1 - d * t) * nhs.clone()
    elif type == 'noise':
        t = 0.8
        mod = d * t * (torch.randn(nhs.shape) * nhs.std() + nhs.mean()) + (1 - d * t) * nhs.clone()
    elif type == 'noise_bg':
        t = 0.8
        mean = nhs[pred == 0].mean()
        std = nhs[pred == 0].std()
        mod = d * t * (torch.randn(nhs.shape) * std + mean) + (1 - d * t) * nhs.clone()
        mod = d * t * bg[: nhs.shape[0]] + (1 - d * t) * mod.clone()
    return mod, d > 0


if __name__ == '__main__':
    from utils_MitoInspect import DataLoader, ParaOverview, id2label, identify_ID


    # Sample Mitochondria
    bg = tifffile.imread('your_path/background.tif')
    bg = bg / bg.max()

    PATH_NHS = 'your_path/Originals/'
    PATH_BASE = 'your_path'
    PATH_BASE_PRED = 'your_path/pred'
    PATH_BASE_ENTROPY = 'your_path/entropy'


    ensemble = ModelEnsemble('your_path/ensemble')
    loader = DataLoader(path_base=PATH_BASE, path_nhs=PATH_NHS, path_base_pred='')
    PARAMETER = ParaOverview(PATH_BASE)

    results = []
    for _ in range(100):
        try:
            nhs, pred, _, _, entropy, drug, id = get_sample(model=ensemble)  
            for cur_type in ['background', 'noise_bg']:
                if nhs.shape[1] != 256 or nhs.shape[2] != 256:
                    continue
                mod, mask = generate_modification(nhs.clone(), torch.Tensor(pred), type=cur_type)
                preds_mod = ensemble.raw_predict(mod.float(), drug, patch_size=(24, 256, 256), stride=(12, 256, 226))

                ent_mod = ensemble.comp_entropy(preds_mod)

                print(f'{entropy[mask>0].mean():.4f} ->  {ent_mod[mask>0].mean():.4f}')
                results.append((id, entropy[mask > 0].mean(), ent_mod[mask > 0].mean(), cur_type))
        except ValueError as e:
            print(e)
            continue

        pd.DataFrame(results, columns=['ID', 'Original', 'Modified', 'Modification_Type']).to_csv(
            'your_path/Entropyresults.csv', index=False
        )
