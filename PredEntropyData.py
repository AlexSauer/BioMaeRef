import torch
import numpy as np
import tifffile
import os


class Data(torch.utils.data.Dataset):
    def __init__(self, data):
        # Data contains a list of tensors like pred, entropy, dist, nhs and possibility additional antibodies
        self.data = [d.float() for d in data]
        assert len(set([d.shape[0] for d in self.data])) == 1, 'All data should have the same length'

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return [d[idx][None] for d in self.data]


def generate_patches(base, ind, size, stride):
    base_patches = torch.from_numpy(base[ind]).unfold(0, size, stride).unfold(1, size, stride).reshape(-1, size, size)
    base_patches_shift_r = (
        torch.from_numpy(base[ind, size // 2 :]).unfold(0, size, stride).unfold(1, size, stride).reshape(-1, size, size)
    ) 
    base_patches_shift_d = (
        torch.from_numpy(base[ind, :, size // 2 :])
        .unfold(0, size, stride)
        .unfold(1, size, stride)
        .reshape(-1, size, size)
    )  
    base_patches = torch.cat([base_patches, base_patches_shift_r, base_patches_shift_d], dim=0)
    return base_patches


def build_pred_data(input_files=None, size=80, stride=80, mito_perc_thresh=0.5, nhs_files=[], antibodies=False):
    assert input_files is not None, 'Input files not provided'
    path_base_nhs = 'your_path'
    path_base_pred = 'your_path/pred'
    path_base_entrpy = 'your_path/entropy'
    path_base_dist = 'your_path/distance_maps'

    all_dist, all_pred, all_entropy, all_nhs = [], [], [], []
    all_tom20, all_mt = [], []

    for i, (pred_file, entropy_file) in enumerate(input_files):
        if nhs_files:
            org_file = tifffile.imread(os.path.join(path_base_nhs, nhs_files[i]))
            nhs = org_file[:, 2].astype('float')
            nhs = np.clip(nhs, 0, np.quantile(nhs.flatten(), 0.99))
            cmin, cmax = nhs.min(), nhs.max()
            nhs = (nhs - cmin) / (cmax - cmin)

            if antibodies:
                mt = org_file[:, 1].astype('float')
                tom20 = org_file[:, 0].astype('float')

        # Slice selection (we don't go to deep or to shallow as the antibody distribution becomes different due to penetration)
        pred = tifffile.imread(os.path.join(path_base_pred, pred_file))
        entropy = tifffile.imread(os.path.join(path_base_entrpy, entropy_file))
        dist = tifffile.imread(os.path.join(path_base_dist, entropy_file.replace('Entropy', 'Dist')))

        for ind in range(10, pred.shape[0] - 10):
            pred_patches = generate_patches(pred, ind, size, stride)
            entropy_patches = generate_patches(entropy, ind, size, stride)
            dist_patches = generate_patches(dist, ind, size, stride)

            if nhs_files:
                nhs_patches = generate_patches(nhs, ind, size, stride)

                if antibodies:
                    mt_patches = generate_patches(mt, ind, size, stride)
                    tom20_patches = generate_patches(tom20, ind, size, stride)

            cur_mask = (pred_patches > 0).float().mean(dim=(1, 2)) > mito_perc_thresh
            entropy_patches = entropy_patches[cur_mask]
            dist_patches = dist_patches[cur_mask]
            pred_patches = pred_patches[cur_mask]
            if nhs_files:
                nhs_patches = nhs_patches[cur_mask]

                if antibodies:
                    mt_patches = mt_patches[cur_mask]
                    tom20_patches = tom20_patches[cur_mask]

            all_pred.append(pred_patches)
            all_entropy.append(entropy_patches)
            all_dist.append(dist_patches)
            if nhs_files:
                all_nhs.append(nhs_patches)

                if antibodies:
                    all_mt.append(mt_patches)
                    all_tom20.append(tom20_patches)

    pred_patches = torch.cat(all_pred, dim=0)
    entropy_patches = torch.cat(all_entropy, dim=0)
    dist_patches = torch.cat(all_dist, dim=0)
    if nhs_files:
        nhs_patches = torch.cat(all_nhs, dim=0)

        if antibodies:
            mt_patches = torch.cat(all_mt, dim=0)
            tom20_patches = torch.cat(all_tom20, dim=0)

    # Normalise entropy into [0,1]
    entropy_patches = entropy_patches / 255.0

    if nhs_files:
        if antibodies:
            data = Data([pred_patches, entropy_patches, dist_patches, nhs_patches, mt_patches, tom20_patches])
        else:
            data = Data([pred_patches, entropy_patches, dist_patches, nhs_patches])
    else:
        data = Data([pred_patches, entropy_patches, dist_patches])
    return data
