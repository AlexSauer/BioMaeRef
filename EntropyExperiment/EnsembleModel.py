import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('your_path/GlobalStaininPrediction')
from full_prediction import load_model_from_config
from data.BasicDataset import BasicDataset
from predict import predict_stack, add_mirrored_slices, remove_mirrored_slices
from utils.Args import Args
from SegProcessing.MajorityVote import gpu_majorityVote

import torch


class ModelEnsemble:
    def __init__(self, model_path):
        self.models = []
        for run in [1, 2, 3, 4, 5, 6]:
            print(f'Loading model for run {run}', flush=True, end='\r')
            current_model_path = (
                os.path.join(model_path, f'Run_{run}', 'args.yaml') if run != 0 else model_path
            )
            self.models.append(load_model_from_config(current_model_path))
        print('Ensemble loaded', flush=True)
        self.args = Args(base_yaml_file=current_model_path, extension_yaml_file=None)

    def raw_predict(
        self,
        data,
        drug_treatment=None,
        patch_size=None,
        stride=None,
    ):
        predictions = []
        prediction_window_size = patch_size if patch_size is not None else self.args.training_size
        prediction_stride = stride if stride is not None else self.args.data_stride
        print(f'Using patch size: {prediction_window_size} with stride {prediction_stride}', flush=True)
        data = add_mirrored_slices(data, prediction_window_size[0] // 2)

        for i, model in enumerate(self.models):
            print(f'Predicting with model {i}', flush=True, end='\r')
            if drug_treatment is not None:
                prediction = predict_stack(
                    data,
                    model,
                    drug_treatment=drug_treatment,
                    img_window=prediction_window_size,
                    stride=prediction_stride,
                )
            else:
                prediction = predict_stack(data, model, img_window=prediction_window_size, stride=prediction_stride)
            prediction = remove_mirrored_slices(prediction, prediction_window_size[0] // 2)
            predictions.append(prediction)

        return predictions

    def extract_data(self, img_path, threshold_file=None, extract_channel=None, nhs_lower_threshold_quantile=None):
        dataset = BasicDataset(
            imgs_dir=[img_path],
            scale=self.args.scale,
            extract_channel=extract_channel,
            config_file_threshold=threshold_file,
            mode='test',
            nhs_lower_threshold_quantile=nhs_lower_threshold_quantile,
        )
        data = dataset.img_list[0]
        if getattr(self.args, 'cat_emb_dim', None) is not None:
            drug_treatment = dataset.drug_list[0]
        else:
            drug_treatment = None

        return data, drug_treatment

    @staticmethod
    def majority_vote(preds):
        preds = [torch.Tensor(ind_pred.astype('float'))[None] for ind_pred in preds]
        preds = torch.cat(preds, dim=0)
        return gpu_majorityVote(preds)

    @staticmethod
    def comp_entropy(preds):
        H = np.zeros(preds[0].shape)
        classes = np.unique(preds[0])

        for k in classes:
            P = np.zeros(preds[0].shape)
            for pred in preds:
                P += pred == k
            P /= len(preds)
            H -= P * np.log(P, out=np.zeros_like(P), where=P > 0)

        return H
