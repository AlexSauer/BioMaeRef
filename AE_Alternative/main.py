import argparse
import lightning.pytorch as pl
import os
import sys
import torch
import yaml
from PredEntropyData import build_pred_data
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging, LearningRateMonitor
from lightning.pytorch.callbacks import TQDMProgressBar
from model_AE import AE_Module
from torch.utils.data import DataLoader

pl.seed_everything(123456)


def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as file:
        data_dict = yaml.safe_load(file)
    return data_dict


parser = argparse.ArgumentParser(description='Run Experiment')
parser.add_argument('config', type=str, help='Config File')
args = parser.parse_args()
cfg = read_yaml(args.config)

# Load data
data = build_pred_data(
    cfg['DATA']['training_files'],
    cfg['MODEL']['img_size'],
    cfg['MODEL']['img_stride'],
    mito_perc_thresh=cfg['DATA']['mito_perc_thresh'],
    type='float',
)
train_loader = DataLoader(data, batch_size=cfg['TRAINING']['batch_size'], shuffle=True, num_workers=8)

# Load model
model = AE_Module(
    depth=cfg['MODEL']['depth'],
    base_channel_size=cfg['MODEL']['base_channel_size'],
    entropy_loss_threshold=cfg['MODEL']['entropy_loss_threshold'],
    classes=5,
    weights=cfg['MODEL']['weights'],
    lr=cfg['TRAINING']['lr'],
    double_int=cfg['MODEL']['double_int'],
)
model = torch.compile(model)

logger = pl.loggers.TensorBoardLogger(cfg['TRAINING']['result_path'], name=cfg['TRAINING']['experiment_name'])
early_stop = EarlyStopping(monitor='train_loss', mode='min', patience=50, check_finite=True)
trainer = pl.Trainer(
    max_epochs=cfg['TRAINING']['num_epochs'],
    accelerator='gpu',
    devices=[0],
    logger=logger,
    callbacks=[
        ModelCheckpoint(every_n_epochs=25, save_top_k=-1),
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(refresh_rate=5),
    ],
)
trainer.fit(model, train_loader)
