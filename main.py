import lightning.pytorch as pl
import os
import torch
import yaml
from PredEntropyData import build_pred_data
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging, LearningRateMonitor
from lightning.pytorch.callbacks import TQDMProgressBar
from model import MAE_Module
from torch.utils.data import DataLoader
pl.seed_everything(123456)

def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as file:
        data_dict = yaml.safe_load(file)
    return data_dict

cfg = read_yaml('config.yaml')

# Load data
data = build_pred_data(
    cfg['DATA']['training_files'],
    cfg['MODEL']['img_size'],
    cfg['MODEL']['img_stride'],
    mito_perc_thresh=cfg['DATA']['mito_perc_thresh'],
)
train_loader = DataLoader(data, batch_size=cfg['TRAINING']['batch_size'], shuffle=True, num_workers=8)

# Load model
model = MAE_Module(cfg)
model = torch.compile(model)

# Training
logger = pl.loggers.TensorBoardLogger(cfg['TRAINING']['result_path'], name=cfg['TRAINING']['experiment_name'])
early_stop = EarlyStopping(monitor='train_loss', mode='min', patience=50, check_finite=True)
trainer = pl.Trainer(
    max_epochs=cfg['TRAINING']['num_epochs'],
    accelerator='gpu',
    devices=[0],
    logger=logger,
    callbacks=[
        ModelCheckpoint(every_n_epochs=50, save_top_k=-1),
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(refresh_rate=50),
    ],
)
trainer.fit(model, train_loader)
