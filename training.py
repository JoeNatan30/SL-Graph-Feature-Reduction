import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from data_loader import keypointFramesDataset
from encoder_decoder_model import LightningModule

num_epochs = 15
num_points = 71
num_features = 2
hidden_features = 128
out_features = 64
learning_rate = 9e-8
batch_size = 16

# DATASET
keypoints_model = 'mediapipe'
dataset_name = 'AEC-DGI156-DGI305'
relative_path = '../ConnectingPoints/split'

general_path = f'{relative_path}/{dataset_name}--{keypoints_model}'

# WandB
wandb = WandbLogger(project="Keypoint-graph-reducer", entity="joenatan30")

# DATALOADER
# Create the dataset from data_loader
full_path = f'{general_path}-Train.hdf5'
dataset_train = keypointFramesDataset(full_path, keypoints_model, num_points)

full_path = f'{general_path}-Val.hdf5'
dataset_val = keypointFramesDataset(full_path, keypoints_model, num_points)

# prepare the pytorch dataLoader
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=4)

# CRITERION
# we chose mean squared error as criterion
criterion = nn.MSELoss()

# MODEL
# initialize the model
model = LightningModule(num_features, num_points, hidden_features, out_features, lr=learning_rate, criterion=criterion)

trainer = pl.Trainer(callbacks=[ModelSummary(max_depth=-1),
                                LearningRateMonitor(logging_interval='epoch'),
                                EarlyStopping(monitor="val_loss_epoch", mode="min")],
                     #profiler="simple",
                     gpus=1,
                     logger = wandb)

trainer.fit(model, dataloader_train, dataloader_val) #ckpt_path="./lightning_logs/version_2/checkpoints/epoch=9-step=6880.ckpt",)
