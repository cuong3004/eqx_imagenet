import os

import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
from IPython.core.display import display
# from pl_bolts.datamodules import CIFAR10DataModule
# from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
# from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.callbacks.progress import TQDMProgressBar
# from pytorch_lightning.loggers import CSVLogger
# from torch.optim.lr_scheduler import OneCycleLR
# from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
import pathlib
from pathlib import Path
import jax
import jax.random as jr
from jax import numpy as jnp
import equinox as eqx
import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as sharding
from eqxvision.models import mobilenet_v3_small
from jax.lib import xla_bridge
import optax
import numpy as np
from data_module import ImagenetModule
from config import args

print(xla_bridge.get_backend().platform)

import tensorflow as tf 
tf.config.optimizer.set_jit(True)


@eqx.filter_jit
def accuracy(predictions, labels):
    predicted_labels = jnp.argmax(predictions, axis=-1)
    
    correct_predictions = jnp.sum(predicted_labels == labels)
    
    total_samples = labels.shape[0]
    accuracy = correct_predictions / total_samples
    
    return accuracy

@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def loss_fn(
    model, model_state, x, y, key
):
    batch_size = x.shape[0]
    batched_keys = jax.random.split(key, num=batch_size)
    
    pred_y, model_state= jax.vmap(
        model, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None)
    )(x, model_state, batched_keys)
    fn = optax.softmax_cross_entropy_with_integer_labels

    return fn(pred_y, y).mean(), [model_state, pred_y]

@eqx.filter_jit
def make_train_step(
    model,
    model_state,
    x,
    y,
    key,
    opt_state,
    opt_update
):
    
    key, new_key = jax.random.split(key)
    
    (loss_value, [model_state, pred_y]), grads = loss_fn(model, model_state, x, y, key)
    updates, opt_state = opt_update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    
    acc = accuracy(pred_y, y)
    stat_dict = {"train_loss": loss_value, "train_acc":acc}
    return model, model_state, stat_dict, new_key, opt_state

@eqx.filter_jit
def make_valid_step(
    model,
    x,
    y,
):
    logits, _ = jax.vmap(
            model
        )(x)
#     print(logits)
    acc = accuracy(logits, y)
    return {"valid_acc": acc}



def create_model(key):
    keys = jax.random.split(key, 3)
    
    model = mobilenet_v3_small(torch_weights=None, num_classes=10)
#     print(model.features)
    # model.features.layers[0].layers[0] = eqx.nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), use_bias=False, key=keys[0])
    # model.features[1].block.layers[0].layers[0] = eqx.nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), use_bias=False, key=keys[1])
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model.maxpool = nn.Identity()
    return model


class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.key = jax.random.PRNGKey(1)
        self.model_key, self.train_key, _ = jax.random.split(self.key, 3)
        
        self.model = create_model(self.model_key)
        self.model_state = eqx.nn.State(self.model)
        
        num_devices = len(jax.devices())
        devices = mesh_utils.create_device_mesh((num_devices,))
        self.shard = sharding.PositionalSharding(devices)
        
        self.global_step_ = 0
        
        self.configure_optimizers()
        

    def prepare_batch(self, batch):
        # x, y = batch
        # print(type(batch))
        batch["images"] = np.transpose(batch["images"], (0, 3, 1, 2))
        # batch = (x, y)
        # print(batch["images"].shape)
        batch = jax.tree_map(lambda x: jax.device_put(x, 
                                                      self.shard.reshape(
                                                          [len(jax.devices())]+[1]*(len(x.shape)-1))
                                                     ), batch)
        return batch
    
    def training_step(self, batch):
        batch = self.prepare_batch(batch)
        x, y = batch["images"], batch["labels"]
        
        self.model, self.model_state, \
        stat_dict, \
        self.train_key, self.opt_state = make_train_step(self.model, self.model_state, 
                                                            x, y,
                                                            self.train_key,
                                                            self.opt_state,
                                                            self.optim.update
                                                          )
        
        stat_dict = jax.tree_map(lambda x: torch.scalar_tensor(x.item()),stat_dict)
        self.log_dict(stat_dict, prog_bar=True, batch_size=args['batch_size_train'])
        return stat_dict
    
    def validation_step(self, batch, batch_idx):
        batch = self.prepare_batch(batch)
        x, y = batch["images"], batch["labels"]
        
        stat_dict = make_valid_step(self.inference_model, x, y)
        stat_dict = jax.tree_map(lambda x: torch.scalar_tensor(x.item()),stat_dict)
        self.log_dict(stat_dict, prog_bar=True, batch_size=args['batch_size_valid'])
        
    def configure_optimizers(self):
        self.optim = optax.adam(3e-4)
        self.opt_state = self.optim.init(eqx.filter(self.model, eqx.is_inexact_array))
    
    def on_fit_end(self):
        pass

    def on_train_epoch_end(self) -> None:
        pathlib.Path.mkdir(Path(".") / f'checkpoints', parents=True, exist_ok=True)
        with open(Path(".") / f"checkpoints/last.eqx", "wb") as f:
            eqx.tree_serialise_leaves(f, self.model)
            eqx.tree_serialise_leaves(f, self.model_state)

    
    def on_validation_epoch_start(self):
        self.inference_model = eqx.nn.inference_mode(self.model)
        self.inference_model = eqx.Partial(self.inference_model, state=self.model_state, key=self.train_key)
        
        

model = LitResnet(lr=0.05)

from pytorch_lightning.loggers import WandbLogger
# wandb_logger = WandbLogger(project="imagenet", name='eqx') 

from pytorch_lightning.loggers import NeptuneLogger
import sys


neptune_logger = NeptuneLogger(
    project=sys.argv[1],
    api_key=sys.argv[2],
    name="exq"
)

imgset_module = ImagenetModule()

trainer = Trainer(
    max_epochs=30,
    accelerator="cpu",
    devices=None,
    logger=neptune_logger,
    profiler="simple"
    # callbacks=[TQDMProgressBar(refresh_rate=10)],
)

trainer.fit(model, imgset_module)