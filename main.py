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
# from torchmetrics.functional import accuracy
import pathlib
from pathlib import Path
import jax
import jax.random as jr
from jax import numpy as jnp
import equinox as eqx
import jax.experimental.mesh_utils as mesh_utils
import jax.sharding as sharding
from eqxvision.models.classification.mobilevitv3 import mobievit_xx_small_v3
# from eqxvision.models import mobilenet_v3_small
from jax.lib import xla_bridge
import optax
import numpy as np
from data_module import ImagenetModule
from config import args

# import dm_pix as pix

print(xla_bridge.get_backend().platform)

import tensorflow as tf 
tf.config.optimizer.set_jit(True)

# input_dtype = jnp.bfloat16
# input_dtype = jnp.float32


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
    # print(len(logits))
    # print(logits.shape)
#     print(logits)
    acc = accuracy(logits, y)
    return {"valid_acc": acc}



def create_model(key):
    keys = jax.random.split(key, 3)
    model = mobievit_xx_small_v3(keys[0], 1000)
    # model = mobilenet_v3_small(torch_weights=None, num_classes=1000)
#     print(model.features)
    # model.features.layers[0].layers[0] = eqx.nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), use_bias=False, key=keys[0])
    # model.features[1].block.layers[0].layers[0] = eqx.nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), use_bias=False, key=keys[1])
    # model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # model.maxpool = nn.Identity()
    return model

# @jax.jit
# def augmentation(key, image):
#     img_size = 224
#     keys = jax.random.split(key, 6)
#     image = jax.image.resize(image, (int(img_size*1.5), int(img_size*1.5), 3), 'linear')
#     image = pix.random_crop(keys[0], image, (img_size,img_size, 3))
#     image = pix.random_flip_left_right(keys[1], image)
#     image = pix.random_brightness(keys[2], image, 0.5)
#     image = pix.random_contrast(keys[3], image, 0.6, 1.4)
#     image = pix.random_saturation(keys[4], image, 0.6, 1.4)
#     image = pix.random_hue(keys[5], image, 0.2)
#     return image
          
# @jax.jit
# def batch_augmentation(key, images):
#     key, newkey = jax.random.split(key)
#     batch_key = jax.random.split(key, images.shape[0])
#     images = jax.vmap(augmentation, in_axes=(0,0))(
#         batch_key, images
#     )
#     return newkey, images

class InitJax:
    def __init__(self):
        self.rng = jax.random.PRNGKey(0)
        self.initializer = jax.nn.initializers.he_normal()

    def random_intit(self, x):
        if jnp.std(x) == 0 or len(x.shape)==1:
            return x
        else:
            self.rng, key  = jax.random.split(self.rng, 2)
            return self.initializer(key, x.shape, jnp.float32)  
            
            


class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.key = jax.random.PRNGKey(1)
        self.model_key, self.train_key, self.data_key = jax.random.split(self.key, 3)
        
        # self.model = create_model(self.model_key)
        init_param_jax = InitJax()
        
        self.model = create_model(self.model_key)
        params, static = eqx.partition(self.model, eqx.is_array)
        params = jax.tree_map(lambda x:init_param_jax.random_intit(x), params)
        # params = jax.tree_map(lambda x:x.astype(input_dtype) if x.dtype!=jnp.bool_ else x, params)
        self.model = eqx.combine(params, static)
        # self.model = model
        self.model_state = eqx.nn.State(self.model)
        
        num_devices = len(jax.devices())
        devices = mesh_utils.create_device_mesh((num_devices,))
        self.shard = sharding.PositionalSharding(devices)
        
        self.global_step_ = 0
        
        self.configure_optimizers()
        

    def prepare_batch(self, batch):
        # x, y = batch
        # print(type(batch))
        
        # self.data_key, batch["images"] = batch_augmentation(self.data_key, batch["images"])
        batch["images"] = np.transpose(batch["images"], (0, 3, 1, 2))
        # batch = (x, y)
        # print(batch["images"].shape)
        batch = jax.tree_map(lambda x: jax.device_put(x, 
                                                      self.shard.reshape(
                                                          [len(jax.devices())]+[1]*(len(x.shape)-1))
                                                     ), batch)
        # batch["images"] = batch["images"].astype(input_dtype)
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
        self.optim = optax.adamw(3e-4)
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
    name="vit_he"
)

imgset_module = ImagenetModule()

trainer = Trainer(
    max_epochs=100,
    accelerator="cpu",
    devices=None,
    logger=neptune_logger,
    profiler="simple"
    # callbacks=[TQDMProgressBar(refresh_rate=10)],
)

trainer.fit(model, imgset_module)