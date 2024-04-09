from dataclasses import field
import tensorflow as tf
from jax import numpy as jnp
import jax
from config import args
from typing import Iterable
from pytorch_lightning import LightningDataModule
from augmentation import TrainRandomAugmentor, ValidRandomAugmentor
import tensorflow_datasets as tfds

# if args["input_dtype"] == "bfloat16":
#     from tensorflow.keras import mixed_precision
#     policy = mixed_precision.Policy('mixed_bfloat16')
#     mixed_precision.set_global_policy(policy)


class ImagenetModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        
        self.train_filenames = []
        for train_dir in args['train_dirs']:
            self.train_filenames.extend(tf.io.gfile.glob(train_dir+'/*.tfrec*'))
            
        self.valid_filenames = []
        for valid_dir in args['valid_dirs']:
            self.valid_filenames.extend(tf.io.gfile.glob(valid_dir+'/*.tfrec*'))
            
        self.train_augmentor = TrainRandomAugmentor(224)
        self.valid_augmentor = ValidRandomAugmentor(224)

    
    def read_tfrecord(self, example):
        # Note how we are defining the example structure here
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'class': tf.io.FixedLenFeature([], tf.int64),
            'filename': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, features)
        image = tf.image.decode_jpeg(example['image'])
        # image = image.
        class_num = example['class']
        # filename = example['filename']
        return {"images":image, "labels":class_num}
    

    # def to_jax(self, sample):
    #     sample['images'] = jnp.array(sample['images'], dtype=jnp.float32)
    #     sample['images'] = jnp.transpose(sample['images'], (0, 3, 1, 2))
    #     sample['labels'] = jnp.array(sample['labels'], dtype=jnp.float32)
    #     return sample['images'], sample['labels']
    
    
    def load_dataset(self, filenames, train=True):
        # assert type(train) == bool
        AUTO = tf.data.AUTOTUNE  
        
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False 
        
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO)
        dataset = dataset.with_options(ignore_order)
        
        dataset = dataset.map(self.read_tfrecord, num_parallel_calls=AUTO)
        
        if train:
            preprocesing = lambda sample: {"images":self.train_augmentor(sample["images"]).astype(args["input_dtype"]), 
                                           "labels":sample["labels"]}
            dataset = dataset.map(preprocesing, num_parallel_calls=AUTO)
            dataset = dataset.repeat()
            dataset = dataset.shuffle(2048)
            dataset = dataset.batch(args["batch_size_train"])
        else:
            preprocesing = lambda sample: {"images":self.valid_augmentor(sample["images"]).astype(args["input_dtype"]), 
                                           "labels":sample["labels"]}
            dataset = dataset.map(preprocesing, num_parallel_calls=AUTO)
            dataset = dataset.repeat()
            dataset = dataset.batch(args["batch_size_valid"])
        
        dataset = dataset.prefetch(AUTO)
    
        return tfds.as_numpy(dataset)
    
    def train_dataloader(self):
        datasetset = self.load_dataset(self.train_filenames, train=True)
        dataloader = DataIterator(iter(datasetset), args["train_step_epoch"])
        return dataloader

    def val_dataloader(self):
        datasetset = self.load_dataset(self.valid_filenames, train=False)
        dataloader = DataIterator(iter(datasetset), args["valid_step_epoch"])
        return dataloader

        
class DataIterator:
    def __init__(self, my_iter: Iterable, len_data: int):
        self.my_iter = my_iter
        self.index = 0
        self.len_my_iter = len_data

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.len_my_iter:
            result = next(self.my_iter)
            self.index += 1
            return result
        else:
            self.index = 0
            raise StopIteration
    
    def __len__(self):
        return self.len_my_iter