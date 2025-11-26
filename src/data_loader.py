import logging

import tensorflow as tf
from keras import layers

from src import config

logger = logging.getLogger(__name__)

def load_full_dataset(data_path=config.DATA_DIR):
    logger.info(f"Loading full data from {data_path}")
    
    full_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode='binary'
    )
    
    logger.info(f"Classes found: {full_ds.class_names}")

    # Shuffle the entire dataset once
    return full_ds.shuffle(buffer_size=1024, seed=config.SEED)

def calculate_split_sizes(full_ds):
    ds_size = tf.data.experimental.cardinality(full_ds).numpy()
    batch_size = config.BATCH_SIZE

    train_ratio = 0.80
    val_ratio = 0.10
    
    train_batches = int(train_ratio * ds_size)
    val_batches = int(val_ratio * ds_size)
    test_batches = ds_size - train_batches - val_batches 

    # Calculate approximate image counts
    total_images = ds_size * batch_size 
    train_images_approx = train_batches * batch_size
    val_images_approx = val_batches * batch_size
    test_images_approx = test_batches * batch_size
    
    logger.info(f"Total images (approx): {total_images}")
    logger.info(f"Train: {train_batches} batches ({train_images_approx} images)")
    logger.info(f"Validation: {val_batches} batches ({val_images_approx} images)")
    logger.info(f"Test: {test_batches} batches ({test_images_approx} images)")
    
    return train_batches, val_batches, test_batches

def load_dataset_splits(data_path=config.DATA_DIR):
    full_ds = load_full_dataset(data_path)
    train_size, val_size, test_size = calculate_split_sizes(full_ds)
    
    train_ds = full_ds.take(train_size)

    val_test_ds = full_ds.skip(train_size)
    val_ds = val_test_ds.take(val_size)

    test_ds = val_test_ds.skip(val_size)
    
    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    logger.info("Dataset loading and optimization complete.")
    
    return train_ds, val_ds, test_ds

def load_dataset_splits_dual(original_data_path, normalized_data_path):
    full_ds_orig = load_full_dataset(original_data_path)
    full_ds_norm = load_full_dataset(normalized_data_path) 
    
    # Calculate sizes based on one dataset
    train_size, val_size, test_size = calculate_split_sizes(full_ds_orig)
    
    # Zip the two datasets together
    full_ds_dual = tf.data.Dataset.zip((full_ds_orig, full_ds_norm))

    def map_dual_input(orig_batch, norm_batch):
        # We assume labels are identical, take the original label
        return (orig_batch[0], norm_batch[0]), orig_batch[1]
    
    full_ds_dual = full_ds_dual.map(map_dual_input)

    # Split the zipped dataset
    train_ds = full_ds_dual.take(train_size)
    val_test_ds = full_ds_dual.skip(train_size)
    val_ds = val_test_ds.take(val_size)
    test_ds = val_test_ds.skip(val_size)
    
    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)    

    logger.info("Dual dataset loading and optimization complete.")
    
    return train_ds, val_ds, test_ds

def get_data_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])