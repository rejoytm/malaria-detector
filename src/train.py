import os
import logging
import json

import tensorflow as tf

from src import config
from src.data_loader import load_dataset_splits, load_dataset_splits_dual
from src.models import build_model, build_fusion_model

logger = logging.getLogger(__name__)

# Generic function to train all models on a specific dataset
def run_experiment(experiment_name, data_dir):
    logger.info(f"Starting Experiment: {experiment_name}")
    logger.info(f"Reading data from: {data_dir}")
    
    # Define and create output directories
    output_base_dir = os.path.join(config.OUTPUTS_DIR, experiment_name)
    output_models_dir = os.path.join(output_base_dir, "models")
    os.makedirs(output_models_dir, exist_ok=True)
    logger.info(f"Saving outputs to: {output_base_dir}")

    # Load Data
    train_ds, val_ds, test_ds = load_dataset_splits(data_dir)
    
    history_dict = {}

    for name in config.MODEL_NAMES:
        logger.info(f"Starting training for {name}")
        model = build_model(name)
        
        logger.info(f"Compiling with Adam, LR={config.LEARNING_RATE}, Max Epochs={config.MAX_EPOCHS}.")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        fit_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.MAX_EPOCHS,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.EARLY_STOP_PATIENCE, restore_best_weights=True)],
            verbose=1
        ).history
        
        # Save Model
        model_path = os.path.join(output_models_dir, f"{name}.keras")
        model.save(model_path)
        
        # Evaluate on the test set
        logger.info(f"Evaluating {name} on the test set.")
        loss, acc = model.evaluate(test_ds, verbose=0)
        
        best_val_acc = max(fit_history.get('val_accuracy', [0]))
        logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
        logger.info(f"Test Accuracy: {acc:.4f}")
        
        # Store history
        history_dict[name] = {
            **fit_history,
            "test_accuracy": acc,
            "test_loss": loss
        }

    history_path = os.path.join(output_base_dir, 'history.json')
    
    try:
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4) 
        logger.info(f"Experiment history saved to {history_path}.")
    except Exception as e:
        logger.error(f"Failed to save experiment history to {history_path}: {e}")    

    logger.info(f"Experiment {experiment_name} finished.")

def run_fusion_experiment(experiment_name, orig_data_dir, norm_data_dir, model_A_name, model_B_name, weights_A_path, weights_B_path):
    logger.info(f"Starting Fusion Experiment: {experiment_name}")
    
    output_base_dir = os.path.join(config.OUTPUTS_DIR, experiment_name)
    os.makedirs(output_base_dir, exist_ok=True)
    logger.info(f"Saving outputs to: {output_base_dir}")

    train_ds, val_ds, test_ds = load_dataset_splits_dual(orig_data_dir, norm_data_dir)
    
    model = build_fusion_model(model_A_name, weights_A_path, model_B_name, weights_B_path)
    
    logger.info(f"Compiling Fusion Model with Adam, LR={config.LEARNING_RATE}, Max Epochs={config.MAX_EPOCHS}.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    fit_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.MAX_EPOCHS,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.EARLY_STOP_PATIENCE, restore_best_weights=True)],
        verbose=1
    ).history
    
    model_path = os.path.join(output_base_dir, "models", f"{experiment_name}.keras")
    model.save(model_path)
    
    logger.info(f"Evaluating {experiment_name} on the test set.")
    loss, acc = model.evaluate(test_ds, verbose=0)
    
    best_val_acc = max(fit_history.get('val_accuracy', [0]))
    logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"Test Accuracy: {acc:.4f}")

    history_dict = {}

    history_dict[experiment_name] = {
        **fit_history,
        "test_accuracy": acc,
        "test_loss": loss
    }

    history_path = os.path.join(output_base_dir, 'history.json')
    
    try:
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
        logger.info(f"Fusion experiment history saved to {history_path}.")
    except Exception as e:
        logger.error(f"Failed to save fusion experiment history to {history_path}: {e}")

    logger.info(f"Fusion experiment {experiment_name} finished.")