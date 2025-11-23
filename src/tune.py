import os
import logging
import json

import tensorflow as tf
import optuna
from optuna.integration import TFKerasPruningCallback

from src import config
from src.data_loader import load_dataset_splits

logger = logging.getLogger(__name__)

# Loads a model and tunes the learning rate for the classification head
def objective(trial, train_ds, val_ds, model_name, base_weights_path):    
    # Define search space
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    
    # Load model
    try:
        model = tf.keras.models.load_model(base_weights_path)
        logger.info(f"Trial {trial.number}: Loaded full model successfully from {base_weights_path}")
    except Exception as e:
        logger.error(f"Trial {trial.number}: Failed to load model: {e}")
        return float('inf') 

    # Freeze the base model and unfreeze the head
    for layer in model.layers:
        layer.trainable = False

    NUM_HEAD_LAYERS = 4
    
    for layer in model.layers[-NUM_HEAD_LAYERS:]:
        layer.trainable = True
        logger.info(f"Trial {trial.number}: Unfrozen layer for tuning: {layer.name}")
            
    # Compile model with trial parameters
    logger.info(f"Trial {trial.number}: Compiling with Adam, LR={lr:.6f}.")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    pruning_callback = TFKerasPruningCallback(trial, 'val_loss')

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=config.EARLY_STOP_PATIENCE, 
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.MAX_EPOCHS,
        callbacks=[pruning_callback, early_stop_callback],
        verbose=1
    )
    
    # Return best validation loss
    best_val_loss = min(history.history['val_loss'])
    
    return best_val_loss

def run_tuning(experiment_name, data_dir, model_name, base_weights_path):
    logger.info(f"Starting Hyperparameter Tuning Experiment: {experiment_name}")
    
    # Define and create output directories
    output_base_dir = os.path.join(config.OUTPUTS_DIR, experiment_name)
    os.makedirs(output_base_dir, exist_ok=True)
    logger.info(f"Saving outputs to: {output_base_dir}")

    # Load data
    train_ds, val_ds, test_ds = load_dataset_splits(data_dir)
    
    # Create a wrapper function for the objective that passes fixed arguments
    func = lambda trial: objective(
        trial, 
        train_ds=train_ds, 
        val_ds=val_ds, 
        model_name=model_name, 
        base_weights_path=base_weights_path
    )
    
    # Create study to minimize the validation loss
    study = optuna.create_study(
        direction="minimize", 
        study_name=experiment_name,
        # Using MedianPruner to stop non-promising trials
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
    )
    
    # Run optimization
    logger.info(f"Starting optimization with {config.N_TRIALS} trials.")
    try:
        study.optimize(func, n_trials=config.N_TRIALS)
    except Exception as e:
        logger.error(f"Optuna optimization failed: {e}")
        return
    
    # Log best trial
    logger.info(f"Best Trial Number: {study.best_trial.number}")
    logger.info(f"Best Validation Loss: {study.best_value:.4f}")
    logger.info(f"Best Parameters: {study.best_params}")
    
    # Save best hyperparameters
    best_params_path = os.path.join(output_base_dir, experiment_name, 'best_params.json')
    try:
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        logger.info(f"Best parameters saved to {best_params_path}.")
    except Exception as e:
        logger.error(f"Failed to save best parameters: {e}")
    
    logger.info(f"Experiment {experiment_name} finished.")