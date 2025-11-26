import argparse
import logging

from src import config
from src.log import setup_logging
from src.train import run_experiment, run_fusion_experiment
from src.data_normalizer import run_normalization
from src.tune import run_tuning

setup_logging()
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode', 
    type=str, 
    required=True, 
    choices=['train_original', 'normalize', 'train_normalized', 'tune', 'train_fusion'],
    help='Select the operation to run.'
)

args = parser.parse_args()

def main():
    if args.mode == 'train_original':
        logger.info("Training models on original images")
        run_experiment(experiment_name='original', data_dir=config.DATA_DIR)
    elif args.mode == 'normalize':
        run_normalization(source_dir=config.DATA_DIR, target_dir=config.NORMALIZED_DATA_DIR)
    elif args.mode == 'train_normalized':
        logger.info("Training models on normalized images")
        run_experiment(experiment_name='normalized', data_dir=config.NORMALIZED_DATA_DIR)
    elif args.mode == 'tune':
        run_tuning(experiment_name='tune_efficientnetv2b2', data_dir=config.NORMALIZED_DATA_DIR, model_name='EfficientNetV2B2', base_weights_path='outputs/normalized/models/EfficientNetV2B2.keras')        
    elif args.mode == 'train_fusion':
        run_fusion_experiment(
            experiment_name='mobilenetv3small_simplecnn_fusion',
            orig_data_dir=config.DATA_DIR,
            norm_data_dir=config.NORMALIZED_DATA_DIR,
            model_A_name='MobileNetV3Small',
            model_B_name='SimpleCNN',
            weights_A_path=('outputs/original_saved/models/MobileNetV3Small.keras'),
            weights_B_path=('outputs/normalized_saved/models/SimpleCNN.keras'),
        )

if __name__ == "__main__":
    main()