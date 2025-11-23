import argparse
import logging

from src import config
from src.log import setup_logging
from src.train import run_experiment
from src.data_normalizer import run_normalization

setup_logging()
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode', 
    type=str, 
    required=True, 
    choices=['train_original', 'normalize', 'train_normalized'],
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

if __name__ == "__main__":
    main()