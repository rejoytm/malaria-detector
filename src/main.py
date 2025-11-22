import argparse
import logging

from src import config
from src.log import setup_logging
from src.train import run_experiment

setup_logging()
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode', 
    type=str, 
    required=True, 
    choices=['train_original'],
    help='Select the operation to run.'
)

args = parser.parse_args()

def main():
    if args.mode == 'train_original':
        logger.info("Training models on original images")
        run_experiment(experiment_name='original', data_dir=config.DATA_DIR)

if __name__ == "__main__":
    main()