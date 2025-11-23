import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'cell_images')
NORMALIZED_DATA_DIR = DATA_DIR + "_norm"
NORMALIZATION_REFERENCE_IMAGE_PATH = os.path.join(DATA_DIR, 'template.png')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Common Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
LEARNING_RATE = 0.0001
SEED = 42

# Hyperparameter optimization
N_TRIALS = 50

# Model Names
MODEL_NAMES = ['SimpleCNN', 'ResNet50', 'VGG19', 'EfficientNetV2B2', 'MobileNetV3Large']

# Evaluation Paths
EVALUATION_DATA_DIR = NORMALIZED_DATA_DIR
EVALUATION_MODEL_PATHS = {model_name: os.path.join(OUTPUTS_DIR, 'normalized', 'models', f'{model_name}.keras') for model_name in MODEL_NAMES}