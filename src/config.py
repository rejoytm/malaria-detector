import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'cell_images')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Common Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
LEARNING_RATE = 0.0001
SEED = 42

# Model Names
MODEL_NAMES = ['SimpleCNN', 'ResNet50', 'VGG19', 'EfficientNetV2B2']