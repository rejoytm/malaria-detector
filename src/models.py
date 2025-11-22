import tensorflow as tf
from keras import layers, models, Input
from keras.applications import ResNet50, VGG19, EfficientNetV2B2

from src import config
from src.data_loader import get_data_augmentation

def build_model(model_name):
    inputs = Input(shape=config.IMG_SIZE + (3,))
    x = get_data_augmentation()(inputs)
    
    if model_name == 'SimpleCNN':
        x = layers.Rescaling(1./255)(x)
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        return models.Model(inputs, outputs, name="SimpleCNN")
    elif model_name == 'ResNet50':
        x = tf.keras.applications.resnet50.preprocess_input(x)
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'VGG19':
        x = tf.keras.applications.vgg19.preprocess_input(x)
        base_model = VGG19(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'EfficientNetV2B2':
        x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
        base_model = EfficientNetV2B2(weights='imagenet', include_top=False, input_tensor=x)
    
    base_model.trainable = False # Freeze layers
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return models.Model(inputs=base_model.input, outputs=outputs, name=model_name)