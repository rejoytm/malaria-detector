DATASETS_CONFIG = {
    "Cell Images": {
        "url": "data/cell_images",
        "description": "Original Cell Images dataset",
        "total_images_count": 27558
    },
    "Cell Images Norm": {
        "url": "data/cell_images_norm",
        "description": "Stain-normalized Cell Images dataset",
        "total_images_count": 27558
    }
}

MODELS_CONFIG = {
    "SimpleCNN": {
        "url": "outputs/original_saved/models/SimpleCNN.keras",
        "description": "Simple 3-layer CNN",
        "is_starred": False,
        "category": "original",
    },
    "ResNet50": {
        "url": "outputs/original_saved/models/ResNet50.keras",
        "description": "Deep residual model",
        "is_starred": False,
        "category": "original",
    },
    "VGG19": {
        "url": "outputs/original_saved/models/VGG19.keras",
        "description": "Classic feature extractor",
        "is_starred": False,
        "category": "original",
    },
    "EfficientNetV2B2": {
        "url": "outputs/original_saved/models/EfficientNetV2B2.keras",
        "description": "Modern efficient CNN",
        "is_starred": False,
        "category": "original",
    },
    "MobileNetV3Large": {
        "url": "outputs/original_saved/models/MobileNetV3Large.keras",
        "description": "Fast lightweight model",
        "is_starred": False,
        "category": "original",
    },
    "MobileNetV3Small": {
        "url": "outputs/original_saved/models/MobileNetV3Small.keras",
        "description": "Faster lightweight model",
        "is_starred": True,
        "category": "original",
    },  
    "SimpleCNN-Norm": {
        "url": "outputs/normalized_saved/models/SimpleCNN.keras",
        "description": "Simple 3-layer CNN",
        "is_starred": False,
        "category": "normalized",
    },
    "ResNet50-Norm": {
        "url": "outputs/normalized_saved/models/ResNet50.keras",
        "description": "Deep residual model",
        "is_starred": False,
        "category": "normalized",
    },
    "VGG19-Norm": {
        "url": "outputs/normalized_saved/models/VGG19.keras",
        "description": "Classic feature extractor",
        "is_starred": False,
        "category": "normalized",
    },
    "EfficientNetV2B2-Norm": {
        "url": "outputs/normalized_saved/models/EfficientNetV2B2.keras",
        "description": "Modern efficient CNN",
        "is_starred": False,
        "category": "normalized",
    },
    "MobileNetV3Large-Norm": {
        "url": "outputs/normalized_saved/models/MobileNetV3Large.keras",
        "description": "Fast lightweight model",
        "is_starred": False,
        "category": "normalized",
    },
    "MobileNetV3Small-Norm": {
        "url": "outputs/normalized_saved/models/MobileNetV3Small.keras",
        "description": "Faster lightweight model",
        "is_starred": True,
        "category": "normalized",
    },    
    "FusionModel": {
        "url": "outputs/fusion_saved/models/FusionModel.keras",
        "description": "Model combining CNN embeddings",
        "is_starred": True,
        "category": "fusion",
    },
}