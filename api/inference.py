import os

import cv2
import numpy as np
import tensorflow as tf

from src.config import IMG_SIZE
from api.config import DATASETS_CONFIG

loaded_models = {}

# Given an image path (which could be original or normalized), returns a tuple containing (original_path, normalized_path)
def get_dual_image_path(input_path):
    ORIGINAL_URL = DATASETS_CONFIG["Cell Images"]["url"]
    NORMALIZED_URL = DATASETS_CONFIG["Cell Images Norm"]["url"]
    
    original_path = ""
    normalized_path = ""
    
    # Determine the type of the input path
    if NORMALIZED_URL in input_path:
        normalized_path = input_path
        original_path = input_path.replace(NORMALIZED_URL, ORIGINAL_URL)
    elif ORIGINAL_URL in input_path:
        original_path = input_path
        normalized_path = input_path.replace(ORIGINAL_URL, NORMALIZED_URL)
    else:
        original_path = input_path
        normalized_path = input_path
        
    return original_path, normalized_path

def load_model_instance(model_name, model_path):
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    try:
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
            
        print(f"Loading model: {model_name}...")
        model = tf.keras.models.load_model(model_path)
        loaded_models[model_name] = model
        print(f"Model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load model '{model_name}': {e}")
        return None

def predict_single(model, img_path):
    img = cv2.imread(img_path)
    
    if img is None:
        return None, 0.0

    # Preprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img_array = np.asarray(img, dtype=np.float32)

    # Add batch dimension
    input_tensor = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(input_tensor, verbose=0)
    raw_score = prediction[0][0] 
    
    if raw_score > 0.5:
        pred_class = "Uninfected"
        confidence = raw_score
    else:
        pred_class = "Parasitized"
        confidence = 1.0 - raw_score
    
    return pred_class, float(confidence)

def predict_single_fusion(model, input_img_path):
    original_image_path, normalized_image_path = get_dual_image_path(input_img_path)

    # Load and preprocess original image
    img_orig = cv2.imread(original_image_path)
    if img_orig is None:
        return None, 0.0

    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_orig = cv2.resize(img_orig, IMG_SIZE)
    input_orig = np.expand_dims(np.asarray(img_orig, dtype=np.float32), axis=0)
    
    # Load and preprocess normalized image
    img_norm = cv2.imread(normalized_image_path)
    if img_norm is None:
        return None, 0.0

    img_norm = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
    img_norm = cv2.resize(img_norm, IMG_SIZE)
    input_norm = np.expand_dims(np.asarray(img_norm, dtype=np.float32), axis=0)

    # Predict using the dual input
    prediction = model.predict([input_orig, input_norm], verbose=0)
    raw_score = prediction[0][0] 
    
    if raw_score > 0.5:
        pred_class = "Uninfected"
        confidence = raw_score
    else:
        pred_class = "Parasitized"
        confidence = 1.0 - raw_score
    
    return pred_class, float(confidence)

def process_batch(model_config, samples):
    model = load_model_instance(model_config["name"], model_config["url"])
    
    if model is None:
        return {
            "model": model_config, 
            "processed_samples": [],
            "metrics": {
                "accuracy": 0,
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0
           }
        }

    processed_samples = []
    
    # Initialize confusion matrix counters
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    total_count = 0

    for sample in samples:
        image_path = sample.get("url", "").lstrip("/")
        ground_truth = sample.get("class_name")

        if model_config["category"] == "fusion":
            pred_class, confidence = predict_single_fusion(model, image_path)
        else:
            original_image_path, normalized_image_path = get_dual_image_path(image_path)
            if model_config["category"] == "normalized":
                pred_class, confidence = predict_single(model, normalized_image_path)
            else:
                pred_class, confidence = predict_single(model, original_image_path)                       

        if pred_class is None:
            continue

        total_count += 1
        is_correct = (pred_class == ground_truth)

        if ground_truth == "Parasitized":
            if pred_class == "Parasitized":
                tp += 1
            else:
                fn += 1
        else:
            if pred_class == "Uninfected":
                tn += 1
            else:
                fp += 1

        processed_samples.append({
            "url": sample.get("url"),
            "class_name": ground_truth,
            "prediction": {
                "class_name": pred_class,
                "confidence": round(confidence, 4),
                "is_correct": is_correct
            }
        })
    
    # Avoid division by zero
    accuracy = (tp + tn) / total_count if total_count > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "model": model_config, 
        "processed_samples": processed_samples,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4)
        }
    }