import os
import random

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.config import DATASETS_CONFIG, MODELS_CONFIG
from api.inference import process_batch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mounts the data directory to serve images statically
app.mount("/data", StaticFiles(directory="data"), name="data")

def count_images(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

@app.get("/datasets")
def get_datasets():
    response_data = []

    for key, value in DATASETS_CONFIG.items():
        entry = value.copy()
        entry["name"] = key
        response_data.append(entry)

    return {"datasets": response_data}

@app.get("/models")
def get_models():
    response_data = []

    for key, value in MODELS_CONFIG.items():
        entry = value.copy()
        entry["name"] = key
        response_data.append(entry)

    return {"models": response_data}

@app.get("/random_samples")
def get_random_samples(dataset_name, count):
    if dataset_name not in DATASETS_CONFIG:
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    
    dataset_path = DATASETS_CONFIG[dataset_name]["url"]

    try:
        count = int(count)
    except ValueError:
        raise HTTPException(status_code=400, detail="Count must be a number")    
    
    candidates = []

    classes_found = [
        d for d in os.listdir(dataset_path) 
        if os.path.isdir(os.path.join(dataset_path, d))
    ]

    for class_name in classes_found:
        class_dir = os.path.join(dataset_path, class_name)
        
        # Store tuple (class_name, file)
        files = [(class_name, f) for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        candidates.extend(files)
    
    if not candidates:
        return {"images": []}

    selected_entries = random.sample(candidates, min(count, len(candidates)))
    
    response_data = []
    for class_name, image in selected_entries:
        response_data.append({
            "url": f"/{dataset_path}/{class_name}/{image}",
            "class_name": class_name
        })
    
    return {"images": response_data}

@app.post("/process")
def get_processing_results(body = Body(...)):
    samples = body.get("samples", [])
    models = body.get("models", [])

    results = []

    for model_config in models:
        model_name = model_config["name"]
        
        if model_name not in MODELS_CONFIG:
            continue
            
        result = process_batch(model_config, samples) 
        results.append(result)

    return {"processing_results": results}