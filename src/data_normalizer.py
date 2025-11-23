import os
import cv2
import logging
import shutil

from staintools import ReinhardColorNormalizer

from src import config

logger = logging.getLogger(__name__)

def find_image_files(base_dir):
    image_paths = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def normalize_image(normalizer, source_path, target_path):
    try:
        img = cv2.imread(source_path)
        if img is None:
            logger.warning(f"Skipping corrupt file: {source_path}")
            return False
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        normalized_img = normalizer.transform(img)
        normalized_img_bgr = cv2.cvtColor(normalized_img, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(target_path, normalized_img_bgr)
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {source_path}: {e}")
        return False

# Performs stain normalization across the full dataset
def run_normalization(source_dir, target_dir):
    if os.path.exists(target_dir):
        logger.warning(f"Removing existing directory: {target_dir}")
        shutil.rmtree(target_dir)
    
    logger.info(f"Starting stain normalization on {source_dir}")
    
    # Prepare target directory
    for class_folder in os.listdir(source_dir):
        source_path = os.path.join(source_dir, class_folder)
        if os.path.isdir(source_path):
            os.makedirs(os.path.join(target_dir, class_folder), exist_ok=True)
            logger.info(f"Target sub-directory created: {class_folder}")
    
    # Initialize normalizer
    if not os.path.exists(config.NORMALIZATION_REFERENCE_IMAGE_PATH):
        logger.error(f"Reference image not found at {config.NORMALIZATION_REFERENCE_IMAGE_PATH}. Cannot normalize.")
        logger.error("Please place a representative image at this path.")
        return False
        
    reference_img = cv2.imread(config.NORMALIZATION_REFERENCE_IMAGE_PATH)
    if reference_img is None:
        logger.error("Reference image failed to load.")
        return False
    reference_img_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
    
    normalizer = ReinhardColorNormalizer()
    normalizer.fit(reference_img_rgb)
    logger.info("Reinhard Color Normalizer fit to reference image.")

    all_image_paths = find_image_files(source_dir)
    
    # Filter out the reference image itself to avoid processing it
    all_image_paths = [p for p in all_image_paths if os.path.abspath(p) != os.path.abspath(config.NORMALIZATION_REFERENCE_IMAGE_PATH)]
    
    logger.info(f"Starting processing of {len(all_image_paths)} images.")

    for i, source_path in enumerate(all_image_paths):
        relative_path = os.path.relpath(source_path, source_dir)
        target_path = os.path.join(target_dir, relative_path)
        
        normalize_image(normalizer, source_path, target_path)
        
        if i > 0 and i % 5000 == 0:
            logger.info(f"Processed {i}/{len(all_image_paths)} images.")

    logger.info("Stain normalization process complete.")
    return True