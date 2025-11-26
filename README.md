# Malaria Detector

A deep learning research project focused on robust malaria parasite detection in blood smears utilizing Transfer Learning, Hyperparameter Tuning, Stain Normalization, and Feature Fusion to maximize accuracy and generalization. The project features a full-stack implementation, including a [FastAPI](https://fastapi.tiangolo.com/) backend and a [SvelteKit](https://svelte.dev/) frontend, which powers a Model Playground.

https://github.com/user-attachments/assets/4113945d-4be8-4b96-8f2d-f7f25dc1ae1f

## 🔬 Methodology

### 1. Dataset

Utilized the [NIH Malaria Cell Images dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) (27,558 images), balanced between Parasitized and Uninfected classes.

### 2. Stain Normalization

To improve generalization and reduce variability across different laboratory conditions, Reinhard Color Normalization was applied to the original images. This technique uses a reference image to correct variations in lighting and staining as illustrated in the figure below.

![Stain Normalization](https://github.com/user-attachments/assets/71a3fccb-1e9f-4ef0-b5a1-7241060096f1)

### 3. Feature Fusion

The final deployment model utilizes a Feature Fusion architecture:

- **Branch A**: MobileNetV3Small trained on original images (captures general texture/color features).
- **Branch B**: SimpleCNN trained on stain normalized images (captures structural features without color bias).

The fused representation achieves 96.34% accuracy, outperforming individual baselines.

## 🛠 Tech Stack

- **Deep Learning (TensorFlow/Keras)**: Used for training and evaluating all 13 models, including custom CNN and transfer learning architectures.
- **Backend (FastAPI)**: Serves dataset metadata, images, and runs model inference.
- **Frontend (SvelteKit)**: Provides an interactive Model Playground GUI for real-time benchmarking, allowing selection of any model, randomized testing, and detailed metrics visualization.

## 📊 Performance Benchmarks

The project benchmarked 6 architectures across two conditions (Original vs. Normalized) and one Fusion model. Hyperparameters were tuned using [Optuna](https://optuna.org/).

### Models Trained on Original Images

| Model Architecture | Accuracy | F1-Score | Precision | Recall |
| :----------------- | :------: | :------: | :-------: | :----: |
| MobileNetV3Small   |  95.29%  |  0.9504  |  0.9629   | 0.9381 |
| MobileNetV3Large   |  94.86%  |  0.9483  |  0.9565   | 0.9403 |
| EfficientNetV2B2   |  93.93%  |  0.9395  |  0.9409   | 0.9381 |
| VGG19              |  93.25%  |  0.9302  |  0.9529   | 0.9086 |
| ResNet50           |  92.82%  |  0.9292  |  0.9444   | 0.9145 |
| SimpleCNN          |  80.39%  |  0.7992  |  0.8207   | 0.7789 |

### Models Trained on Stain Normalized Images

| Model Architecture | Accuracy | F1-Score | Precision | Recall |
| :----------------- | :------: | :------: | :-------: | :----: |
| MobileNetV3Large   |  94.90%  |  0.9463  |  0.9516   | 0.9410 |
| MobileNetV3Small   |  94.90%  |  0.9461  |  0.9543   | 0.9381 |
| EfficientNetV2B2   |  93.75%  |  0.9395  |  0.9530   | 0.9263 |
| SimpleCNN          |  93.68%  |  0.9348  |  0.9684   | 0.9035 |
| ResNet50           |  93.03%  |  0.9285  |  0.9437   | 0.9138 |
| VGG19              |  92.17%  |  0.9230  |  0.9317   | 0.9145 |

> **Key Insight:** While complex architectures like ResNet and VGG maintained similar performance due to their ability to generalize across stain variations in original images, SimpleCNN achieved a significant 13% accuracy boost post-normalization, showcasing the efficacy of preprocessing for lightweight models.

### Feature Fusion Model

| Model Architecture           | Accuracy | F1-Score | Precision | Recall |
| :--------------------------- | :------: | :------: | :-------: | :----: |
| MobileNetV3Small + SimpleCNN |  96.34%  |  0.9505  |  0.9602   | 0.9410 |

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js & npm

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/rejoytm/malaria-detector.git
   cd malaria-detector
   ```

2. Create and activate a virtual environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Download Data & Model Weights:
   Download `data.zip` and `outputs.zip` from the following link: [Resources (Google Drive)](https://drive.google.com/drive/folders/1hPB4P2rAQF7CwTnFTclA3XM0EOrsfn7v). Extract both zip files into the project root directory. Your file structure should look like this:

   ```plaintext
   malaria-detector/
   ├── data/                  # Extracted from data.zip
   │   ├── cell_images/
   │   │   ├── Parasitized/
   │   │   ├── Uninfected/
   │   │   └── template.png
   │   └── cell_images_norm/
   │       ├── Parasitized/
   │       └── Uninfected/
   ├── outputs/               # Extracted from outputs.zip
   │   ├── original_saved/
   │   │   └── models/
   │   │       ├── SimpleCNN.keras
   │   │       ├── ResNet50.keras
   │   │       └── ...
   │   ├── normalized_saved/
   │   │   └── models/
   │   │       ├── SimpleCNN.keras
   │   │       └── ...
   │   └── fusion_saved/
   │       └── models/
   │           └── FusionModel.keras
   ├── api/
   ├── frontend/
   └── ...
   ```

### Backend Setup

```bash
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Navigate to [http://localhost:5173](http://localhost:5173) to view the Model Playground.
