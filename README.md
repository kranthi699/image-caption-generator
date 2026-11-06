# Plant Disease Detection Using CNN

Classifies plant leaf images (Tomato, Potato, Pepper, Apple, Grape, Corn, ...) into healthy / disease classes using a Convolutional Neural Network, with a Streamlit web UI.

## Pipeline
Image -> resize (128x128) -> augmentation (train only) -> rescale -> 4x [Conv + ReLU + BatchNorm + MaxPool] -> Flatten -> Dense(256) + Dropout -> Softmax -> disease + confidence

## Setup
```bash
pip install -r requirements.txt
```

## Dataset (automatic download)
```bash
python download_data.py                                              # all 38 classes
python download_data.py --crops tomato potato pepper apple grape corn  # only the 6 crops in the brief
```
This fetches PlantVillage from Kaggle via `kagglehub` and creates `data/PlantVillage/<class>/*.jpg`. If Kaggle asks for credentials, create an API token at kaggle.com > Settings and set `KAGGLE_USERNAME` / `KAGGLE_KEY`.

## Train
```bash
python train.py --epochs 15          # uses data/PlantVillage by default
# optional higher-accuracy transfer learning:
python train.py --data_dir path/to/PlantVillage --model mobilenet
```
Outputs in `outputs/`: `plant_disease_model.keras`, `class_names.json`, `training_curves.png`, `confusion_matrix.png`, `classification_report.txt`.

## Run the app
```bash
streamlit run app.py
```

## Google Colab quick start
```python
from google.colab import drive; drive.mount('/content/drive')
!pip install -q -r requirements.txt
!python download_data.py --crops tomato potato pepper apple grape corn
!python train.py --epochs 15   # set Runtime > GPU first
```
Download the `outputs/` folder, place it next to `app.py`, then run Streamlit locally.

## Expected results
Custom CNN: roughly 94-98% validation accuracy on PlantVillage; MobileNetV2: 98%+. (Typical figures, so report your own measured numbers.)

## Notes for your report
- PlantVillage images are lab-style (plain background), so accuracy drops on real field photos. State this as a limitation; future work: train on field images (e.g. PlantDoc), Grad-CAM explainability, mobile deployment (TFLite).
- Metrics to include: accuracy, per-class precision/recall/F1, confusion matrix, training curves.
