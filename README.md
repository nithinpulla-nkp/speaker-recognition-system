# Speaker Recognition System

A machine learning-based speaker recognition system that identifies speakers using MFCC features and Gaussian Mixture Models (GMM) or Support Vector Machines (SVM).

## Features

- **MFCC Feature Extraction**: 20 MFCC + 20 Delta MFCC coefficients (40 total features)
- **Dual Model Support**: GMM (97.14% accuracy) and SVM (70.08% accuracy) models
- **Unknown Speaker Detection**: Handles speakers not in training data
- **Model Persistence**: Save and load trained models

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Training Data

Organize your training audio files in the following structure:
```
training_data/
├── speaker1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── speaker2/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
└── ...
```

### 2. Train Models

```bash
python speaker_recognition.py
# Choose option 1 for training
# Enter the path to your training_data directory
```

### 3. Test Recognition

```bash
python speaker_recognition.py  
# Choose option 2 for testing
# Enter path to test audio file
```

## Project Structure

- `speaker_recognition.py` - Main implementation
- `requirements.txt` - Python dependencies
- `models/` - Saved trained models (created after training)

## System Architecture & Code Flow

### Overall System Block Diagram

```
Audio Input → Feature Extraction → Model Training/Testing → Speaker Identification
     ↓              ↓                     ↓                        ↓
  .wav/.mp3    [MFCC + Delta]      [GMM/SVM Models]           [Speaker Name]
     ↓              ↓                     ↓                        ↓
 Raw Signal    40 Coefficients     Model Parameters         Confidence Score
```

### Training Phase Flow

```
Training Data Directory
         ↓
┌─────────────────────┐
│  Audio Files        │
│  (Multiple speakers)│
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Feature Extraction │
│  • Load audio       │
│  • Extract MFCC     │
│  • Extract Delta    │
│  • Combine features │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Model Training     │
│  • GMM per speaker  │
│  • SVM (all data)   │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Save Models        │
│  • .gmm files       │
│  • .pkl files       │
└─────────────────────┘
```

### Testing/Recognition Phase Flow

```
Test Audio File
         ↓
┌─────────────────────┐
│  Feature Extraction │
│  • Same as training │
│  • 40 coefficients  │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Load Trained Models│
│  • GMM models       │
│  • SVM model        │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Score Calculation  │
│  • GMM likelihood   │
│  • SVM probability  │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Decision Making    │
│  • Compare scores   │
│  • Apply threshold  │
│  • Unknown detection│
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Output Result      │
│  • Speaker name     │
│  • Confidence score │
└─────────────────────┘
```

### Feature Extraction Pipeline

```
Raw Audio Signal (22.05 kHz)
         ↓
┌─────────────────────┐
│     Framing         │
│  • 15-20ms frames   │
│  • Overlapping      │
└─────────────────────┘
         ↓
┌─────────────────────┐
│     Windowing       │
│  • Prevent leakage  │
│  • Hamming window   │
└─────────────────────┘
         ↓
┌─────────────────────┐
│       FFT           │
│  • Frequency domain │
│  • Spectral analysis│
└─────────────────────┘
         ↓
┌─────────────────────┐
│   Mel Filtering     │
│  • 26 mel filters   │
│  • Perceptual scale │
└─────────────────────┘
         ↓
┌─────────────────────┐
│   Log & DCT         │
│  • Log power        │
│  • 20 MFCC coeffs   │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Delta Features     │
│  • Temporal dynamics│
│  • 20 delta coeffs  │
└─────────────────────┘
         ↓
Combined Feature Vector (40 dimensions)
```

### Code Flow in Classes

#### SpeakerRecognition Class Methods

```python
# Main workflow methods:
__init__()              # Initialize parameters
    ↓
extract_features()      # Audio → MFCC+Delta features
    ↓
prepare_training_data() # Directory → Feature dictionary
    ↓
train_gmm_models()      # Features → Individual GMM models
    ↓
train_svm_model()       # Features → Single SVM model
    ↓
save_models()           # Models → Disk storage

# Prediction methods:
load_models()           # Disk → Memory
    ↓
predict_gmm()           # Features → Speaker + Score
predict_svm()           # Features → Speaker + Confidence
```

### Model Architecture Details

#### GMM (Gaussian Mixture Model)
```
For each speaker:
┌─────────────────────┐
│   Training Data     │
│  (MFCC features)    │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  EM Algorithm       │
│  • E-step: Estimate │
│  • M-step: Maximize │
│  • Iterate until    │
│    convergence      │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  GMM Parameters     │
│  • Means (μ)        │
│  • Covariances (Σ)  │
│  • Weights (π)      │
└─────────────────────┘
```

#### SVM (Support Vector Machine)
```
All Training Data
         ↓
┌─────────────────────┐
│  Feature Matrix     │
│  X: [n_samples,     │
│       n_features]   │
│  y: [speaker_labels]│
└─────────────────────┘
         ↓
┌─────────────────────┐
│  RBF Kernel         │
│  • Non-linear       │
│  • High dimensional │
│  • Separating plane │
└─────────────────────┘
         ↓
┌─────────────────────┐
│  Decision Boundary  │
│  • Support vectors  │
│  • Hyperplane      │
└─────────────────────┘
```

## Technical Details

### Feature Extraction
- **MFCC Features**: 20 coefficients using mel-frequency cepstral analysis
- **Delta Features**: 20 delta coefficients capturing temporal dynamics  
- **Audio Processing**: 22.05 kHz sample rate, windowing, overlapping frames
- **Total Dimensions**: 40 features per audio frame

### Model Specifications
- **GMM**: 16 components per speaker, diagonal covariance, EM algorithm
- **SVM**: RBF kernel, probability estimates enabled
- **Unknown Detection**: Score-based threshold (-50 default for GMM)

### Based on Original Research
This implementation is based on the academic project "Speaker Recognition Using Machine Learning" by:
- M Praveenkumar (CB.EN.U4ECE18145)
- P Nithin (CB.EN.U4ECE18146) 
- Swaminathan T (CB.EN.U4ECE18158)
- V Vaishnav Rengan (CB.EN.U4ECE18163)

Department of Electronics and Communication, Amrita School of Engineering, Coimbatore (2018-2022)