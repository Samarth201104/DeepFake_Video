# DeepFake Video Detection

A machine learning project that detects deepfake videos using the Xception neural network architecture. This system analyzes video frames to classify content as either real or fake with high accuracy.

## Project Overview

This project leverages deep learning techniques to identify manipulated or artificially generated video content. It uses an Xception model trained from scratch on your deepfake dataset, capable of analyzing facial regions in videos to distinguish authentic footage from synthetic or manipulated content.

## Features

- **Video Deepfake Detection**: Analyzes videos frame-by-frame to detect fake content
- **Face Detection**: Uses RetinaFace for accurate facial region detection
- **Web Interface**: Flask-based web application for easy video upload and analysis
- **Trained Model**: Includes `xception_best.h5` model trained from scratch on your deepfake dataset
- **Data Preprocessing**: Automated pipeline for dataset preparation and augmentation
- **Performance Evaluation**: Built-in evaluation metrics and result visualization

## File Descriptions

### Core Scripts

- **train.py**: Model training with GPU optimization and mixed precision
  - Handles data augmentation, class balancing, and callbacks
  - Saves best model based on validation accuracy
  
- **preprocess.py**: Dataset preprocessing pipeline
  - Extracts faces using RetinaFace
  - Applies augmentation and resizing
  - Balances real/fake video counts

- **split.py**: Quality-aware dataset splitting
  - Filters low-quality images (size, blur)
  - Splits into train/val/test with 70/15/15 ratio
  - Ensures randomization for unbiased splits

- **evaluate.py**: Comprehensive model evaluation
  - Generates metrics (accuracy, precision, recall, F1, ROC-AUC)
  - Creates confusion matrices and ROC curves
  - Saves visualizations to evaluation_results/

- **app.py**: Flask web application (currently commented)
  - User-friendly interface for video upload
  - Real-time deepfake detection with confidence scores
  - Video result visualization

- **main.py**: Real-time preprocessing progress monitor
  - Tracks video and frame counts during preprocessing
  - Updates every 60 seconds
  - Useful for monitoring long preprocessing jobs

### Model File

- **xception_best.h5**: Trained Xception model
  - Best model checkpoint from training (selected by validation accuracy)
  - Trained from scratch on your dataset
  - Saved in Keras H5 format
  - Ready for inference

### Configuration Files

- **requirements.txt**: Python package dependencies
  - All necessary packages with pinned versions
  - Includes both production and development packages

## Directory Structure Explained

- **dataset/**: Original video frames or video files
  - real/: Authentic video content
  - fake/: Deepfake or manipulated content

- **preprocessed/**: Extracted and preprocessed face crops
  - Applied augmentation and normalization
  - Input to split.py

- **preprocessed_split/**: Quality-filtered and organized data
  - train/: 70% for model training
  - val/: 15% for validation during training
  - test/: 15% for final evaluation
  - Each split contains real/ and fake/ subdirectories

- **evaluation_results/**: Model evaluation outputs
  - confusion_matrix_counts.png: Raw counts
  - confusion_matrix_percent.png: Normalized percentages
  - roc_curve.png: ROC curve visualization
  - metrics_report.txt: Numerical results

- **uploads/**: User-uploaded files for Flask app
- **static/**: CSS and web assets
- **templates/**: HTML templates for Flask web UI

## Requirements

- Python 3.8+
- GPU support (CUDA for TensorFlow-GPU recommended)
- Windows/Linux/macOS

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Samarth201104/DeepFake_Video.git
cd DeepFake_Video
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### Key Dependencies

- **TensorFlow/Keras 2.10.0**: Deep learning framework with GPU support
- **OpenCV 4.11.0**: Video and image processing (also includes headless version)
- **RetinaFace / InsightFace**: Advanced face detection and recognition
- **Scikit-learn**: Machine learning utilities and metrics
- **Flask 3.1.2**: Web framework for the Flask web UI
- **Pillow/ImageIO**: Image manipulation and processing
- **NumPy 1.23.5**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization for evaluation metrics
- **Albumentations**: Advanced data augmentation
- **ImgAug**: Image augmentation library
- **MTCNN**: Alternative face detection method
- **ONNX**: Model format support

## Usage

### Data Preparation

#### 1. Preprocess Dataset

```bash
python preprocess.py
```

Preprocessing pipeline:
- Extracts faces from videos using RetinaFace (GPU-accelerated)
- Processes every 15th frame to balance computation and data coverage
- Resizes all face crops to 256×256 pixels
- Applies data augmentation (rotation, shifts, zooming, flips)
- Balances dataset by matching fake video count to real video count
- Outputs preprocessed faces to `preprocessed/{real,fake}/` directories

#### 2. Split Dataset

```bash
python split.py
```

Dataset splitting features:
- Filters low-quality images using:
  - **Size threshold**: Removes faces smaller than 30×30 pixels
  - **Blur detection**: Uses Laplacian variance to filter blurry images (threshold: 20)
- Splits data into:
  - **Train**: 70% for model training
  - **Validation**: 15% for tuning hyperparameters
  - **Test**: 15% for final evaluation
- Organized structure: `preprocessed_split/{train,val,test}/{real,fake}/`
- Randomizes image order before splitting

### Training

```bash
python train.py
```

The training script includes:
- **GPU Memory Growth**: Automatically configures GPU to allocate memory as needed
- **Mixed Precision Training**: Uses float16 for faster computation and reduced memory usage
- **Data Augmentation**: Applies rotation, shifting, zooming, and flipping during training
- **Class Balancing**: Automatically computes class weights to handle imbalanced datasets
- **Fine-tuning**: Freezes early Xception layers and trains the last 20 layers
- **Early Stopping**: Stops training if validation loss doesn't improve for 5 epochs
- **Learning Rate Reduction**: Reduces learning rate by 30% if validation loss plateaus
- **Model Checkpointing**: Saves the best model based on validation accuracy

Training hyperparameters:
- Batch size: 16
- Epochs: 50 (with early stopping)
- Learning rate: 1e-4
- Optimizer: Adam
- Loss function: Binary Crossentropy
- Image size: 256×256 pixels

The best model is automatically saved as `xception_best.h5`.

### Evaluation

```bash
python evaluate.py
```

Comprehensive model evaluation that generates:
- **Metrics**: Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Classification Report**: Detailed per-class metrics
- **Confusion Matrix**: Two versions
  - Counts: Absolute number of predictions
  - Percentage: Normalized by true labels
- **ROC Curve**: Visual representation of model performance across thresholds
- **Visualizations**: Saved as PNG files in `evaluation_results/`

All results are displayed interactively and saved for documentation.

### Monitoring Progress

```bash
python main.py
```

Real-time monitoring of preprocessing progress with frame and video counts.

### Web Interface (Optional)

```bash
python app.py
```

Launches a Flask web application for interactive video upload and deepfake detection. Access at `http://localhost:5000`

## Model Architecture

**Base Model**: Xception (Extreme Inception)
- Initialized with ImageNet weights (transfer learning base)
- Depthwise separable convolutions for efficiency
- Input: 256×256 RGB images
- Output: Single neuron with sigmoid activation (binary classification)

**Custom Layers**:
- Global Average Pooling 2D (reduces spatial dimensions)
- Dropout (50% rate for regularization)
- Dense output layer with sigmoid activation

**Training Configuration**:
- Last 20 layers of Xception are trainable
- Earlier layers remain frozen to preserve ImageNet features and accelerate convergence
- Trained from scratch on your deepfake dataset
- Optimized with Adam optimizer (lr=1e-4)
- Uses binary crossentropy loss for binary classification
- Implements mixed precision training for efficiency
- Best model saved based on validation accuracy

**Performance**:
- Lightweight and efficient for real-time inference
- Fast face-by-face and video-level predictions
- Balanced accuracy across real and fake samples

## How It Works

### Pipeline Overview

1. **Face Detection**: RetinaFace detects all facial regions in video frames with high precision
2. **Face Extraction**: Crops detected faces and resizes to 256×256 pixels (model input size)
3. **Normalization**: Scales pixel values to [0, 1] range for model input
4. **Classification**: Xception model predicts real/fake probability for each face
5. **Aggregation**: Frame-level predictions combined across multiple frames
6. **Video-Level Verdict**: Averages face probabilities to determine overall video classification
7. **Confidence Score**: Reports certainty as max(probability, 1-probability)

### Processing Flow

```
Video Input
    ↓
Frame Extraction (every 15th frame)
    ↓
Face Detection (RetinaFace)
    ↓
Face Preprocessing (Crop & Resize to 256×256)
    ↓
Xception Prediction (Binary: 0=Real, 1=Fake)
    ↓
Probability Aggregation
    ↓
Video Classification (Real/Fake) + Confidence
```

### Key Parameters

- **Frame Skip**: Process every 15th frame to balance computation and coverage
- **Face Threshold**: Minimum face confidence from RetinaFace detection
- **Classification Threshold**: 0.5 probability threshold (configurable)
- **Batch Processing**: Supports multiple faces per frame

## Data Processing Details

### Preprocessing Pipeline

Face Extraction:
- Uses RetinaFace for accurate face detection (GPU-accelerated)
- Extracts faces every 15 frames to balance data volume and temporal diversity
- Crops facial regions with proper padding
- Detects and handles multiple faces per frame

Augmentation Techniques:
- **Rotation**: ±15 degrees
- **Shifting**: ±10% horizontal and vertical
- **Zooming**: ±20% scale variation
- **Flipping**: Random horizontal flips
- **Fill Mode**: Nearest pixel value for handling borders

Quality Filtering (split.py):
- **Minimum Size**: 30×30 pixels (removes very small faces)
- **Blur Detection**: Laplacian variance threshold of 20
  - Filters out motion blur and out-of-focus frames
  - Ensures model trains on clear, usable images

### Training Augmentation

Applied during training (flow_from_directory):
- Rescaling to [0, 1] range
- 15° rotation range
- ±10% width and height shifts
- ±20% zoom range
- Horizontal flips
- Nearest-neighbor fill for edge pixels

Validation/Test augmentation:
- Only rescaling to [0, 1] (no geometric transformations)
- Ensures fair evaluation without synthetic variations

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives among predicted positives
- **Recall**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

Results are saved in `evaluation_results/` directory.

## Performance Optimization

### GPU Acceleration

The project automatically configures GPU acceleration:

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

- **Memory Growth**: Allocates GPU memory dynamically (not upfront) to avoid OOM errors
- **Mixed Precision**: Uses float16 for computation and float32 for outputs
  - Reduces memory usage by ~50%
  - Speeds up training on modern GPUs (RTX, A100, etc.)
  - Maintains numerical stability with automatic loss scaling

### System Requirements

- **NVIDIA GPU**: CUDA-capable GPU (GTX 1050 or better)
- **CUDA Toolkit**: 11.x (compatible with TensorFlow 2.10.0)
- **cuDNN**: 8.x (for optimal performance)
- **CPU Alternative**: Can run on CPU, but training will be significantly slower

### Optimization Tips

1. **Reduce batch size** (from 16 to 8 or 4) if experiencing GPU memory issues
2. **Use mixed precision** for faster training (enabled by default)
3. **Adjust frame_skip** in preprocess.py for faster preprocessing
4. **Enable gradient accumulation** for larger effective batch sizes on smaller GPUs

## Troubleshooting

**Issue**: GPU not detected or CUDA errors
- Check CUDA installation: `nvidia-smi`
- Verify TensorFlow GPU: `python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"`
- Reinstall tensorflow-gpu if needed: `pip install --upgrade tensorflow-gpu==2.10.0`

**Issue**: Out of memory (OOM) errors
- Reduce batch size in training scripts (16 → 8 or 4)
- Reduce image size (256 → 224)
- Enable memory growth (done automatically)
- Clear GPU cache: Restart the Python kernel

**Issue**: Face detection failures or poor accuracy
- Ensure video quality is adequate (minimum 480p recommended)
- Check lighting conditions (RetinaFace works best with good lighting)
- Verify frame rate and resolution
- Try increasing frame_skip or using alternative detectors (MTCNN)

**Issue**: Training convergence problems
- Increase epochs or patience for early stopping
- Adjust learning rate (decrease if oscillating)
- Check class balance in training data
- Verify data augmentation isn't too aggressive

**Issue**: Low evaluation accuracy
- Check data quality (preprocess.py quality filters may be too strict)
- Increase training data volume
- Adjust class weight balancing
- Experiment with different architectures or pre-training strategies

## Future Improvements

- Support for multiple deepfake generation methods detection
- Real-time video stream processing
- Mobile deployment
- Ensemble models for improved accuracy
- Temporal analysis across frames

## License

This project is part of the DeepFake_Video research repository.

## Contact

For questions or contributions, please contact the project maintainer at the repository.

## References

- Xception: Deep Learning with Depthwise Separable Convolutions
- RetinaFace: Single-stage Dense Face Localisation in the Wild
- Deepfake detection research papers and benchmarks
