# YOLOv5 Road Safety Helmet Detection

A computer vision project for detecting motorcycle riders, helmets, and number plates using YOLOv5. This system can identify riders wearing helmets, those without helmets, and capture number plates for road safety enforcement.

## 🎯 Project Overview

This project uses YOLOv5 to detect and classify:
- **With Helmet**: Riders wearing protective helmets
- **Without Helmet**: Riders not wearing helmets
- **Rider**: General rider detection
- **Number Plate**: Vehicle license plate detection

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Git
- WSL2 (if using Windows)

## 🚀 Installation

### 1. Clone YOLOv5 Repository

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n yolov5 python=3.8
conda activate yolov5

# Or using venv
python -m venv yolov5-env
source yolov5-env/bin/activate  # On Windows: yolov5-env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python detect.py --source data/images --weights yolov5s.pt --conf 0.25
```

## 📁 Dataset Structure

Organize your dataset inside the yolov5 directory:

```
yolov5/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── helmet_data.yaml
├── detect.py
├── train.py
├── val.py
└── yolov5s.pt
```

### Dataset Configuration (data/helmet_data.yaml)

Create your dataset YAML file inside the data directory:

```yaml
# data/helmet_data.yaml
train: data/train/images
val: data/val/images

nc: 4
names: ["with helmet", "without helmet", "rider", "number plate"]
```

## 🏋️ Training

### Basic Training Command

```bash
# Navigate to yolov5 directory first
cd yolov5

# Train with your helmet dataset
python train.py --img 640 --batch 16 --epochs 50 --data data/helmet_data.yaml --weights yolov5s.pt --name helmet_model --hyp data/hyps/hyp.scratch.yaml
```

### Training Parameters Explained

- `--img 640`: Input image size
- `--batch 16`: Batch size (adjust based on GPU memory)
- `--epochs 50`: Number of training epochs
- `--data data/helmet_data.yaml`: Path to your custom dataset YAML file
- `--weights yolov5s.pt`: Pre-trained weights
- `--name helmet_model`: Experiment name
- `--hyp data/hyps/hyp.scratch.yaml`: Hyperparameters file

### Advanced Training Options

```bash
# For better performance with more epochs
python train.py --img 640 --batch 16 --epochs 100 --data data/helmet_data.yaml --weights yolov5s.pt --name helmet_model_v2 --hyp data/hyps/hyp.scratch.yaml --patience 10

# For transfer learning with different model sizes
python train.py --img 640 --batch 8 --epochs 50 --data data/helmet_data.yaml --weights yolov5m.pt --name helmet_model_medium

# Resume training from checkpoint
python train.py --resume runs/train/helmet_model/weights/last.pt
```

## 📊 Data Preparation

### Setting Up Your Dataset

1. **Create the directory structure**:
```bash
cd yolov5
mkdir -p data/train/images data/train/labels data/val/images data/val/labels
```

2. **Copy your images and labels**:
```bash
# Copy training images
cp /path/to/your/train/images/* data/train/images/

# Copy training labels
cp /path/to/your/train/labels/* data/train/labels/

# Copy validation images
cp /path/to/your/val/images/* data/val/images/

# Copy validation labels
cp /path/to/your/val/labels/* data/val/labels/
```

3. **Verify your dataset**:
```bash
# Check if images and labels match
python -c "
import os
train_imgs = len(os.listdir('data/train/images'))
train_lbls = len(os.listdir('data/train/labels'))
val_imgs = len(os.listdir('data/val/images'))
val_lbls = len(os.listdir('data/val/labels'))
print(f'Train: {train_imgs} images, {train_lbls} labels')
print(f'Val: {val_imgs} images, {val_lbls} labels')
"
```

## 🔍 Inference

### Detect on Images

```bash
# From yolov5 directory
python detect.py --source path/to/images --weights runs/train/helmet_model/weights/best.pt --conf 0.4 --save-txt --save-conf
```

### Detect on Video

```bash
python detect.py --source path/to/video.mp4 --weights runs/train/helmet_model/weights/best.pt --conf 0.4
```

### Real-time Webcam Detection

```bash
python detect.py --source 0 --weights runs/train/helmet_model/weights/best.pt --conf 0.4
```

### Test with Sample Data

```bash
# Test with COCO128 sample data first
python detect.py --source data/images --weights yolov5s.pt --conf 0.25

# Test with your trained model
python detect.py --source data/val/images --weights runs/train/helmet_model/weights/best.pt --conf 0.4
```

## 📊 Model Evaluation

### Validation

```bash
python val.py --data data/helmet_data.yaml --weights runs/train/helmet_model/weights/best.pt --img 640
```

### Export Model

```bash
# Export to ONNX
python export.py --weights runs/train/helmet_model/weights/best.pt --include onnx

# Export to TensorRT
python export.py --weights runs/train/helmet_model/weights/best.pt --include engine --device 0
```

## 📈 Monitoring Training

Training results are saved in `runs/train/helmet_model/`:

- `weights/best.pt`: Best model weights
- `weights/last.pt`: Last epoch weights
- `results.png`: Training metrics plots
- `confusion_matrix.png`: Confusion matrix
- `train_batch*.jpg`: Training batch samples

### View Training Progress

```bash
# Install tensorboard
pip install tensorboard

# View training logs
tensorboard --logdir runs/train
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train.py --batch 8  # or smaller
   ```

2. **Dataset Path Issues**
   - Ensure paths in YAML file are relative to the YOLOv5 directory
   - Use forward slashes (/) even on Windows

3. **Label Format**
   - Ensure labels are in YOLO format: `class x_center y_center width height`
   - Values should be normalized (0-1)

### Performance Optimization

```bash
# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node 2 train.py --batch 32 --data data/helmet_data.yaml --weights yolov5s.pt --device 0,1

# Mixed precision training
python train.py --data data/helmet_data.yaml --weights yolov5s.pt --amp
```

## 📋 Requirements

Key dependencies:
- torch>=1.7.0
- torchvision>=0.8.1
- opencv-python>=4.1.1
- Pillow>=7.1.2
- PyYAML>=5.3.1
- requests>=2.23.0
- tqdm>=4.41.0
- tensorboard>=2.4.1
- wandb
- seaborn>=0.11.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) for the base framework
- Road safety dataset contributors
- Open source computer vision community

## 📞 Support

For issues and questions:
1. Check the [YOLOv5 documentation](https://docs.ultralytics.com/)
2. Search existing issues on GitHub
3. Create a new issue with detailed information

---

**Note**: This project is designed for educational and research purposes in road safety. Ensure compliance with local privacy and surveillance regulations when deploying in production environments.