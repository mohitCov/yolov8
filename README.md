# YOLOv8-CBAM Implementation

This repository contains an implementation attempt to integrate CBAM (Convolutional Block Attention Module) into YOLOv8. The goal is to enhance YOLOv8's feature extraction capabilities by adding attention mechanisms after each C2f block.

## Project Structure

```
├── ultralytics-main/          # Modified ultralytics repository
│   └── ultralytics/
│       ├── cfg/
│       │   ├── models/
│       │   │   └── v8/
│       │   │       └── yolov8-cbam.yaml    # Modified YOLOv8 architecture with CBAM
│       │   └── datasets/
│       │       └── kitti.yaml              # KITTI dataset configuration
│       └── nn/
│           └── modules/
│               └── block.py                # Contains CBAM and C2fCBAM implementations
├── train_yolo_cbam.py         # Training script
└── data.yaml                  # Dataset configuration
```

## Modifications Made

1. **CBAM Implementation**: 
   - Added CBAM module in `block.py`
   - Implemented channel and spatial attention mechanisms
   - Integrated with C2f block to create C2fCBAM

2. **Model Architecture**:
   - Modified YOLOv8 architecture to use C2fCBAM blocks
   - Maintained original network structure while adding attention mechanisms

## Current Issues

We are facing the following challenges:

1. **Import Issues**:
   ```python
   from ultralytics import YOLO
   ImportError: cannot import name 'YOLO' from 'ultralytics' (unknown location)
   ```
   Despite proper installation and path configuration, the YOLO class import is failing.

2. **Installation Challenges**:
   - Local installation of modified ultralytics package shows successful but imports fail
   - Package shows as installed: `C:\Users\Mohit khadikar\YOLOv8_CBAM\ultralytics-main`
   - Editable install (`pip install -e .`) completes but module access issues persist

## Setup Attempts

1. **Installation Steps Tried**:
   ```bash
   cd C:\Users\Mohit khadikar\YOLOv8_CBAM\ultralytics-main
   pip uninstall ultralytics -y
   pip install -e .
   ```

2. **Import Verification**:
   ```python
   import ultralytics
   print(ultralytics.__file__)  # Shows correct path but YOLO import fails
   ```

## Dataset Configuration

Using KITTI dataset with 8 classes:
- Classes: Car, Pedestrian, Van, Cyclist, Person_sitting, Misc, Truck, Tram
- Training images: F:/YOLOmodification/datasets/KITTI_Dataset/dataset/train/images
- Validation images: F:/YOLOmodification/datasets/KITTI_Dataset/dataset/val/images

## Next Steps

1. Resolve import issues with the YOLO class
2. Verify CBAM integration in the architecture
3. Test training pipeline
4. Evaluate model performance

## Contributing

If you have experience with YOLOv8 modifications or have successfully integrated attention modules, your contributions would be greatly appreciated. Please check the issues section for current challenges.

## References

1. Original YOLOv8: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
2. CBAM Paper: [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)

## Status

🚧 **Work in Progress** 🚧

Currently working on resolving import issues and ensuring proper integration of CBAM modules. 