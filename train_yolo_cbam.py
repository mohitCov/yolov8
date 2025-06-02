import sys
import os

# Get absolute path to ultralytics directory
current_dir = os.path.dirname(os.path.abspath(__file__))
ultralytics_path = os.path.join(current_dir, 'ultralytics-main')

# Add to Python path
if ultralytics_path not in sys.path:
    sys.path.insert(0, ultralytics_path)

# Import after adding to path
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.modules.block import C2fCBAM, CBAM

# Create a new YOLO model from the custom YAML
model = YOLO(os.path.join(ultralytics_path, 'ultralytics/cfg/models/v8/yolov8-cbam.yaml'))

# Train the model
results = model.train(
    data=os.path.join(ultralytics_path, 'ultralytics/cfg/datasets/kitti.yaml'),
    epochs=5,
    imgsz=640,
    batch=8,
    device='cpu',
    verbose=True,
    project='runs/train',
    name='yolov8_cbam_kitti'
) 