import os
import cv2
import numpy as np
from tqdm import tqdm

def get_image_size(image_path):
    print(f"Reading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img.shape[1], img.shape[0]  # width, height

def convert_kitti_to_yolo(label_path, image_path, class_mapping):
    try:
        # Get image dimensions
        img_width, img_height = get_image_size(image_path)
        print(f"Image dimensions: {img_width}x{img_height}")
        
        # Read KITTI label file
        print(f"Reading label file: {label_path}")
        yolo_labels = []
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f.readlines(), 1):
                parts = line.strip().split()
                if len(parts) < 15:
                    print(f"Warning: Line {line_num} has insufficient parts: {line.strip()}")
                    continue
                if parts[0] == 'DontCare':
                    continue
                    
                # Parse KITTI format
                class_name = parts[0]
                if class_name not in class_mapping:
                    print(f"Warning: Unknown class '{class_name}' in line {line_num}")
                    continue
                    
                class_id = class_mapping[class_name]
                try:
                    x1, y1, x2, y2 = map(float, parts[4:8])
                except ValueError as e:
                    print(f"Error parsing coordinates in line {line_num}: {line.strip()}")
                    continue
                
                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Ensure values are within [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
        print(f"Converted {len(yolo_labels)} labels for {os.path.basename(image_path)}")
        return yolo_labels
    except Exception as e:
        print(f"Error processing {label_path}: {str(e)}")
        return []

def process_dataset(base_dir, split, class_mapping):
    print(f"\nProcessing {split} set...")
    images_dir = os.path.join(base_dir, split, 'images')
    labels_dir = os.path.join(base_dir, split, 'labels')
    yolo_labels_dir = os.path.join(base_dir, split, 'labels_yolo')
    
    # Verify directories exist
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    if not os.path.exists(labels_dir):
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    
    # Create YOLO labels directory
    os.makedirs(yolo_labels_dir, exist_ok=True)
    print(f"YOLO labels will be saved to: {yolo_labels_dir}")
    
    # Process all images
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    print(f"Found {len(image_files)} images")
    
    successful_conversions = 0
    for img_file in tqdm(image_files, desc=f'Converting {split} set'):
        base_name = os.path.splitext(img_file)[0]
        label_file = os.path.join(labels_dir, f'{base_name}.txt')
        image_file = os.path.join(images_dir, img_file)
        
        if os.path.exists(label_file):
            # Convert labels
            yolo_labels = convert_kitti_to_yolo(label_file, image_file, class_mapping)
            
            if yolo_labels:
                # Save YOLO format labels
                yolo_label_file = os.path.join(yolo_labels_dir, f'{base_name}.txt')
                with open(yolo_label_file, 'w') as f:
                    f.write('\n'.join(yolo_labels))
                successful_conversions += 1
        else:
            print(f"Warning: No label file found for {img_file}")
    
    print(f"\nCompleted {split} set conversion:")
    print(f"Total images: {len(image_files)}")
    print(f"Successful conversions: {successful_conversions}")

def main():
    print("Starting KITTI to YOLO conversion...")
    
    # Define class mapping (KITTI class name to index)
    class_mapping = {
        'Car': 0,
        'Pedestrian': 1,
        'Van': 2,
        'Cyclist': 3,
        'Person_sitting': 4,
        'Misc': 5,
        'Truck': 6,
        'Tram': 7
    }
    print("\nClass mapping:")
    for name, idx in class_mapping.items():
        print(f"  {name}: {idx}")
    
    # Base directory of the KITTI dataset
    base_dir = "F:/YOLOmodification/datasets/KITTI_Dataset/dataset"
    print(f"\nBase directory: {base_dir}")
    
    try:
        # Process both train and val sets
        process_dataset(base_dir, 'train', class_mapping)
        process_dataset(base_dir, 'val', class_mapping)
        
        print("\nConversion completed successfully!")
        print("Labels are saved in 'labels_yolo' directories.")
    except Exception as e:
        print(f"\nError during conversion: {str(e)}")
        raise

if __name__ == '__main__':
    main() 