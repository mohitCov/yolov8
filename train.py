import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from fromScratch import YOLOv8_Custom, YOLODataset
import argparse
from tqdm import tqdm
import yaml
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 Custom')
    parser.add_argument('--data', type=str, default='data.yaml', help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()

def create_data_yaml():
    """Create a default data.yaml if it doesn't exist"""
    if not os.path.exists('data.yaml'):
        data = {
            'train': 'data/train/images',
            'val': 'data/val/images',
            'nc': 80,  # number of classes
            'names': ['class1', 'class2', '...']  # class names
        }
        with open('data.yaml', 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        print("Created default data.yaml. Please modify it with your dataset information.")
        exit(0)

def compute_yolo_loss(outputs, targets):
    """Compute YOLO loss for classification and regression"""
    # Unpack outputs
    pred_cls, pred_reg = outputs
    
    # Classification loss (BCE with logits)
    cls_loss = F.binary_cross_entropy_with_logits(pred_cls, targets[:, :, 0].long())
    
    # Regression loss (IoU loss)
    pred_boxes = pred_reg.sigmoid()  # Convert to 0-1 range
    target_boxes = targets[:, :, 1:5]  # [x, y, w, h]
    
    # Calculate IoU between predicted and target boxes
    iou_loss = 1 - box_iou(pred_boxes, target_boxes).mean()
    
    # Total loss (you can adjust the weights)
    total_loss = cls_loss + iou_loss
    return total_loss

def box_iou(box1, box2):
    """Calculate IoU between box1 and box2"""
    # Convert to x1, y1, x2, y2 format
    b1_x1, b1_y1 = box1[:, :, 0] - box1[:, :, 2] / 2, box1[:, :, 1] - box1[:, :, 3] / 2
    b1_x2, b1_y2 = box1[:, :, 0] + box1[:, :, 2] / 2, box1[:, :, 1] + box1[:, :, 3] / 2
    b2_x1, b2_y1 = box2[:, :, 0] - box2[:, :, 2] / 2, box2[:, :, 1] - box2[:, :, 3] / 2
    b2_x2, b2_y2 = box2[:, :, 0] + box2[:, :, 2] / 2, box2[:, :, 1] + box2[:, :, 3] / 2
    
    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    return iou

def main():
    args = parse_args()
    
    # Create default data.yaml if it doesn't exist
    create_data_yaml()
    
    # Load dataset configuration
    with open(args.data) as f:
        data = yaml.safe_load(f)
    
    # Create datasets
    train_dataset = YOLODataset(
        img_dir=data['train'],
        label_dir=data['train'].replace('images', 'labels'),
        img_size=args.img_size,
        augment=True
    )
    
    val_dataset = YOLODataset(
        img_dir=data['val'],
        label_dir=data['val'].replace('images', 'labels'),
        img_size=args.img_size,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn if hasattr(val_dataset, 'collate_fn') else None
    )
    
    # Initialize model
    model = YOLOv8_Custom(num_classes=data['nc'])
    model = model.to(args.device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, (images, targets) in enumerate(train_progress):
            images = images.to(args.device)
            targets = targets.to(args.device)
            
            # Forward pass
            outputs = model(images)
            loss = compute_yolo_loss(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            train_loss += loss.item()
            train_progress.set_postfix({'loss': train_loss / (batch_idx + 1)})
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validation'):
                images = images.to(args.device)
                targets = targets.to(args.device)
                outputs = model(images)
                val_loss += compute_yolo_loss(outputs, targets).item()
        
        val_loss /= len(val_loader)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, 'latest_model.pth')
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')

if __name__ == '__main__':
    main() 