import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
import torch
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignatureStampDetector:
    """
    YOLOv8-based detector for signatures and stamps in documents
    Uses your existing 500 invoice images for training/validation
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 train_data_path: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights (.pt file)
            train_data_path: Path to your train folder with 500 images
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.train_data_path = train_data_path
        
        # Check if CUDA is available
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO model from {model_path}")
        else:
            # Load pretrained model
            self.model = YOLO('yolov8n.pt')
            logger.info("Loaded pretrained YOLOv8n model")
            
            # Check if we should train on your data
            if train_data_path and os.path.exists(train_data_path):
                logger.info(f"Training data available at {train_data_path}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Class names
        self.class_names = {
            0: 'signature',
            1: 'stamp'
        }
        
        # Confidence thresholds
        self.conf_thresholds = {
            'signature': 0.5,
            'stamp': 0.6
        }
        
        # NMS threshold
        self.iou_threshold = 0.5
        
        logger.info(f"Detector initialized on {self.device}")
    
    def create_yolo_dataset(self, annotations_file: Optional[str] = None):
        """
        Create YOLO dataset structure from your 500 images
        You need to create annotations for these images
        """
        if not self.train_data_path:
            logger.error("No training data path provided")
            return None
        
        # Check if images exist
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(Path(self.train_data_path).glob(f"*{ext}")))
            image_files.extend(list(Path(self.train_data_path).glob(f"*{ext.upper()}")))
        
        logger.info(f"Found {len(image_files)} images in {self.train_data_path}")
        
        if len(image_files) == 0:
            logger.error("No images found in train folder")
            return None
        
        # Create dataset directory structure
        dataset_dir = "yolo_dataset"
        train_images_dir = os.path.join(dataset_dir, "images", "train")
        val_images_dir = os.path.join(dataset_dir, "images", "val")
        train_labels_dir = os.path.join(dataset_dir, "labels", "train")
        val_labels_dir = os.path.join(dataset_dir, "labels", "val")
        
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(train_labels_dir, exist_ok=True)
        os.makedirs(val_labels_dir, exist_ok=True)
        
        # Split data (80% train, 20% val)
        train_count = int(len(image_files) * 0.8)
        train_files = image_files[:train_count]
        val_files = image_files[train_count:]
        
        # Create symbolic links or copy images
        logger.info("Creating dataset structure...")
        
        # If annotations file provided, parse it
        annotations = {}
        if annotations_file and os.path.exists(annotations_file):
            try:
                with open(annotations_file, 'r') as f:
                    annotations = json.load(f)
                logger.info(f"Loaded annotations from {annotations_file}")
            except Exception as e:
                logger.error(f"Error loading annotations: {e}")
        
        # Process training images
        for i, img_path in enumerate(train_files):
            img_name = img_path.stem
            
            # Copy/Link image
            dest_path = os.path.join(train_images_dir, img_path.name)
            if not os.path.exists(dest_path):
                try:
                    os.symlink(img_path.resolve(), dest_path)
                except:
                    import shutil
                    shutil.copy2(img_path, dest_path)
            
            # Create label file if annotations exist
            if img_name in annotations:
                label_data = annotations[img_name]
                label_path = os.path.join(train_labels_dir, f"{img_name}.txt")
                self._create_label_file(label_path, label_data, img_path)
        
        # Process validation images
        for i, img_path in enumerate(val_files):
            img_name = img_path.stem
            
            # Copy/Link image
            dest_path = os.path.join(val_images_dir, img_path.name)
            if not os.path.exists(dest_path):
                try:
                    os.symlink(img_path.resolve(), dest_path)
                except:
                    import shutil
                    shutil.copy2(img_path, dest_path)
            
            # Create label file if annotations exist
            if img_name in annotations:
                label_data = annotations[img_name]
                label_path = os.path.join(val_labels_dir, f"{img_name}.txt")
                self._create_label_file(label_path, label_data, img_path)
        
        # Create dataset YAML
        yaml_content = f"""
path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/val

names:
  0: signature
  1: stamp
        """
        
        yaml_path = os.path.join(dataset_dir, "dataset.yaml")
        with open(yaml_path, 'w') as f:
            f.write(yaml_content.strip())
        
        logger.info(f"Created YOLO dataset at {dataset_dir}")
        logger.info(f"Training images: {len(train_files)}")
        logger.info(f"Validation images: {len(val_files)}")
        
        return yaml_path
    
    def _create_label_file(self, label_path: str, label_data: Dict, img_path: Path):
        """Create YOLO format label file"""
        try:
            # Read image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Cannot read image {img_path}")
                return
            
            height, width = img.shape[:2]
            
            lines = []
            
            # Process signature bounding boxes
            for sig_bbox in label_data.get('signatures', []):
                x1, y1, x2, y2 = sig_bbox
                
                # Convert to YOLO format
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                
                lines.append(f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
            
            # Process stamp bounding boxes
            for stamp_bbox in label_data.get('stamps', []):
                x1, y1, x2, y2 = stamp_bbox
                
                # Convert to YOLO format
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                
                lines.append(f"1 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")
            
            # Write label file
            with open(label_path, 'w') as f:
                f.write('\n'.join(lines))
                
        except Exception as e:
            logger.error(f"Error creating label file {label_path}: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Same preprocessing as before"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        # Enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Same detection as before"""
        try:
            processed_image = self.preprocess_image(image)
            img_height, img_width = processed_image.shape[:2]
            
            results = self.model(
                processed_image,
                conf=0.25,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, confidence, class_id in zip(boxes, confidences, class_ids):
                        class_name = self.class_names.get(class_id, f'class_{class_id}')
                        class_threshold = self.conf_thresholds.get(class_name, 0.5)
                        
                        if confidence < class_threshold:
                            continue
                        
                        x1, y1, x2, y2 = map(float, box)
                        x1 = max(0, min(x1, img_width))
                        y1 = max(0, min(y1, img_height))
                        x2 = max(0, min(x2, img_width))
                        y2 = max(0, min(y2, img_height))
                        
                        area = (x2 - x1) * (y2 - y1)
                        img_area = img_width * img_height
                        
                        if area < 0.0001 * img_area or area > 0.5 * img_area:
                            continue
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'class': class_name,
                            'confidence': float(confidence),
                            'area': area,
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                        })
            
            detections = self.non_max_suppression(detections)
            return detections
            
        except Exception as e:
            logger.error(f"Error in detection: {e}")
            return []
    
    def non_max_suppression(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Same NMS as before"""
        if not detections:
            return []
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        filtered = []
        
        while detections:
            best = detections.pop(0)
            filtered.append(best)
            
            to_remove = []
            for i, det in enumerate(detections):
                iou = self.calculate_iou(best['bbox'], det['bbox'])
                if iou > iou_threshold:
                    to_remove.append(i)
            
            for i in reversed(to_remove):
                detections.pop(i)
        
        return filtered
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Same IoU calculation as before"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def extract_signature_stamp_info(self, image: np.ndarray) -> Dict[str, Any]:
        """Same extraction as before"""
        detections = self.detect(image)
        
        results = {
            'signature': {
                'present': False,
                'bbox': None,
                'confidence': 0.0,
                'count': 0
            },
            'stamp': {
                'present': False,
                'bbox': None,
                'confidence': 0.0,
                'count': 0
            },
            'all_detections': detections
        }
        
        signature_dets = [d for d in detections if d['class'] == 'signature']
        stamp_dets = [d for d in detections if d['class'] == 'stamp']
        
        if signature_dets:
            best_sig = max(signature_dets, key=lambda x: x['confidence'])
            results['signature'] = {
                'present': True,
                'bbox': best_sig['bbox'],
                'confidence': best_sig['confidence'],
                'count': len(signature_dets)
            }
        
        if stamp_dets:
            best_stamp = max(stamp_dets, key=lambda x: x['confidence'])
            results['stamp'] = {
                'present': True,
                'bbox': best_stamp['bbox'],
                'confidence': best_stamp['confidence'],
                'count': len(stamp_dets)
            }
        
        return results
    
    def train_on_existing_data(self, epochs: int = 50, save_path: str = "models/trained_model.pt"):
        """
        Train model on your existing 500 images
        Note: You need to create annotations first
        """
        try:
            # First create YOLO dataset structure
            dataset_yaml = self.create_yolo_dataset()
            
            if not dataset_yaml:
                logger.error("Could not create dataset structure")
                return False
            
            logger.info(f"Starting training with dataset: {dataset_yaml}")
            
            # Train the model
            self.model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=640,
                batch=16,
                device=self.device,
                workers=4,
                save=True,
                save_period=10,
                pretrained=True
            )
            
            # Save trained model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)
            logger.info(f"Model trained and saved to {save_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False
    
    def auto_annotate(self, output_file: str = "auto_annotations.json"):
        """
        Create automatic annotations using the current model
        This can help bootstrap the annotation process
        """
        if not self.train_data_path:
            logger.error("No training data path provided")
            return
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(Path(self.train_data_path).glob(f"*{ext}")))
        
        logger.info(f"Auto-annotating {len(image_files)} images...")
        
        annotations = {}
        
        for i, img_path in enumerate(image_files):
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Detect signatures and stamps
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                detections = self.detect(img_rgb)
                
                # Filter and organize detections
                signatures = []
                stamps = []
                
                for det in detections:
                    if det['class'] == 'signature' and det['confidence'] > 0.7:
                        signatures.append(det['bbox'])
                    elif det['class'] == 'stamp' and det['confidence'] > 0.7:
                        stamps.append(det['bbox'])
                
                if signatures or stamps:
                    annotations[img_path.stem] = {
                        'signatures': signatures,
                        'stamps': stamps,
                        'image_path': str(img_path)
                    }
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_files)} images")
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
        
        # Save annotations
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        logger.info(f"Auto-annotations saved to {output_file}")
        logger.info(f"Annotated {len(annotations)} images")
        
        return annotations

# Singleton instance
_detector = None

def get_detector(model_path: Optional[str] = None, 
                 train_data_path: Optional[str] = None,
                 device: str = 'cpu') -> SignatureStampDetector:
    global _detector
    if _detector is None:
        _detector = SignatureStampDetector(model_path, train_data_path, device)
    return _detector

if __name__ == "__main__":
    # Test with your existing data
    detector = SignatureStampDetector(train_data_path="train")
    
    # Check your images
    print(f"Training data path: train/")
    
    # Try to auto-annotate
    annotations = detector.auto_annotate("auto_annotations.json")
    
    # Create dataset structure
    dataset_yaml = detector.create_yolo_dataset("auto_annotations.json")
    
    if dataset_yaml:
        print(f"\nDataset created at: {dataset_yaml}")
        print("You can now train the model with:")
        print("detector.train_on_existing_data(epochs=50)")
    else:
        print("\nCould not create dataset. Check if images exist in train/ folder")