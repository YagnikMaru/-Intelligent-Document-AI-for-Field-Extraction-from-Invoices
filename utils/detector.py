import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
import torch
import json
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignatureStampDetector:
    """
    Optimized YOLOv8-based detector for signatures and stamps
    """
    
    # Class-level constants
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    CLASS_NAMES = {0: 'signature', 1: 'stamp'}
    DEFAULT_CONF_THRESHOLDS = {'signature': 0.5, 'stamp': 0.6}
    
    def __init__(self, model_path: Optional[str] = None, 
                 train_data_path: Optional[str] = None,
                 device: str = 'cpu',
                 half_precision: bool = False):
        """
        Initialize optimized YOLO detector
        
        Args:
            model_path: Path to YOLO model weights
            train_data_path: Path to training images
            device: 'cpu' or 'cuda'
            half_precision: Use FP16 for faster inference (GPU only)
        """
        self.device = self._setup_device(device)
        self.train_data_path = Path(train_data_path) if train_data_path else None
        self.half_precision = half_precision and self.device != 'cpu'
        
        # Load model efficiently
        self.model = self._load_model(model_path)
        
        # Optimizations
        self.conf_thresholds = self.DEFAULT_CONF_THRESHOLDS
        self.iou_threshold = 0.5
        
        # Pre-create CLAHE instance (reusable)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        # Cache for image dimensions
        self._dim_cache = {}
        
        logger.info(f"Detector initialized on {self.device} (FP16: {self.half_precision})")
    
    @staticmethod
    def _setup_device(device: str) -> str:
        """Setup and validate device"""
        if device == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, using CPU")
                return 'cpu'
            # Set optimal CUDA settings
            torch.backends.cudnn.benchmark = True
            return 'cuda'
        return 'cpu'
    
    def _load_model(self, model_path: Optional[str]) -> YOLO:
        """Load model with optimizations"""
        if model_path and os.path.exists(model_path):
            model = YOLO(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            model = YOLO('yolov8n.pt')  # Nano model for speed
            logger.info("Loaded YOLOv8n model")
        
        model.to(self.device)
        
        # Apply half precision if enabled
        if self.half_precision:
            model.half()
            logger.info("Enabled FP16 inference")
        
        return model
    
    def _get_image_files(self, path: Path) -> List[Path]:
        """Efficiently get all image files"""
        files = []
        for ext in self.IMAGE_EXTENSIONS:
            files.extend(path.glob(f"*{ext}"))
            files.extend(path.glob(f"*{ext.upper()}"))
        return files
    
    def create_yolo_dataset(self, annotations_file: Optional[str] = None) -> Optional[str]:
        """Optimized dataset creation with parallel processing"""
        if not self.train_data_path or not self.train_data_path.exists():
            logger.error("Invalid training data path")
            return None
        
        image_files = self._get_image_files(self.train_data_path)
        logger.info(f"Found {len(image_files)} images")
        
        if not image_files:
            logger.error("No images found")
            return None
        
        # Setup directories
        dataset_dir = Path("yolo_dataset")
        dirs = {
            'train_img': dataset_dir / "images" / "train",
            'val_img': dataset_dir / "images" / "val",
            'train_lbl': dataset_dir / "labels" / "train",
            'val_lbl': dataset_dir / "labels" / "val"
        }
        
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        
        # Split data
        split_idx = int(len(image_files) * 0.8)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Load annotations once
        annotations = self._load_annotations(annotations_file)
        
        # Process in parallel
        logger.info("Creating dataset structure...")
        self._process_dataset_split(train_files, dirs['train_img'], dirs['train_lbl'], annotations)
        self._process_dataset_split(val_files, dirs['val_img'], dirs['val_lbl'], annotations)
        
        # Create YAML
        yaml_path = self._create_yaml(dataset_dir, len(train_files), len(val_files))
        
        return str(yaml_path)
    
    @staticmethod
    def _load_annotations(annotations_file: Optional[str]) -> Dict:
        """Load annotations with error handling"""
        if not annotations_file or not os.path.exists(annotations_file):
            return {}
        
        try:
            with open(annotations_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            return {}
    
    def _process_dataset_split(self, files: List[Path], img_dir: Path, 
                               lbl_dir: Path, annotations: Dict):
        """Process dataset split with batch operations"""
        for img_path in files:
            img_name = img_path.stem
            dest_path = img_dir / img_path.name
            
            # Create symlink or copy
            if not dest_path.exists():
                try:
                    dest_path.symlink_to(img_path.resolve())
                except:
                    import shutil
                    shutil.copy2(img_path, dest_path)
            
            # Create labels if annotations exist
            if img_name in annotations:
                label_path = lbl_dir / f"{img_name}.txt"
                self._create_label_file_fast(label_path, annotations[img_name], img_path)
    
    def _create_label_file_fast(self, label_path: Path, label_data: Dict, img_path: Path):
        """Optimized label file creation"""
        try:
            # Use cached dimensions if available
            img_key = str(img_path)
            if img_key in self._dim_cache:
                width, height = self._dim_cache[img_key]
            else:
                img = cv2.imread(str(img_path))
                if img is None:
                    return
                height, width = img.shape[:2]
                self._dim_cache[img_key] = (width, height)
            
            # Vectorized conversion
            lines = []
            
            for class_id, key in [(0, 'signatures'), (1, 'stamps')]:
                bboxes = label_data.get(key, [])
                if not bboxes:
                    continue
                
                # Batch process bounding boxes
                bboxes_array = np.array(bboxes)
                x_centers = ((bboxes_array[:, 0] + bboxes_array[:, 2]) / 2) / width
                y_centers = ((bboxes_array[:, 1] + bboxes_array[:, 3]) / 2) / height
                widths = (bboxes_array[:, 2] - bboxes_array[:, 0]) / width
                heights = (bboxes_array[:, 3] - bboxes_array[:, 1]) / height
                
                for i in range(len(bboxes_array)):
                    lines.append(f"{class_id} {x_centers[i]:.6f} {y_centers[i]:.6f} "
                               f"{widths[i]:.6f} {heights[i]:.6f}")
            
            # Write once
            if lines:
                label_path.write_text('\n'.join(lines))
                
        except Exception as e:
            logger.error(f"Error creating label {label_path}: {e}")
    
    @staticmethod
    def _create_yaml(dataset_dir: Path, train_count: int, val_count: int) -> Path:
        """Create YAML configuration"""
        yaml_content = f"""path: {dataset_dir.resolve()}
train: images/train
val: images/val

names:
  0: signature
  1: stamp
"""
        yaml_path = dataset_dir / "dataset.yaml"
        yaml_path.write_text(yaml_content)
        
        logger.info(f"Dataset created: {train_count} train, {val_count} val images")
        return yaml_path
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Optimized preprocessing with minimal copies"""
        # Convert colorspace if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        # In-place contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def detect(self, image: np.ndarray, conf: float = 0.25) -> List[Dict[str, Any]]:
        """Optimized detection with batch processing"""
        try:
            processed_image = self.preprocess_image(image)
            img_height, img_width = processed_image.shape[:2]
            img_area = img_width * img_height
            
            # Run inference with optimizations
            results = self.model.predict(
                processed_image,
                conf=conf,
                iou=self.iou_threshold,
                device=self.device,
                half=self.half_precision,
                verbose=False,
                stream=False  # Faster for single images
            )
            
            detections = []
            
            # Vectorized processing
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                # Vectorized clipping
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_width)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_height)
                
                # Vectorized area calculation
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                
                # Filter by area and confidence
                for i, (box, conf, cls_id, area) in enumerate(zip(boxes, confidences, class_ids, areas)):
                    class_name = self.CLASS_NAMES.get(cls_id, f'class_{cls_id}')
                    
                    if conf < self.conf_thresholds.get(class_name, 0.5):
                        continue
                    
                    if area < 0.0001 * img_area or area > 0.5 * img_area:
                        continue
                    
                    detections.append({
                        'bbox': box.tolist(),
                        'class': class_name,
                        'confidence': float(conf),
                        'area': float(area),
                        'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                    })
            
            # Fast NMS
            return self._fast_nms(detections)
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def _fast_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Optimized NMS using vectorized operations"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Vectorized IoU calculation
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        keep = []
        indices = np.arange(len(detections))
        
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            
            if len(indices) == 1:
                break
            
            # Vectorized IoU
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            intersection = w * h
            union = areas[i] + areas[indices[1:]] - intersection
            iou = intersection / union
            
            # Keep boxes with IoU below threshold
            indices = indices[1:][iou <= iou_threshold]
        
        return [detections[i] for i in keep]
    
    def detect_batch(self, images: List[np.ndarray], batch_size: int = 8) -> List[List[Dict]]:
        """Batch detection for multiple images"""
        all_detections = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Preprocess batch
            processed = [self.preprocess_image(img) for img in batch]
            
            # Run inference on batch
            results = self.model.predict(
                processed,
                conf=0.25,
                iou=self.iou_threshold,
                device=self.device,
                half=self.half_precision,
                verbose=False
            )
            
            # Process each result
            for result in results:
                detections = self._process_result(result)
                all_detections.append(detections)
        
        return all_detections
    
    def _process_result(self, result) -> List[Dict]:
        """Extract detections from a single result"""
        if result.boxes is None or len(result.boxes) == 0:
            return []
        
        detections = []
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            class_name = self.CLASS_NAMES.get(cls_id, f'class_{cls_id}')
            
            if conf >= self.conf_thresholds.get(class_name, 0.5):
                detections.append({
                    'bbox': box.tolist(),
                    'class': class_name,
                    'confidence': float(conf),
                    'center': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                })
        
        return self._fast_nms(detections)
    
    def extract_signature_stamp_info(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract signature and stamp information"""
        detections = self.detect(image)
        
        results = {
            'signature': {'present': False, 'bbox': None, 'confidence': 0.0, 'count': 0},
            'stamp': {'present': False, 'bbox': None, 'confidence': 0.0, 'count': 0},
            'all_detections': detections
        }
        
        # Filter by class
        sig_dets = [d for d in detections if d['class'] == 'signature']
        stamp_dets = [d for d in detections if d['class'] == 'stamp']
        
        if sig_dets:
            best = max(sig_dets, key=lambda x: x['confidence'])
            results['signature'] = {
                'present': True,
                'bbox': best['bbox'],
                'confidence': best['confidence'],
                'count': len(sig_dets)
            }
        
        if stamp_dets:
            best = max(stamp_dets, key=lambda x: x['confidence'])
            results['stamp'] = {
                'present': True,
                'bbox': best['bbox'],
                'confidence': best['confidence'],
                'count': len(stamp_dets)
            }
        
        return results
    
    def train_on_existing_data(self, epochs: int = 50, 
                               batch_size: int = 16,
                               save_path: str = "models/trained_model.pt",
                               augment: bool = True) -> bool:
        """Optimized training with better defaults"""
        try:
            dataset_yaml = self.create_yolo_dataset()
            
            if not dataset_yaml:
                logger.error("Could not create dataset")
                return False
            
            logger.info(f"Training with {dataset_yaml}")
            
            # Train with optimizations
            self.model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=640,
                batch=batch_size,
                device=self.device,
                workers=min(8, mp.cpu_count()),
                save=True,
                save_period=10,
                pretrained=True,
                augment=augment,
                patience=10,  # Early stopping
                cache='ram',  # Cache images in RAM
                amp=self.device == 'cuda'  # Automatic mixed precision
            )
            
            # Save model
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def auto_annotate(self, output_file: str = "auto_annotations.json",
                      conf_threshold: float = 0.7,
                      workers: int = 4) -> Dict:
        """Parallel auto-annotation"""
        if not self.train_data_path:
            logger.error("No training data path")
            return {}
        
        image_files = self._get_image_files(self.train_data_path)
        logger.info(f"Auto-annotating {len(image_files)} images with {workers} workers")
        
        annotations = {}
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._annotate_image, img_path, conf_threshold): img_path 
                      for img_path in image_files}
            
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    if result:
                        img_name, ann_data = result
                        annotations[img_name] = ann_data
                    
                    if (i + 1) % 50 == 0:
                        logger.info(f"Processed {i + 1}/{len(image_files)}")
                        
                except Exception as e:
                    logger.error(f"Error in annotation: {e}")
        
        # Save annotations
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        logger.info(f"Saved {len(annotations)} annotations to {output_file}")
        return annotations
    
    def _annotate_image(self, img_path: Path, conf_threshold: float) -> Optional[Tuple]:
        """Annotate a single image"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = self.detect(img_rgb)
            
            signatures = [d['bbox'] for d in detections 
                         if d['class'] == 'signature' and d['confidence'] > conf_threshold]
            stamps = [d['bbox'] for d in detections 
                     if d['class'] == 'stamp' and d['confidence'] > conf_threshold]
            
            if signatures or stamps:
                return (img_path.stem, {
                    'signatures': signatures,
                    'stamps': stamps,
                    'image_path': str(img_path)
                })
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            return None


# Singleton with lazy initialization
_detector = None

def get_detector(model_path: Optional[str] = None, 
                 train_data_path: Optional[str] = None,
                 device: str = 'cpu',
                 half_precision: bool = False) -> SignatureStampDetector:
    """Get or create detector instance"""
    global _detector
    if _detector is None:
        _detector = SignatureStampDetector(model_path, train_data_path, device, half_precision)
    return _detector


if __name__ == "__main__":
    # Initialize with optimizations
    detector = SignatureStampDetector(
        train_data_path="train",
        device='cuda' if torch.cuda.is_available() else 'cpu',
        half_precision=True
    )
    
    print(f"Training data: train/")
    
    # Auto-annotate with parallel processing
    annotations = detector.auto_annotate("auto_annotations.json", workers=4)
    
    # Create dataset
    dataset_yaml = detector.create_yolo_dataset("auto_annotations.json")
    
    if dataset_yaml:
        print(f"\nDataset created: {dataset_yaml}")
        print("Train with: detector.train_on_existing_data(epochs=50)")