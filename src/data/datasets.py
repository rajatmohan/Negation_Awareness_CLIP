"""
Dataset classes for negation-aware CLIP tasks.
Handles loading and preprocessing of different annotation formats.
"""

import json
import os
import random
import csv
import ast
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch


class NegationJSONDataset(Dataset):
    """
    Parses the NegationCLIP annotations JSON.
    Format: {"annotations": [{"caption": str, "updated_caption": str, "image_id": str}]}
    """
    def __init__(self, json_path, max_samples=None, shuffle=True, seed=42):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.items = data['annotations']
        
        if shuffle:
            random.seed(seed)
            random.shuffle(self.items)
        
        if max_samples is not None:
            self.items = self.items[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "pos_text": item['caption'],
            "neg_text": item['updated_caption'],
            **({"image_id": item['image_id']} if "image_id" in item else {})
        }


class COCOValLlamaDataset(Dataset):
    """
    Parses COCO val annotations from LLM rephrasing.
    Format: [{"caption_0": str, "caption_1": str, "image_path": str, ...}]
    Supports both list and dict formats.
    """
    def __init__(self, json_path, max_samples=None, shuffle=True, seed=42):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle both dict and list formats
        self.items = data if isinstance(data, list) else list(data.values())
        
        if shuffle:
            random.seed(seed)
            random.shuffle(self.items)
        
        if max_samples is not None:
            self.items = self.items[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "pos_text": item['caption_0'],
            "neg_text": item['caption_1'],
            "image_id": item['image_path']
        }


class NegRefCOCOgDataset(Dataset):
    """
    Parses NegRefCOCOg annotations and loads cropped images.
    Format: [{"phrase": str, "image": str, "ref_bbox": list, "bbox_list": list}]
    Provides: text (phrase), positive_image (ref_bbox crop), negative_image (alternative bbox crop)
    """
    def __init__(self, json_path, images_dir="./train2014", max_samples=None, shuffle=True, seed=42):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        self.items = data if isinstance(data, list) else list(data.values())
        self.images_dir = images_dir
        
        if shuffle:
            random.seed(seed)
            random.shuffle(self.items)
        
        if max_samples is not None:
            self.items = self.items[:max_samples]
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.items)
    
    def _crop_bbox(self, image, bbox):
        """Crop image using bounding box [x_min, y_min, width, height]"""
        if bbox is None or len(bbox) < 4:
            return image
        
        x_min, y_min, width, height = bbox
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_min + width), int(y_min + height)

        # Ensure bbox is within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.width, x_max)
        y_max = min(image.height, y_max)
        
        return image.crop((x_min, y_min, x_max, y_max))

    def __getitem__(self, idx):
        item = self.items[idx]
        text = item['phrase']
        image_filename = item['image']
        ref_bbox = item.get('ref_bbox', None)
        bbox_list = item.get('bbox_list', [])
        
        # Load the full image
        image_path = os.path.join(self.images_dir, image_filename)
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}")
            # Return dummy tensors if image not found
            dummy_tensor = torch.zeros(3, 224, 224)
            return {
                "text": text,
                "positive_image": dummy_tensor,
                "negative_image": dummy_tensor
            }
        
        # Crop using ref_bbox (reference/positive region)
        positive_image = self._crop_bbox(image, ref_bbox)
        positive_image_tensor = self.transform(positive_image)

        # Select alternative (negative) bbox
        neg_image_idx = 1 if len(bbox_list) > 1 and bbox_list[1] != ref_bbox else 0
        negative_image = self._crop_bbox(image, bbox_list[neg_image_idx] if neg_image_idx < len(bbox_list) else None)
        negative_image_tensor = self.transform(negative_image)
        
        return {
            "text": text,
            "positive_image": positive_image_tensor,
            "negative_image": negative_image_tensor
        }


class VALSEDataset(Dataset):
    """
    Parses VALSE annotations JSON (Visual Semantic Evaluation).
    Format: {id: {"caption": str, "image_file": str, "foil": str}}
    Returns image tensor along with positive and negative texts.
    """
    def __init__(self, json_path, images_dir="./images", max_samples=None, shuffle=True, seed=42):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert dict to list format
        self.items = list(data.values())
        self.images_dir = images_dir
        
        if shuffle:
            random.seed(seed)
            random.shuffle(self.items)
        
        if max_samples is not None:
            self.items = self.items[:max_samples]
        
        # Image preprocessing (same as NegRefCOCOgDataset)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Load image
        image_path = os.path.join(self.images_dir, item['image_file'])
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}")
            # Return dummy tensor if image not found
            dummy_tensor = torch.zeros(3, 224, 224)
            return {
                "pos_text": item['caption'],
                "neg_text": item['foil'],
                "image": dummy_tensor
            }
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        return {
            "pos_text": item['caption'],
            "neg_text": item['foil'],
            "image": image_tensor
        }


class NegatedRetrievalCSVDataset(Dataset):
    """
    Parses negated image retrieval CSV dataset.
    Format: CSV with columns: positive_objects, negative_objects, filepath, image_id, captions
    Each row contains:
    - positive_objects: List of objects present in image
    - negative_objects: List of objects NOT present in image
    - filepath: Path to image file
    - image_id: COCO image ID
    - captions: 5 negation-aware captions for the image
    
    Returns for each image:
    - image_id: COCO image ID (ground truth)
    - filepath: Path to image
    - positive_objects: Objects present
    - negative_objects: Objects absent
    - captions: List of 5 captions (queries)
    """
    def __init__(self, csv_path, max_samples=None, shuffle=True, seed=42):
        self.items = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse list columns
                try:
                    pos_objs = ast.literal_eval(row['positive_objects']) if row['positive_objects'] else []
                    neg_objs = ast.literal_eval(row['negative_objects']) if row['negative_objects'] else []
                    captions = ast.literal_eval(row['captions']) if row['captions'] else []
                except:
                    continue
                
                self.items.append({
                    'image_id': row['image_id'],
                    'filepath': row['filepath'],
                    'positive_objects': pos_objs,
                    'negative_objects': neg_objs,
                    'captions': captions
                })
        
        if shuffle:
            random.seed(seed)
            random.shuffle(self.items)
        
        if max_samples is not None:
            self.items = self.items[:max_samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            'image_id': item['image_id'],
            'filepath': item['filepath'],
            'positive_objects': item['positive_objects'],
            'negative_objects': item['negative_objects'],
            'captions': item['captions']
        }


class ImageNetDataset(Dataset):
    """
    ImageNet dataset supporting multiple structures:
    
    1. Tiny ImageNet (200 classes, 10,000 val images):
        imagenet_root/
            tiny-imagenet-200/
                val/
                    images/              (all images: val_0.JPEG, val_1.JPEG, ...)
                    val_annotations.txt  (filename\twnid\tx1\ty1\tx2\ty2)
                wnids.txt               (one class wnid per line)
                words.txt               (wnid\tclass_name)
    
    2. Full ImageNet (1000 classes):
        imagenet_root/
            val/
                n01440764/              (class folder with wnid)
                    images...
                n01443537/
                    images...
    
    Automatically detects which structure is used.
    """
    def __init__(self, root_dir, split='val', max_samples=None, shuffle=True, seed=42):
        """
        Args:
            root_dir: Path to ImageNet root directory
            split: 'val' or 'train' (default: 'val')
            max_samples: Limit number of samples (default: None = all)
            shuffle: Whether to shuffle samples (default: True)
            seed: Random seed for shuffling (default: 42)
        """
        self.root_dir = root_dir
        self.split = split
        self.max_samples = max_samples
        self.seed = seed
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        # Detect dataset type and load accordingly
        self._detect_and_load_dataset()
        
        if shuffle:
            random.seed(seed)
            random.shuffle(self.items)
        
        if max_samples is not None:
            self.items = self.items[:max_samples]
    
    def _detect_and_load_dataset(self):
        """Detect ImageNet structure (Tiny or Full) and load accordingly"""
        # Check for Tiny ImageNet structure
        tiny_imagenet_path = os.path.join(self.root_dir, 'tiny-imagenet-200')
        if os.path.exists(tiny_imagenet_path):
            print("✓ Detected Tiny ImageNet structure")
            self._load_tiny_imagenet(tiny_imagenet_path)
        else:
            # Fall back to Full ImageNet structure
            print("✓ Detected Full ImageNet structure")
            self._load_full_imagenet()
    
    def _load_tiny_imagenet(self, tiny_imagenet_path):
        """Load Tiny ImageNet (200 classes)"""
        self.items = []
        
        # Load wnid to class name mapping
        wnids_path = os.path.join(tiny_imagenet_path, 'wnids.txt')
        words_path = os.path.join(tiny_imagenet_path, 'words.txt')
        
        # Load wnids in order
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        self.idx_to_wnid = {idx: wnid for wnid, idx in self.wnid_to_idx.items()}
        
        # Load words (wnid -> class name)
        self.wnid_to_name = {}
        if os.path.exists(words_path):
            with open(words_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t', 1)
                    if len(parts) == 2:
                        wnid, name = parts
                        self.wnid_to_name[wnid] = name.split(',')[0]  # Take first name only
        
        # Load annotations
        split_dir = os.path.join(tiny_imagenet_path, self.split)
        annotations_path = os.path.join(split_dir, f'{self.split}_annotations.txt')
        images_dir = os.path.join(split_dir, 'images')
        
        if not os.path.exists(annotations_path):
            raise ValueError(f"Annotations file not found: {annotations_path}")
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        
        # Parse annotations: filename wnid x1 y1 x2 y2
        with open(annotations_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    filename = parts[0]
                    wnid = parts[1]
                    
                    if wnid in self.wnid_to_idx:
                        image_path = os.path.join(images_dir, filename)
                        class_idx = self.wnid_to_idx[wnid]
                        
                        self.items.append({
                            'image_path': image_path,
                            'class_idx': class_idx,
                            'wnid': wnid,
                            'filename': filename
                        })
        
        print(f"✓ Tiny ImageNet {self.split}: {len(self.items)} images, {len(self.wnid_to_idx)} classes")
    
    def _load_full_imagenet(self):
        """Load Full ImageNet (1000 classes with class folders)"""
        self.items = []
        split_dir = os.path.join(self.root_dir, self.split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Scan for class folders (wnid format: n01440764, etc.)
        class_folders = sorted([d for d in os.listdir(split_dir) 
                               if os.path.isdir(os.path.join(split_dir, d))])
        
        # Build mapping from wnid to class index
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(class_folders)}
        self.idx_to_wnid = {idx: wnid for wnid, idx in self.wnid_to_idx.items()}
        self.wnid_to_name = {}  # Will use wnid as name if not available
        
        # Collect all images
        for wnid in class_folders:
            class_dir = os.path.join(split_dir, wnid)
            class_idx = self.wnid_to_idx[wnid]
            
            for image_file in os.listdir(class_dir):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    image_path = os.path.join(class_dir, image_file)
                    self.items.append({
                        'image_path': image_path,
                        'class_idx': class_idx,
                        'wnid': wnid,
                        'filename': image_file
                    })
        
        print(f"✓ Full ImageNet {self.split}: {len(self.items)} images, {len(class_folders)} classes")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        
        try:
            # Load and preprocess image
            image = Image.open(item['image_path']).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            print(f"Warning: Could not load image {item['image_path']}: {e}")
            # Return dummy tensor
            image_tensor = torch.zeros(3, 224, 224)
        
        # Get class name
        class_idx = item['class_idx']
        wnid = item['wnid']
        
        # Use cached name mapping if available, otherwise use wnid
        if wnid in self.wnid_to_name:
            class_name = self.wnid_to_name[wnid]
        else:
            class_name = wnid
        
        return {
            'image': image_tensor,
            'class_idx': class_idx,
            'wnid': wnid,
            'class_name': class_name,
            'image_path': item['image_path']
        }
    
    def get_classname(self, class_idx):
        """Get class name for a given class index"""
        wnid = self.idx_to_wnid.get(class_idx)
        if wnid and wnid in self.wnid_to_name:
            return self.wnid_to_name[wnid]
        elif wnid:
            return wnid
        else:
            return f"class_{class_idx}"
    
    def get_all_classnames(self):
        """Get all class names (for zero-shot classification)"""
        classnames = []
        for class_idx in range(len(self.wnid_to_idx)):
            classnames.append(self.get_classname(class_idx))
        return classnames
    
    def num_classes(self):
        """Get total number of classes"""
        return len(self.wnid_to_idx)
