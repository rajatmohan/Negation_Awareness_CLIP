"""Evaluation metrics and utilities"""

import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def evaluate_pairwise_preference(model_adapter, dataset_loader, preprocess, device, max_samples=None):
    """
    Evaluate model on pairwise image preference task.
    Score = 1 if sim(text, pos_image) > sim(text, neg_image)
    
    Args:
        model_adapter: PairwiseModelAdapter instance
        dataset_loader: DataLoader with 'text'/'prompt' and image keys
        preprocess: Image preprocessing function
        device: 'cuda' or 'cpu'
        max_samples: Limit samples for quick eval
    
    Returns:
        dict with 'model', 'correct', 'total', 'avg_score'
    """
    model_adapter.model.eval()

    total = 0
    correct = 0

    for batch in tqdm(dataset_loader, desc=f"Evaluating {model_adapter.name}"):
        # Find correct keys
        prompt_key = next((k for k in ["prompt", "promt", "phrase", "text"] if k in batch), None)
        pos_key = next((k for k in ["pos_image", "positive_image", "image_pos"] if k in batch), None)
        neg_key = next((k for k in ["neg_image", "negative_image", "image_neg"] if k in batch), None)
        
        if not all([prompt_key, pos_key, neg_key]):
            raise KeyError(f"Could not find required keys in batch. Found: {list(batch.keys())}")

        prompts = batch[prompt_key]
        batch_size = len(prompts) if isinstance(prompts, (list, tuple)) else 1

        for i in range(batch_size):
            # Extract individual items
            if isinstance(prompts, (list, tuple)):
                prompt = prompts[i]
            else:
                prompt = prompts
            
            pos_image = batch[pos_key][i] if isinstance(batch[pos_key], (list, torch.Tensor)) and batch[pos_key].shape[0] > 1 else batch[pos_key]
            neg_image = batch[neg_key][i] if isinstance(batch[neg_key], (list, torch.Tensor)) and batch[neg_key].shape[0] > 1 else batch[neg_key]

            # Convert to tensors
            if isinstance(pos_image, str):
                pos_image = Image.open(pos_image).convert("RGB")
                pos_image = preprocess(pos_image).unsqueeze(0).to(device)
            elif isinstance(pos_image, Path):
                pos_image = Image.open(str(pos_image)).convert("RGB")
                pos_image = preprocess(pos_image).unsqueeze(0).to(device)
            elif isinstance(pos_image, torch.Tensor):
                if pos_image.dim() == 3:
                    pos_image = pos_image.unsqueeze(0)
                pos_image = pos_image.to(device)

            if isinstance(neg_image, str):
                neg_image = Image.open(neg_image).convert("RGB")
                neg_image = preprocess(neg_image).unsqueeze(0).to(device)
            elif isinstance(neg_image, Path):
                neg_image = Image.open(str(neg_image)).convert("RGB")
                neg_image = preprocess(neg_image).unsqueeze(0).to(device)
            elif isinstance(neg_image, torch.Tensor):
                if neg_image.dim() == 3:
                    neg_image = neg_image.unsqueeze(0)
                neg_image = neg_image.to(device)

            # Encode and compute similarity
            txt = model_adapter.encode_text(prompt)
            pos = model_adapter.encode_image(pos_image)
            neg = model_adapter.encode_image(neg_image)

            pos_sim = (pos @ txt.T).item()
            neg_sim = (neg @ txt.T).item()

            correct += int(pos_sim > neg_sim)
            total += 1

            if max_samples is not None and total >= max_samples:
                avg = correct / max(total, 1)
                return {
                    "model": model_adapter.name,
                    "correct": correct,
                    "total": total,
                    "avg_score": avg,
                }

    avg = correct / max(total, 1)
    return {
        "model": model_adapter.name,
        "correct": correct,
        "total": total,
        "avg_score": avg,
    }


def evaluate_image_text_retrieval(model_adapter, dataset_loader, device, max_samples=None):
    """
    Evaluate model on image-to-text retrieval task (e.g., VALSE dataset).
    Score = 1 if sim(image, pos_text) > sim(image, neg_text)
    
    Args:
        model_adapter: PairwiseModelAdapter instance
        dataset_loader: DataLoader with 'pos_text', 'neg_text', and 'image' keys
        device: 'cuda' or 'cpu'
        max_samples: Limit samples for quick eval
    
    Returns:
        dict with 'model', 'correct', 'total', 'avg_score'
    """
    model_adapter.model.eval()

    total = 0
    correct = 0

    for batch in tqdm(dataset_loader, desc=f"Evaluating {model_adapter.name} (Image-Text Retrieval)"):
        # Find correct keys
        pos_text_key = next((k for k in ["pos_text", "positive_text", "text_pos"] if k in batch), None)
        neg_text_key = next((k for k in ["neg_text", "negative_text", "text_neg"] if k in batch), None)
        image_key = next((k for k in ["image", "img", "image_tensor"] if k in batch), None)
        
        if not all([pos_text_key, neg_text_key, image_key]):
            raise KeyError(f"Could not find required keys in batch. Found: {list(batch.keys())}")

        pos_texts = batch[pos_text_key]
        neg_texts = batch[neg_text_key]
        images = batch[image_key]
        
        batch_size = len(pos_texts) if isinstance(pos_texts, (list, tuple)) else 1

        for i in range(batch_size):
            # Extract individual items
            if isinstance(pos_texts, (list, tuple)):
                pos_text = pos_texts[i]
            else:
                pos_text = pos_texts
            
            if isinstance(neg_texts, (list, tuple)):
                neg_text = neg_texts[i]
            else:
                neg_text = neg_texts
            
            image = images[i] if isinstance(images, (list, torch.Tensor)) and images.shape[0] > 1 else images

            # Ensure image is tensor
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                image = image.to(device)
            else:
                raise TypeError(f"Expected image to be a tensor, got {type(image)}")

            # Encode and compute similarity
            img_embedding = model_adapter.encode_image(image)
            pos_text_embedding = model_adapter.encode_text(pos_text)
            neg_text_embedding = model_adapter.encode_text(neg_text)

            pos_sim = (img_embedding @ pos_text_embedding.T).item()
            neg_sim = (img_embedding @ neg_text_embedding.T).item()

            correct += int(pos_sim > neg_sim)
            total += 1

            if max_samples is not None and total >= max_samples:
                avg = correct / max(total, 1)
                return {
                    "model": model_adapter.name,
                    "correct": correct,
                    "total": total,
                    "avg_score": avg,
                }

    avg = correct / max(total, 1)
    return {
        "model": model_adapter.name,
        "correct": correct,
        "total": total,
        "avg_score": avg,
    }


def evaluate_zero_shot_classification(model_adapter, dataset, device, max_samples=None, prompt_template="This is a photo of {}."):
    """
    Evaluate model on zero-shot image classification task (ImageNet).
    
    For each image:
    1. Get image embedding
    2. For each class: compute text embedding of "This is a photo of <class>"
    3. Find class with highest similarity
    4. Compare predicted class with ground truth
    
    Args:
        model_adapter: PairwiseModelAdapter instance with encode_text and encode_image
        dataset: ImageNet-like dataset with get_all_classnames() and num_classes() methods
        device: 'cuda' or 'cpu'
        max_samples: Limit samples for quick eval (default: None = all)
        prompt_template: Template for class description (default: "This is a photo of {}.")
    
    Returns:
        dict with metrics:
            - 'model': model name
            - 'accuracy': Top-1 accuracy
            - 'correct': number of correct predictions
            - 'total': total samples evaluated
            - 'top5_accuracy': Top-5 accuracy (if applicable)
            - 'per_class_accuracy': dict mapping class names to accuracy
    """
    import torch.nn.functional as F
    
    model_adapter.model.eval()
    
    total = 0
    correct = 0
    top5_correct = 0
    per_class_stats = {}
    
    # Pre-compute text embeddings for all classes
    print(f"  Pre-computing text embeddings for {dataset.num_classes()} classes...")
    classnames = dataset.get_all_classnames()
    text_embeddings = []
    
    with torch.no_grad():
        for class_idx, class_name in enumerate(classnames):
            prompt = prompt_template.format(class_name)
            text_emb = model_adapter.encode_text(prompt)
            text_embeddings.append(text_emb)
            
            # Initialize per-class stats
            per_class_stats[class_name] = {
                'correct': 0,
                'total': 0,
                'accuracy': 0.0
            }
        
        # Stack all text embeddings: (num_classes, embedding_dim)
        text_embeddings = torch.cat(text_embeddings, dim=0).to(device)
    
    # Evaluate on images
    print(f"  Evaluating on images (max {max_samples or len(dataset)} samples)...")
    
    with torch.no_grad():
        for img_idx in tqdm(range(min(len(dataset), max_samples or len(dataset))), 
                           desc=f"Zero-shot evaluation ({model_adapter.name})"):
            sample = dataset[img_idx]
            image_tensor = sample['image'].unsqueeze(0).to(device)
            true_class_idx = sample['class_idx']
            class_name = sample['class_name']
            
            # Get image embedding
            image_emb = model_adapter.encode_image(image_tensor)
            
            # Compute similarity with all class embeddings: (1, num_classes)
            image_emb = F.normalize(image_emb, dim=-1)
            similarities = image_emb @ text_embeddings.T
            similarities = similarities.squeeze(0)  # (num_classes,)
            
            # Get top-1 and top-5 predictions
            top_classes = torch.topk(similarities, k=min(5, dataset.num_classes())).indices
            pred_class_idx = top_classes[0].item()
            
            # Check top-1 accuracy
            if pred_class_idx == true_class_idx:
                correct += 1
            
            # Check top-5 accuracy
            if true_class_idx in top_classes.cpu().numpy():
                top5_correct += 1
            
            # Update per-class stats
            per_class_stats[class_name]['total'] += 1
            if pred_class_idx == true_class_idx:
                per_class_stats[class_name]['correct'] += 1
            
            total += 1
    
    # Compute per-class accuracy
    for class_name in per_class_stats:
        stats = per_class_stats[class_name]
        if stats['total'] > 0:
            stats['accuracy'] = stats['correct'] / stats['total']
    
    # Compute overall metrics
    top1_accuracy = correct / max(total, 1)
    top5_accuracy = top5_correct / max(total, 1)
    
    return {
        "model": model_adapter.name,
        "accuracy": top1_accuracy,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "correct": correct,
        "top5_correct": top5_correct,
        "total": total,
        "per_class_accuracy": per_class_stats,
    }


def evaluate_zero_shot_classification_with_cache(model_adapter, dataset, device, cache_file=None, max_samples=None, prompt_template="This is a photo of {}."):
    """
    Zero-shot classification with optional caching of text embeddings.
    Useful for evaluating multiple models on same dataset without recomputing text embeddings.
    
    Args:
        model_adapter: PairwiseModelAdapter instance
        dataset: ImageNet-like dataset
        device: 'cuda' or 'cpu'
        cache_file: Optional path to cache text embeddings (e.g., "text_embeddings_cache.pt")
        max_samples: Limit samples for quick eval
        prompt_template: Template for class description
    
    Returns:
        dict with evaluation metrics (same as evaluate_zero_shot_classification)
    """
    import os
    
    model_adapter.model.eval()
    
    total = 0
    correct = 0
    top5_correct = 0
    per_class_stats = {}
    
    # Compute or load text embeddings
    classnames = dataset.get_all_classnames()
    
    if cache_file and os.path.exists(cache_file):
        print(f"  Loading cached text embeddings from {cache_file}...")
        cache_data = torch.load(cache_file)
        text_embeddings = cache_data['embeddings'].to(device)
        cached_classnames = cache_data['classnames']
        
        # Verify cache matches current dataset
        if cached_classnames != classnames:
            print(f"  ⚠ Cache classnames don't match dataset. Recomputing...")
            text_embeddings = None
    else:
        text_embeddings = None
    
    # Compute text embeddings if not cached
    if text_embeddings is None:
        print(f"  Pre-computing text embeddings for {dataset.num_classes()} classes...")
        text_embeddings = []
        
        with torch.no_grad():
            for class_idx, class_name in enumerate(classnames):
                prompt = prompt_template.format(class_name)
                text_emb = model_adapter.encode_text(prompt)
                text_embeddings.append(text_emb)
            
            text_embeddings = torch.cat(text_embeddings, dim=0).to(device)
        
        # Cache if requested
        if cache_file:
            print(f"  Saving text embeddings to {cache_file}...")
            torch.save({
                'embeddings': text_embeddings.cpu(),
                'classnames': classnames
            }, cache_file)
    
    # Initialize per-class stats
    for class_name in classnames:
        per_class_stats[class_name] = {
            'correct': 0,
            'total': 0,
            'accuracy': 0.0
        }
    
    # Evaluate on images
    print(f"  Evaluating on images (max {max_samples or len(dataset)} samples)...")
    
    with torch.no_grad():
        for img_idx in tqdm(range(min(len(dataset), max_samples or len(dataset))), 
                           desc=f"Zero-shot evaluation ({model_adapter.name})"):
            sample = dataset[img_idx]
            image_tensor = sample['image'].unsqueeze(0).to(device)
            true_class_idx = sample['class_idx']
            class_name = sample['class_name']
            
            # Get image embedding
            image_emb = model_adapter.encode_image(image_tensor)
            
            # Compute similarity with all class embeddings
            similarities = image_emb @ text_embeddings.T
            similarities = similarities.squeeze(0)
            
            # Get top-1 and top-5 predictions
            top_classes = torch.topk(similarities, k=min(5, dataset.num_classes())).indices
            pred_class_idx = top_classes[0].item()
            
            # Check top-1 accuracy
            if pred_class_idx == true_class_idx:
                correct += 1
            
            # Check top-5 accuracy
            if true_class_idx in top_classes.cpu().numpy():
                top5_correct += 1
            
            # Update per-class stats
            per_class_stats[class_name]['total'] += 1
            if pred_class_idx == true_class_idx:
                per_class_stats[class_name]['correct'] += 1
            
            total += 1
    
    # Compute per-class accuracy
    for class_name in per_class_stats:
        stats = per_class_stats[class_name]
        if stats['total'] > 0:
            stats['accuracy'] = stats['correct'] / stats['total']
    
    # Compute overall metrics
    top1_accuracy = correct / max(total, 1)
    top5_accuracy = top5_correct / max(total, 1)
    
    return {
        "model": model_adapter.name,
        "accuracy": top1_accuracy,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "correct": correct,
        "top5_correct": top5_correct,
        "total": total,
        "per_class_accuracy": per_class_stats,
    }


