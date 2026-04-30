# Negation Awareness CLIP

A comprehensive study on evaluating and improving Vision-Language Models' ability to understand and handle negation in vision-language tasks.

## Project Overview

This project explores negation awareness in CLIP (Contrastive Language-Image Pre-training) models through multiple evaluation experiments. The work includes evaluation of 8 models on 3 datasets - NegRefCOCOg, VALSE and TinyImage Net


## Project Structure

```
Negation_Awareness_CLIP/
├── extract_embeddings.ipynb             # Extract text embeddings (run first)
├── experiments.ipynb                    # Main experiments (run after extract_embeddings)
├── final.ipynb                          # Final analysis 
├── src/                                 # Core utilities and modules
│   ├── data.py                         # Dataset loading and processing
│   ├── features.py                     # Feature extraction and caching
│   ├── llm.py                          # LLM clients (Qwen) for text processing
│   ├── models.py                       # Model definitions (CLIP variants, DEO, NegationSteeredCLIP)
│   ├── training.py                     # Training utilities for negation classifiers
│   ├── evaluation.py                   # Evaluation metrics and adapters
│   └── experiments.py                  # Experimental pipelines
├── data/                               # All dataset files (location for moved datasets)
│   ├── NegRefCOCOg.json               # NegRefCOCOg dataset annotations
│   ├── existence.json                  # VALSE dataset for negation evaluation
│   ├── COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv
│   ├── negationclip_ViT-B32.pth       # NegationCLIP pretrained weights
│   ├── learned_vectors/               # Learned negation direction vectors
│   ├── embeddings_cache/              # Cached image/text embeddings
│   ├── imagenet/                      # ImageNet/Tiny ImageNet dataset
│   └── val2017/                       # COCO validation images
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

### Prerequisites

- Python 3.8+
- PyTorch with CUDA support
- CLIP, Transformers, and other dependencies (see requirements.txt)

### Installation

```bash
# Clone the repository
cd Negation_Awareness_CLIP

# Install dependencies
pip install -r requirements.txt

```

### Dataset Setup

All datasets should be placed in the `data/` directory:

```
data/
├── NegRefCOCOg.json
├── existence.json
├── COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv
├── negationclip_ViT-B32.pth
├── learned_vectors/
├── imagenet/
└── val2017/
```

## Running Experiments

### Main Experiments (experiments.ipynb)

The main notebook contains three core experiments:

1. **Experiment 1: NegRefCOCOg**
   - Evaluates ability to distinguish positive from negative images
   - Metric: Binary preference accuracy

2. **Experiment 2: VALSE**
   - Single image, 2 captions - match with correct caption
   - Metric: Binary preference accuracy

3. **Experiment 3: Zero-Shot Classification (Tiny ImageNet)**
   - Tests models on zero-shot image classification
   - Metrics: Top-1 and Top-5 accuracy

#### How to Run:

**Step 1: Extract Text Embeddings**

First, run the text embedding extraction notebook:

```python
# Run in Jupyter/IPython
jupyter notebook extract_embeddings.ipynb

# This notebook:
# - Extracts CLIP text embeddings for all queries in the datasets
# - Caches embeddings for faster experiment execution
# - Prepares data for the main experiments
# - Estimated time: 30-60 minutes
```

**Step 2: Run Experiments**

After embeddings are extracted, run the main experiments:

```python
# Run in Jupyter/IPython
jupyter notebook experiments.ipynb

# This notebook contains three core experiments:
# - Execute cells sequentially
# - Cell 1: Setup and imports
# - Cell 2: Define helper functions (get_models, etc.)
# - Cell 3+: Run individual experiments or all together
# - Estimated time: 2-8 hours depending on dataset size and hardware
```



## Key Files Description

### extract_embeddings.ipynb (Main - Run First)
- **Purpose**: Extract and cache CLIP text embeddings for all dataset queries
- **Inputs**: Dataset files (NegRefCOCOg.json, existence.json, etc.)
- **Outputs**: Cached text embeddings in data/embeddings_cache/
- **Note**: Must be executed before experiments.ipynb

### experiments.ipynb (Main - Run After extract_embeddings.ipynb)
- **Purpose**: Run all three core experiments
- **Inputs**: Dataset files, extracted embeddings from extract_embeddings.ipynb, learned vectors, pretrained checkpoints
- **Outputs**: Experiment results, cached image embeddings
- **Note**: Depends on embeddings extracted by extract_embeddings.ipynb

### src/ Modules

## Cached Features and Large Data

Due to size constraints, the following pre-computed features and cached datasets are hosted on Google Drive:

- **Cached Image Embeddings** (CLIP embeddings for all images in COCO val2017, ImageNet)
- **Text Decompositions** (Query decompositions for all datasets)
- **Pretrained Model Weights** (NegationCLIP checkpoints, learned vectors)
- **Evaluation Results** (Intermediate computation caches)

### Download Link
**[Google Drive Folder]** PLACEHOLDER_FOR_DRIVE_LINK

*To be added: Insert the Google Drive link here for accessing cached features and decompositions*

### How to Use Cached Features

1. Download the cache files from the Google Drive link
2. Extract to the `/embeddings_cache/` directory
3. The notebooks will automatically detect and use cached embeddings

