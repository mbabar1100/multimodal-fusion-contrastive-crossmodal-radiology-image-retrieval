Multimodal Fusion and Contrastive Cross-Modal Learning for Radiology Image-Caption Retrieval

This project implements and compares two multimodal learning approaches for radiology image–caption retrieval using the ROCOv2 dataset. The work investigates how cross-modal attention fusion and contrastive dual-tower embeddings perform on medical image–text alignment tasks. It includes full training pipelines, retrieval evaluation with Recall@K, and an analysis of performance, efficiency, and scalability.

📌 Objectives

Build a cross-modal attention fusion model for image–text matching.

Build a contrastive two-tower model that learns shared multimodal embeddings.

Perform radiology image-caption retrieval (image-to-text and text-to-image).

Evaluate using:

Recall@1

Recall@5

Recall@10

Compare:

Retrieval performance

Model efficiency

Architectural differences

Qualitative retrieval behavior

📂 Dataset: ROCOv2

ROCOv2 (Radiology Objects in Context) is a multimodal medical dataset containing:

79,789 radiology images

Medical captions extracted from PubMed articles

Multiple imaging modalities: X-ray, CT, MRI, Ultrasound, etc.

Sources:

HuggingFace: eltorio/ROCOv2-radiology

License: CC BY-NC-SA 4.0

Each sample includes:

image — radiology image

caption — corresponding descriptive caption

concepts — UMLS/MedCAT concepts (optional for this assignment)

🧠 Methods Implemented
1. Cross-Modal Attention Fusion (Part 1)

A fusion model that:

Extracts spatial patch features using ResNet-50

Encodes text using BERT

Applies multi-head cross-attention (Text = Query, Image = Key/Value)

Produces a fused representation for:

Binary match prediction

Pairwise scoring for retrieval

Training: Binary Cross-Entropy with in-batch negative sampling
Retrieval: Full pairwise scoring across the test gallery

2. Contrastive Two-Tower Embeddings (Part 2)

A CLIP-style model with separate encoders:

Image Tower: ResNet-50 + projection MLP

Text Tower: BERT + projection MLP

Outputs are L2-normalized embeddings in a shared space

Trained using InfoNCE contrastive loss

Training: Symmetric image-to-text and text-to-image contrastive loss
Retrieval: Fast matrix multiplication between precomputed galleries

📊 Evaluation

Two retrieval tasks:

Image → Caption Retrieval

Caption → Image Retrieval

Metrics:

Recall@1

Recall@5

Recall@10

Performance differences:

Part 1: Pairwise inference (slow, quadratic)

Part 2: Similarity search via embeddings (fast, scalable)

📁 Repository Structure
code/
│── data_utils.py        # Dataset wrapper and preprocessing
│── models.py            # Cross-modal attention + two-tower models
│── train_part1.py       # Training script for cross-modal fusion model
│── train_part2.py       # Training script for contrastive model
│── eval.py              # Retrieval evaluation and Recall@K
│── utils.py             # Helper utilities
checkpoints/             # Saved model weights (optional)
report.pdf               # Assignment report (to be added)
README.md                # This file

⚙️ Installation
pip install torch torchvision transformers datasets tqdm scikit-learn pillow numpy

▶️ How to Run (Jupyter Example)
1. Load dataset
from datasets import load_dataset
roco = load_dataset("eltorio/ROCOv2-radiology")
train_ds = roco["train"]
valid_ds = roco["validation"]
test_ds  = roco["test"]

2. Train Part 1 (Cross-Modal Attention)
from code.train_part1 import main as train_part1_main
model = train_part1_main(train_ds, epochs=2, batch_size=8, device="cuda")

3. Train Part 2 (Contrastive Two-Tower)
from code.train_part2 import main as train_part2_main
img_tower, txt_tower = train_part2_main(train_ds, epochs=2, batch_size=32, device="cuda")

4. Evaluate Retrieval (Part 2 example)
from code.eval import evaluate_part2
metrics = evaluate_part2(img_tower, txt_tower, test_ds, device="cuda")
print(metrics)

📈 Expected Outputs

Recall@K metrics for both models

Side-by-side retrieval comparison

Top-3 retrieved captions/images for sample queries

Discussion of speed, scalability, and qualitative differences

📘 Report (to be added)

Place your final assignment report here:

report.pdf


The report includes:

Model descriptions

Retrieval metrics tables

Performance comparison

Architectural analysis

Qualitative retrieval examples
