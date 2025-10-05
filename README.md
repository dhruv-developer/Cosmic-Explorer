# Cosmic Explorer — NASA Space Apps Challenge 2025

**Theme:** Embiggen Your Eyes  
**Event:** Delhi Local Event – NASA Space Apps Challenge 2025  

**Team Members:**  
Sneha Roychowdhury, Dhruv Dawar, Shriya Sandilya, Vamika Mendiratta  

**Deployed Application:** [https://cosmic-explorer-frontend.vercel.app/](https://cosmic-explorer-frontend.vercel.app/)  
**Backend Repository:** [https://github.com/dhruv-developer/Cosmic-Explorer](https://github.com/dhruv-developer/Cosmic-Explorer)  
**Frontend Repository:** [https://github.com/dhruv-developer/Cosmic-Explorer-Frontend](https://github.com/dhruv-developer/Cosmic-Explorer-Frontend)

---

## Overview

**Cosmic Explorer** is an AI-powered planetary image exploration system designed to interpret and interact with high-resolution NASA datasets. It integrates vision-language understanding, metadata reasoning, and super-resolution imaging to provide meaningful scientific insights about planetary surfaces like Mars. The platform allows users to zoom into massive spatial images, detect geological features, and ask natural language questions — bridging the gap between machine perception and human curiosity.

---

## Objective

To make planetary image exploration intuitive, explainable, and accessible by enabling scientists, engineers, and space enthusiasts to interact with trillion-pixel NASA imagery through AI-powered tools that combine visual analysis and natural language reasoning.

---

## System Architecture

### 1. Vision Module (BLIP)
- Performs image captioning and semantic understanding
- Input: Planetary images (RGB/multispectral)
- Output: Descriptive summaries such as "A Mars surface image with a crater and ice patch"

### 2. Metadata Module
- Processes structured feature metadata: coordinates, radius, spectral bands, and detection confidence
- Converts metadata into natural language context to complement visual understanding

### 3. Zoom and Super-Resolution Module
- Enables region-of-interest (ROI) selection and enhancement
- Produces high-resolution tiles for detailed inspection using super-resolution models

### 4. Fusion and Reasoning Module (Flan-T5)
- Fuses visual captions, metadata, and optional user queries
- Generates scientific interpretations, 3D reconstruction hints, and structured JSON outputs for further analysis

---

## Tech Stack

| Component | Tools and Frameworks |
|-----------|----------------------|
| **Frontend** | React.js, Next.js, TailwindCSS, Vercel |
| **Backend** | Python, FastAPI, PyTorch, Hugging Face Transformers |
| **Models** | BLIP (Vision-Language), Flan-T5 (Text Reasoning), ESRGAN (Super-Resolution) |
| **Libraries** | OpenCV, NumPy, Matplotlib, torchvision |
| **Data** | NASA Planetary Image Datasets / Dummy Mars Dataset |
| **Deployment** | Vercel (Frontend), Local/Cloud (Backend) |

---

## Installation and Setup

### Backend
```bash
git clone https://github.com/dhruv-developer/Cosmic-Explorer.git
cd Cosmic-Explorer
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
git clone https://github.com/dhruv-developer/Cosmic-Explorer-Frontend.git
cd Cosmic-Explorer-Frontend
npm install
npm run dev
```

---

## Features

- Interactive zoom and feature-based exploration of planetary imagery
- Automated detection of craters, ice patches, and surface anomalies
- Scientific caption generation using vision-language models
- Natural language querying for feature-specific insights
- JSON output for downstream AI and visualization pipelines
- Super-resolution module for enhanced spatial analysis

---

## Example Output

**Input:** Mars surface image containing craters and ice regions

**Output:**
```json
{
  "semantic_description": "The image shows a large crater with an adjacent ice patch near the southern ridge.",
  "3D_inference_tags": ["crater", "ice_patch"],
  "provenance_notes": "Metadata and image fusion"
}
```

---

## Creativity and Impact

Cosmic Explorer redefines how we perceive and interact with planetary data. It transforms static imagery into an intelligent, conversational experience — where users can explore, interpret, and learn from vast spatial datasets. By merging visual AI and scientific reasoning, it empowers researchers, educators, and the public to uncover new insights about our universe, making space data more accessible and engaging.
