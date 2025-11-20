# Multi-Head DiffNet++ Recommendation System

A socially-aware product recommendation system using multi-head attention and neural diffusion.

## Features
- Multi-head attention mechanism (4 heads)
- Social influence diffusion
- Dual output (rating + helpfulness prediction)
- Interactive Streamlit UI

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python quick_start.py --mode train

# Run UI
streamlit run app.py
```

## Architecture
- 64-dimensional embeddings
- 2-layer diffusion process
- 1M+ trainable parameters