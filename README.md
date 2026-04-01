# 🩺 Skin Disease Detection & LLM Advisor System

AI-powered skin disease detection using EfficientNet-B0 + Gemini Flash LLM recommendations.

## Project Structure
```
skin-disease-ai/
├── app/
│   ├── main.py          # FastAPI app
│   ├── model.py         # CNN inference
│   ├── llm.py           # LLM recommendations (Gemini Flash)
│   └── preprocess.py    # Image preprocessing
├── model/
│   ├── efficientnet_skin.pth   # Trained model weights
│   └── class_labels.json       # Class names
├── ui/
│   └── app.py           # Gradio UI
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get Gemini API Key (FREE)
- Go to https://aistudio.google.com/
- Create API Key
- Set it in `app/llm.py` or as environment variable:
```bash
set GEMINI_API_KEY=your_key_here       # Windows
export GEMINI_API_KEY=your_key_here    # Linux/Mac
```

### 3. Run FastAPI backend
```bash
cd app
uvicorn main:app --reload --port 8000
```

### 4. Run Gradio UI (new terminal)
```bash
cd ui
python app.py
```

- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **UI**: http://localhost:7860

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/classes` | List all 10 disease classes |
| POST | `/analyze_skin` | Upload image → get prediction + LLM advice |

### Example Response
```json
{
  "disease": "2. Melanoma 15.75k",
  "confidence": 0.9991,
  "confidence_pct": "99.91%",
  "top5_predictions": [...],
  "recommendations": "...",
  "next_steps": "...",
  "tips": "..."
}
```

## Model Info
- **Architecture**: EfficientNet-B0 (Transfer Learning)
- **Dataset**: Kaggle Skin Diseases Image Dataset (10 classes)
- **Accuracy**: 80.64% validation accuracy
- **Classes**: Eczema, Melanoma, Atopic Dermatitis, BCC, Melanocytic Nevi, BKL, Psoriasis, Seborrheic Keratoses, Tinea Ringworm, Warts
## Model Weights
Download the trained model from [Google Drive](https://drive.google.com/file/d/1IP7YQilO6QKjNbyXS5Iw5GRg82ahNOJw/view?usp=sharing) and place it at:
`model/efficientnet_skin.pth
## Docker
```bash
docker build -t skin-disease-api .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key skin-disease-api
```