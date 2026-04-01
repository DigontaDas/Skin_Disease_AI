from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io

from model import load_model, predict
from llm import get_recommendations
from preprocess import preprocess_image

app = FastAPI(
    title="Skin Disease Detection API",
    description="CNN + LLM powered skin disease detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
model, class_names, device = load_model()


@app.get("/")
def root():
    return {"message": "Skin Disease Detection API is running ✅"}


@app.get("/classes")
def get_classes():
    return {"classes": class_names, "total": len(class_names)}


@app.post("/analyze_skin")
async def analyze_skin(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted.")

    image_bytes = await file.read()

    try:
        # Step 1: Preprocess
        image_tensor = preprocess_image(image_bytes)

        # Step 2: CNN Prediction
        disease, confidence, top5 = predict(image_tensor, model, class_names, device)

        # Step 3: LLM Recommendations
        recommendations = get_recommendations(disease, confidence)

        return {
            "disease"        : disease,
            "confidence"     : round(confidence, 4),
            "confidence_pct" : f"{confidence * 100:.2f}%",
            "top5_predictions": top5,
            "recommendations": recommendations["recommendations"],
            "next_steps"     : recommendations["next_steps"],
            "tips"           : recommendations["tips"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)