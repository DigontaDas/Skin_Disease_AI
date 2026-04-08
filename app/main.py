from fastapi import FastAPI, HTTPException, File, UploadFile
import gradio as gr
import io
import traceback

# 1. FIXED IMPORTS: Explicitly tell Docker to look inside the 'app' folder
from app.ai_model import load_model, predict
from app.llm import get_recommendations
from app.preprocess import preprocess_image

app = FastAPI(
    title="Skin Disease Detection API",
    description="CNN + LLM powered skin disease detection",
    version="1.0.0"
)

# Load model on startup
model, class_names, device = load_model()

@app.get("/health")
def api_root():
    return {"message": "Skin Disease Detection API is running ✅"}

@app.get("/classes")
def api_get_classes():
    return {"classes": class_names, "total": len(class_names)}

@app.post("/analyze_skin")
async def api_analyze_skin(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)
        disease, confidence, top5 = predict(image_tensor, model, class_names, device)
        recommendations = get_recommendations(disease, confidence)
        return {
            "disease"         : disease,
            "confidence"      : round(confidence, 4),
            "confidence_pct"  : f"{confidence * 100:.2f}%",
            "top5_predictions": top5,
            "recommendations" : recommendations["recommendations"],
            "next_steps"      : recommendations["next_steps"],
            "tips"            : recommendations["tips"]
        }
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def gradio_predict(image):
    if image is None:
        return "Please upload an image.", "", "", "", ""
    
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    
    try:
        image_tensor = preprocess_image(img_bytes.getvalue())
        disease, confidence, top5 = predict(image_tensor, model, class_names, device)
        recommendations = get_recommendations(disease, confidence)
        
        top5_text = "\n".join(
            [f"{i+1}. {p['disease']}  —  {p['confidence']*100:.2f}%"
             for i, p in enumerate(top5)]
        )
        return (disease,
                f"{confidence * 100:.2f}%",
                top5_text,
                recommendations["recommendations"],
                recommendations["next_steps"],
                recommendations["tips"])
    except Exception as e:
        return f"Error: {e}", "", "", "", "", ""


# Gradio UI Definition
with gr.Blocks(title="Skin Disease Detector") as demo:
    gr.Markdown("# 🩺 Skin Disease Detection & LLM Advisor")
    gr.Markdown("Upload a skin image to get AI-powered disease detection and recommendations.")
    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Skin Image")
        with gr.Column():
            disease_out    = gr.Textbox(label="Detected Disease")
            confidence_out = gr.Textbox(label="Confidence")
            top5_out       = gr.Textbox(label="Top 5 Predictions", lines=5)
            
    with gr.Row():
        recs_out  = gr.Textbox(label="Recommendations", lines=3)
        steps_out = gr.Textbox(label="Next Steps", lines=3)
        tips_out  = gr.Textbox(label="Daily Care Tips", lines=3)
        
    analyze_btn = gr.Button("🔍 Analyze", variant="primary")
    
    analyze_btn.click(
        fn=gradio_predict,
        inputs=[image_input],
        outputs=[disease_out, confidence_out, top5_out, recs_out, steps_out, tips_out]
    )

app = gr.mount_gradio_app(app, demo, path="/")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
