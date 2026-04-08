import gradio as gr
import requests

API_URL = "http://127.0.0.1:7860/analyze_skin"

def analyze_skin(image):
    if image is None:
        return "Please upload an image.", "", "", "", ""
    
    # Convert PIL image to bytes
    import io
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    try:
        response = requests.post(
            API_URL,
            files={"file": ("image.jpg", img_bytes, "image/jpeg")},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        disease     = data.get("disease", "N/A")
        confidence  = data.get("confidence_pct", "N/A")
        recs        = data.get("recommendations", "N/A")
        steps       = data.get("next_steps", "N/A")
        tips        = data.get("tips", "N/A")

        # Format top 5
        top5 = data.get("top5_predictions", [])
        top5_text = "\n".join(
            [f"{i+1}. {p['disease']}  —  {p['confidence']*100:.2f}%"
             for i, p in enumerate(top5)]
        )

        return disease, confidence, top5_text, recs, steps, tips

    except Exception as e:
        return f"Error: {e}", "", "", "", "", ""


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
        fn=analyze_skin,
        inputs=[image_input],
        outputs=[disease_out, confidence_out, top5_out, recs_out, steps_out, tips_out]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860)