import os
import json
import requests


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "gemini_api_key")
GEMINI_URL     = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


def build_prompt(disease: str, confidence: float) -> str:
    conf_pct = round(confidence * 100, 1)

    # Clean up class name (remove numbering prefix like "2. Melanoma 15.75k")
    clean_name = disease.split(". ", 1)[-1] if ". " in disease else disease
    # Remove dataset size suffix like "15.75k", "1677", etc.
    import re
    clean_name = re.sub(r'\s*[-–]?\s*[\d.]+k?\s*$', '', clean_name).strip()

    return f"""You are a dermatology assistant AI. A patient uploaded a skin image.

The AI model detected: **{clean_name}** with {conf_pct}% confidence.

Please respond ONLY with a valid JSON object (no markdown, no extra text) in this exact format:
{{
  "recommendations": "2-3 sentences about this condition and general advice",
  "next_steps": "2-3 concrete next steps the patient should take",
  "tips": "2-3 practical daily care tips for managing this condition"
}}

Important:
- Be helpful but always recommend consulting a dermatologist
- Keep each field concise (2-3 sentences max)
- Do NOT use any markdown formatting inside the JSON values
- Return ONLY the JSON object"""


def get_recommendations(disease: str, confidence: float) -> dict:
    """Call Gemini Flash and return structured recommendations."""

    # Fallback in case API key is missing or call fails
    fallback = {
        "recommendations": f"The AI detected {disease}. Please consult a certified dermatologist for proper diagnosis and treatment.",
        "next_steps"     : "1. Take clear photos of the affected area. 2. Book an appointment with a dermatologist. 3. Avoid self-medicating.",
        "tips"           : "Keep the area clean and moisturized. Avoid scratching or irritating the skin. Use sunscreen daily."
    }

    if GEMINI_API_KEY == "PASTE_YOUR_GEMINI_KEY_HERE":
        print("[LLM] No API key set — returning fallback response.")
        return fallback

    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [{"text": build_prompt(disease, confidence)}]
                }
            ],
            "generationConfig": {
                "temperature"    : 0.4,
                "maxOutputTokens": 512
            }
        }

        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=15
        )
        response.raise_for_status()

        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)

        # Validate keys exist
        for key in ["recommendations", "next_steps", "tips"]:
            if key not in result:
                result[key] = fallback[key]

        return result

    except Exception as e:
        print(f"[LLM] Error: {e} — returning fallback.")
        return fallback
