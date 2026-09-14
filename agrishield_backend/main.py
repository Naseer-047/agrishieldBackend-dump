from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image
import io

from diagnose import predict, generate_heatmap
# from mrl.mrl_assessment import assess_crop_safety

app = FastAPI(title="AgriShield Backend")
from fastapi.middleware.cors import CORSMiddleware

# Add this block to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (perfect for hackathon dev)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>AgriShield</title>
        </head>
        <body style="font-family: Arial; text-align: center; padding: 80px;">
            <h1>🌱 AgriShield</h1>
            <h2>Plant Disease Diagnosis + MRL Safety Check</h2>
            <p>✅ Backend is running successfully</p>
            <p>MobileNetV2 + Grad-CAM &nbsp;|&nbsp; MRL/PHI Assessment Engine</p>
            <br>
            <a href="/docs">Open API Testing</a>
        </body>
    </html>
    """

# Define the expected JSON payload shape
class MRLRequest(BaseModel):
    crop: str
    pesticide: str
    initial_residue: float
    spray_date: str
    target_date: str = None  # Optional

# @app.post("/mrl-check")
# def mrl_check(req: MRLRequest):
#     result = assess_crop_safety(
#         crop=req.crop,
#         pesticide=req.pesticide,
#         initial_residue=req.initial_residue,
#         spray_date=req.spray_date,
#         target_date=req.target_date
#     )
#     return result

import google.generativeai as genai

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
genai.configure(api_key=GEMINI_API_KEY)
vision_model = genai.GenerativeModel('gemini-flash-latest')

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # 1. Verify with Gemini Vision
    try:
        # Resize image for Gemini to make it extremely fast and prevent timeouts
        gemini_image = image.copy()
        gemini_image.thumbnail((512, 512)) # Increase slightly to preserve details
        prompt = "Is this an image of a plant, a leaf, a crop, or a farm? Answer ONLY with the word YES or NO."
        response = vision_model.generate_content([prompt, gemini_image])
        answer = response.text.strip().upper()
        print(f"Gemini Answer: {answer}") # Debugging
        
        # If it clearly says NO (and doesn't say YES)
        if "NO" in answer and "YES" not in answer:
            return JSONResponse(
                status_code=400, 
                content={"error": "not_a_leaf", "message": "Not a plant or leaf."}
            )
    except Exception as e:
        print("Gemini Error:", e)
        pass

    # 2. Run actual ML Model
    label, confidence, class_idx = predict(image)
    heatmap_b64 = generate_heatmap(image, class_idx)

    return JSONResponse({
        "disease": label.replace("___", " - ").replace("_", " "),
        "confidence_percent": round(confidence * 100, 2),
        "heatmap_base64": heatmap_b64
    })

from pydantic import BaseModel
from datetime import datetime, timedelta

class MRLRequest(BaseModel):
    crop: str
    pesticide: str
    spray_date: str
    application_rate: str = "label_default"
    location: str = "Unknown"
    destination: str = "Domestic"
    crop_stage: str = "pre_harvest"

@app.post("/mrl-risk")
async def check_mrl_risk(req: MRLRequest):
    # Mock deterministic logic for the prototype
    
    # 1. Parse dates
    try:
        spray_date = datetime.strptime(req.spray_date, "%d %b %Y")
    except ValueError:
        try:
            spray_date = datetime.strptime(req.spray_date, "%Y-%m-%d")
        except ValueError:
            spray_date = datetime.now() # Fallback

    current_date = datetime.now()
    days_since_spray = (current_date - spray_date).days

    # 2. Mock PHI rules based on pesticide
    pesticide_lower = req.pesticide.lower()
    phi = 5 # Default 5 days
    mrl_limit = 3.0
    
    if "lambda" in pesticide_lower:
        phi = 7
        mrl_limit = 1.0
    elif "copper" in pesticide_lower:
        phi = 3
        mrl_limit = 5.0
    elif "imidacloprid" in pesticide_lower:
        phi = 14
        mrl_limit = 0.5
        
    # Check if unknown pesticide
    if req.pesticide == "Unknown" or not req.pesticide:
        return {
            "status": "HOLD",
            "risk_level": "unknown",
            "estimated_residue_risk": 0.0,
            "mrl_limit": 0.0,
            "safe_harvest_date": "Unknown",
            "phi_remaining_days": 0,
            "confidence": 0.0,
            "explanation": "AgriShield could not verify the required pesticide/MRL information."
        }

    # 3. Calculate Result
    if days_since_spray < 0:
        days_since_spray = 0 # Prevent negative if spray date is future

    if days_since_spray >= phi:
        status = "SAFE"
        risk_level = "low"
        phi_remaining = 0
        safe_harvest_date = current_date.strftime("%d %b %Y")
        explanation = "The recommended waiting period has been satisfied and no known rule conflict is detected."
    else:
        status = "WAIT"
        risk_level = "moderate"
        phi_remaining = phi - days_since_spray
        safe_harvest_date = (current_date + timedelta(days=phi_remaining)).strftime("%d %b %Y")
        explanation = "Your spray was applied recently and the recommended waiting period is still active."

    return {
        "status": status,
        "risk_level": risk_level,
        "estimated_residue_risk": 0.62 if status == "WAIT" else 0.15,
        "mrl_limit": mrl_limit,
        "safe_harvest_date": safe_harvest_date,
        "phi_remaining_days": phi_remaining,
        "confidence": 0.85,
        "explanation": explanation
    }


class MRLCheckRequest(BaseModel):
    crop: str
    pesticide: str
    predicted_residue: float

if __name__ == "__main__":
    import uvicorn
    # This tells PyCharm to actually start the web server on port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


# @app.post("/mrl-check")
# def mrl_check(request: MRLCheckRequest):
#     result = assess_crop_safety(
#         crop=request.crop,
#         pesticide=request.pesticide,
#         predicted_residue=request.predicted_residue
#     )
#     return JSONResponse(result)
