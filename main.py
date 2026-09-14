from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import io
import hashlib
from datetime import datetime
from database import create_tables, log_diagnosis,log_mrl
from mrl_engine import calculate_residue, calculate_safe_harvest_time

from diagnose import predict, generate_heatmap
from drone.drone_analyze import analyze_drone_image
app = FastAPI(title="Plant Disease Diagnose")
create_tables()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>AgriShield</title>
        </head>
        <body style="font-family: Arial; text-align: center; padding: 80px;">
            <h1>🌱 AgriShield</h1>
            <h2>Plant Disease Diagnosis</h2>
            <p>✅ Backend is running successfully</p>
            <p>MobileNetV2 + Grad-CAM</p>
            <br>
            <a href="/docs">Open API Testing</a>
        </body>
    </html>
    """


@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    contents = await file.read()
    image_hash = hashlib.sha256(contents).hexdigest()
    timestamp = datetime.now().isoformat()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    label, confidence, class_idx, top_predictions = predict(image)
    log_diagnosis(
    image_hash,
    label,
    float(confidence),
    timestamp
)
    heatmap_b64 = generate_heatmap(image, class_idx)

    return JSONResponse({
        "disease": label.replace("___", " - ").replace("_", " "),
        "confidence_percent": round(confidence * 100, 2),
        "top_predictions": top_predictions,
        "heatmap_base64": heatmap_b64
    })
@app.post("/mrl-check")
async def mrl_check(
    pesticide: str,
    application_date: str,
    harvest_date: str,
    c0: float,
    dt50: float,
    mrl: float,
    days: float
):
    estimated_residue = calculate_residue(c0, dt50, days)

    safe_harvest_days = calculate_safe_harvest_time(
        c0, dt50, mrl
    )

    safe = estimated_residue <= mrl

    timestamp = datetime.now().isoformat()

    log_mrl(
        pesticide,
        application_date,
        harvest_date,
        estimated_residue,
        safe,
        timestamp
    )

    return {
        "initial_residue_mg_kg": c0,
        "estimated_residue_mg_kg": round(estimated_residue, 3),
        "mrl_mg_kg": mrl,
        "days_after_application": days,
        "safe": safe,
        "estimated_safe_harvest_days": round(safe_harvest_days, 2),
        "estimated_note": "Based on published degradation data; not a certified laboratory measurement."
    }
@app.post("/drone-analyze")
async def drone_analyze(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = analyze_drone_image(image)

    return result