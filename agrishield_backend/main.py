from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
import requests
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io
import hashlib
from datetime import datetime, date
from fastapi.middleware.cors import CORSMiddleware

from database import create_tables, log_diagnosis, log_mrl
from mrl.mrl_engine import calculate_residue, calculate_safe_harvest_time
from diagnose import predict, generate_heatmap
# Assuming find_mrl is still available to look up the half-life and MRL automatically
from mrl.mrl_lookup import find_mrl

app = FastAPI(title="Plant Disease Diagnose")
create_tables()
app = FastAPI(title="Plant Disease Diagnose")

# Add this entire block to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS
    allow_headers=["*"],  # Allows all headers
)

create_tables() # This should already be here

class LocationRequest(BaseModel):
    latitude: float = 12.9716  # Fallback to Bengaluru coordinates if missing
    longitude: float = 77.5946


@app.post("/weather")
def get_weather(loc: LocationRequest):
    # Fetching temperature, humidity, and rain (crucial for pesticide wash-off)
    url = f"https://api.open-meteo.com/v1/forecast?latitude={loc.latitude}&longitude={loc.longitude}&current=temperature_2m,relative_humidity_2m,precipitation&timezone=auto"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            current = data.get("current", {})
            return {
                "status": "success",
                "temperature_c": current.get("temperature_2m"),
                "humidity_percent": current.get("relative_humidity_2m"),
                "precipitation_mm": current.get("precipitation"),
                "message": "Weather data fetched successfully."
            }
        return {"status": "error", "explanation": "Failed to fetch from Open-Meteo"}
    except Exception as e:
        return {"status": "error", "explanation": str(e)}
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
            <br>
            <a href="/docs">Open API Testing</a>
        </body>
    </html>
    """

class ApplicationEvent(BaseModel):
    crop: str
    pesticide: str
    spray_date: str
    destination: Optional[str] = "Domestic"
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    contents = await file.read()
    image_hash = hashlib.sha256(contents).hexdigest()
    timestamp = datetime.now().isoformat()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Adjusted to match your teammate's updated predict function
    label, confidence, class_idx = predict(image)

    log_diagnosis(image_hash, label, float(confidence), timestamp)
    heatmap_b64 = generate_heatmap(image, class_idx)

    return JSONResponse({
        "disease": label.replace("___", " - ").replace("_", " "),
        "confidence_percent": round(confidence * 100, 2),
        "heatmap_base64": heatmap_b64
    })


@app.post("/mrl-risk")
def mrl_risk(event: ApplicationEvent):
    mrl_data = find_mrl(event.crop, event.pesticide)
    if not mrl_data:
        return {"status": "HOLD", "explanation": "Missing MRL data for this crop/pesticide."}

    mrl_limit = mrl_data["mrl_mg_per_kg"]
    dt50 = mrl_data["half_life_days"]
    c0 = 2.5  # Label default baseline

    d_spray = datetime.strptime(event.spray_date, "%Y-%m-%d")
    days_elapsed = (datetime.now() - d_spray).days
    if days_elapsed < 0:
        return {"status": "ERROR", "explanation": "Spray date cannot be in the future."}

    # Weather Integration: Rain Wash-off calculation
    weather_modifier = 1.0
    weather_note = "No location provided; using standard decay."

    if event.latitude and event.longitude:
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={event.latitude}&longitude={event.longitude}&current=precipitation&timezone=auto"
            resp = requests.get(url, timeout=3).json()
            rain_mm = resp.get("current", {}).get("precipitation", 0)

            if rain_mm > 0:
                # Hackathon logic: 15% residue wash-off per mm of rain (capped at 50% reduction)
                wash_off = min(0.50, rain_mm * 0.15)
                weather_modifier = 1.0 - wash_off
                weather_note = f"Rainfall detected ({rain_mm}mm). Applied {int(wash_off * 100)}% wash-off reduction."
            else:
                weather_note = "Clear weather detected at location. Standard decay applied."
        except Exception:
            weather_note = "Weather API timeout; using standard decay."

    # Apply math with the new weather modifier
    effective_c0 = c0 * weather_modifier
    estimated_residue = calculate_residue(effective_c0, dt50, days_elapsed)
    safe_harvest_days = calculate_safe_harvest_time(effective_c0, dt50, mrl_limit)

    safe = estimated_residue <= mrl_limit
    status = "SAFE" if safe else "WAIT"

    timestamp = datetime.now().isoformat()
    log_mrl(
        event.pesticide,
        event.spray_date,
        datetime.now().strftime("%Y-%m-%d"),
        estimated_residue,
        safe,
        timestamp
    )

    return {
        "status": status,
        "estimated_residue_mg_kg": round(estimated_residue, 3),
        "mrl_limit": mrl_limit,
        "days_after_application": days_elapsed,
        "safe_harvest_countdown_days": max(0, round(safe_harvest_days - days_elapsed, 1)),
        "weather_adjustment": weather_note,
        "explanation": "Based on published degradation data; not a certified laboratory measurement."
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
