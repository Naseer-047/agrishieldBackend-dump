from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image
import io

from diagnose import predict, generate_heatmap
# from mrl.mrl_assessment import assess_crop_safety

app = FastAPI(title="AgriShield Backend")
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os

# Add this block to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (perfect for hackathon dev)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- MongoDB Integration ---
# Provide default connection string if not found in env
MONGO_URI = os.environ.get("MONGO_URI", "mongodb+srv://g486822_db_user:Naseer@cluster0.x8v75pd.mongodb.net/")

@app.on_event("startup")
async def startup_db_client():
    print(f"Connecting to MongoDB...")
    app.mongodb_client = AsyncIOMotorClient(MONGO_URI)
    app.database = app.mongodb_client.get_database("agrishield_db")
    print("Connected to MongoDB!")

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()
    print("Closed MongoDB connection.")

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
import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
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

    response_data = {
        "disease": label.replace("___", " - ").replace("_", " "),
        "confidence_percent": round(confidence * 100, 2),
        "heatmap_base64": heatmap_b64
    }

    # Save to MongoDB
    try:
        if hasattr(app, "database"):
            disease_collection = app.database.get_collection("disease_reports")
            # Don't save the huge heatmap to the DB for space reasons
            db_record = response_data.copy()
            db_record["heatmap_base64"] = None
            db_record["created_at"] = datetime.now().isoformat()
            await disease_collection.insert_one(db_record)
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")

    return JSONResponse(response_data)

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

from mrl.mrl_assessment import assess_crop_safety
import math

@app.post("/mrl-risk")
async def check_mrl_risk(req: MRLRequest):
    # 1. Parse dates to YYYY-MM-DD
    try:
        spray_date = datetime.strptime(req.spray_date, "%d %b %Y").strftime("%Y-%m-%d")
    except ValueError:
        try:
            spray_date = datetime.strptime(req.spray_date, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            spray_date = datetime.now().strftime("%Y-%m-%d") # Fallback

    # 2. Assume a default initial residue since farmer shouldn't enter this
    default_initial_residue = 2.0 
    
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

    # 3. Call Pavan's real logic
    assessment = assess_crop_safety(
        crop=req.crop,
        pesticide=req.pesticide,
        initial_residue=default_initial_residue,
        spray_date=spray_date
    )

    # Handle errors/unknowns from Pavan's logic
    if assessment.get("status") in ["UNKNOWN", "ERROR"]:
        return {
            "status": "HOLD",
            "risk_level": "unknown",
            "estimated_residue_risk": 0.0,
            "mrl_limit": 0.0,
            "safe_harvest_date": "Unknown",
            "phi_remaining_days": 0,
            "confidence": 0.0,
            "explanation": assessment.get("message", "AgriShield could not verify the required pesticide/MRL information.")
        }

    # 4. Map Pavan's statuses to our UI statuses
    pavan_status = assessment["status"]
    
    if pavan_status == "SAFE":
        ui_status = "SAFE"
        risk_level = "low"
        phi_remaining = 0
        explanation = "The recommended waiting period has been satisfied and no known rule conflict is detected."
    else:
        # WARNING, NEAR_LIMIT, DANGER -> WAIT
        ui_status = "WAIT"
        risk_level = "moderate" if pavan_status in ["WARNING", "NEAR_LIMIT"] else "high"
        explanation = "Your spray was applied recently and the estimated residue is above the safe limit."

        # Calculate PHI remaining based on half life
        # Using Pavan's mrl_data logic
        from mrl.mrl_lookup import find_mrl
        mrl_data = find_mrl(req.crop, req.pesticide)
        dt50 = mrl_data["half_life_days"]
        current_residue = assessment["predicted_residue_mg_per_kg"]
        mrl = mrl_data["mrl_mg_per_kg"]
        
        if current_residue <= mrl:
            phi_remaining = 0
            ui_status = "SAFE"
        else:
            k = math.log(2) / dt50
            safe_days = math.log(current_residue / mrl) / k
            phi_remaining = math.ceil(safe_days)

    current_date = datetime.now()
    safe_harvest_date = (current_date + timedelta(days=phi_remaining)).strftime("%d %b %Y")

    response_data = {
        "status": ui_status,
        "risk_level": risk_level,
        "estimated_residue_risk": assessment["percentage_of_mrl"] / 100.0,
        "mrl_limit": assessment["mrl_mg_per_kg"],
        "safe_harvest_date": safe_harvest_date,
        "phi_remaining_days": phi_remaining,
        "confidence": 0.85,
        "explanation": explanation,
        "crop": req.crop,
        "pesticide": req.pesticide,
        "spray_date": spray_date,
        "created_at": current_date.isoformat()
    }
    
    # Save to MongoDB
    try:
        if hasattr(app, "database"):
            mrl_collection = app.database.get_collection("mrl_reports")
            await mrl_collection.insert_one(response_data.copy())
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")

    return response_data


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
