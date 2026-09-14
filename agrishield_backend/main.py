from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from PIL import Image
import io

from diagnose import predict, generate_heatmap
from mrl.mrl_assessment import assess_crop_safety

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

@app.post("/mrl-check")
def mrl_check(req: MRLRequest):
    result = assess_crop_safety(
        crop=req.crop,
        pesticide=req.pesticide,
        initial_residue=req.initial_residue,
        spray_date=req.spray_date,
        target_date=req.target_date
    )
    return result

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    label, confidence, class_idx = predict(image)
    heatmap_b64 = generate_heatmap(image, class_idx)

    return JSONResponse({
        "disease": label.replace("___", " - ").replace("_", " "),
        "confidence_percent": round(confidence * 100, 2),
        "heatmap_base64": heatmap_b64
    })


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
