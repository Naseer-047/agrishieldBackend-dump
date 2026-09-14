from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
import io

from diagnose import predict, generate_heatmap

app = FastAPI(title="Plant Disease Diagnose")


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
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    label, confidence, class_idx, top_predictions = predict(image)
    heatmap_b64 = generate_heatmap(image, class_idx)

    return JSONResponse({
        "disease": label.replace("___", " - ").replace("_", " "),
        "confidence_percent": round(confidence * 100, 2),
        "top_predictions": top_predictions,
        "heatmap_base64": heatmap_b64
    })