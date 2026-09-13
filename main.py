from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from diagnose import predict, generate_heatmap

app = FastAPI(title="Plant Disease Diagnose")

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    label, confidence, class_idx = predict(image)
    heatmap_b64 = generate_heatmap(image, class_idx)
    
    return JSONResponse({
        "class": label,
        "confidence": round(confidence, 4),
        "heatmap_base64": heatmap_b64
    })