import os
import sys
import yaml
import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure our app directory is exactly first in path, and remove the parent directory
# to prevent 'src' namespace collisions on Render (which clones into a folder named 'src')
app_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(app_dir)

if parent_dir in sys.path:
    sys.path.remove(parent_dir)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from src.inference.pipeline import TrafficViolationInference
from src.visualization.visualizer import draw_violations

app = FastAPI(title="AutoViolate-CV API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    print("Loading AutoViolate-CV Models...")
    # Load config
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    pipeline = TrafficViolationInference.from_config(cfg)
    print("Models loaded successfully!")

@app.get("/")
def read_root():
    return {"status": "AutoViolate-CV Backend is Running!"}

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image format"}

    # Run inference
    # Note: Since this is a demo without real trained weights, we might see the synthetic detections
    # But it will run the pipeline perfectly
    result = pipeline.infer(img, image_id=file.filename)
    
    # Draw annotations
    viols = [v.to_dict() for v in result.violations]
    annotated_img = draw_violations(img, viols)
    
    # Convert annotated image to base64 for the frontend
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Calculate total fine
    total_fine = sum(v["fine_inr"] for v in viols)

    return {
        "filename": result.image_id,
        "inference_time_ms": result.inference_ms,
        "total_violations": len(viols),
        "total_fine_inr": total_fine,
        "violations": viols,
        "annotated_image": f"data:image/jpeg;base64,{img_base64}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
