import shutil
import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from app.detector import detect_objects

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=JSONResponse)
def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.post("/api/detect")
async def detect_api(image: UploadFile = File(...)):
    temp_file_path = f"temp_{image.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
        
    try:
        result = detect_objects(temp_file_path, conf_threshold=0.25)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    return result