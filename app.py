import os
import cv2
import time
import threading
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from yolo_inference import YOLOInference

app = FastAPI()

# Setup directories
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
PROCESSED_FOLDER = os.path.join(os.getcwd(), 'static', 'processed')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize YOLO
yolo_infer = YOLOInference(model_path='yolo11x.pt')

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/toggle_heatmap")
async def toggle_heatmap():
    yolo_infer.set_heatmap_enabled(not yolo_infer.enable_heat_map)
    return RedirectResponse(url="/live_preview", status_code=302)

@app.post("/upload")
async def upload(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    
    # Save uploaded file
    with open(video_path, "wb") as buffer:
        content = await video.read()
        buffer.write(content)
    
    processed_filename = f"processed_{video.filename}"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
    
    # Start processing in background thread
    threading.Thread(
        target=yolo_infer.process_video,
        args=(video_path, processed_path),
        daemon=True
    ).start()
    
    return {"message": "File uploaded successfully"}

@app.get("/live_preview", response_class=HTMLResponse)
async def live_preview(request: Request):
    return templates.TemplateResponse("live_preview.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    def generate():
        while True:
            if yolo_infer.latest_frame is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', yolo_infer.latest_frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
            time.sleep(0.03)
    
    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/set_zoom")
async def set_zoom(row: int = Query(default=-1), col: int = Query(default=-1)):
    yolo_infer.set_zoom_cell(row, col)
    return {"status": "OK"}

@app.get("/zoom_feed")
async def zoom_feed():
    def gen():
        while True:
            subimg = yolo_infer.get_zoomed_subimage()
            if subimg is None:
                time.sleep(0.1)
                continue
            ret, buffer = cv2.imencode('.jpg', subimg)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
            time.sleep(0.03)
    
    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/process_video")
async def process_video_route():
    input_video_path = os.path.join(UPLOAD_FOLDER, "input.mp4")
    output_video_path = os.path.join(PROCESSED_FOLDER, "output.mp4")
    
    threading.Thread(
        target=yolo_infer.process_video,
        args=(input_video_path, output_video_path),
        daemon=True
    ).start()
    
    return RedirectResponse(url="/live_preview", status_code=302)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)