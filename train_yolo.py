from ultralytics import YOLO
import os

def train():
    # Get the absolute path to the script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths using relative references
    data_path = os.path.join(base_dir, "dataset", "data.yaml")

    # Load YOLO model from scratch (no old weights)
    model = YOLO("yolo11x.pt")  # or use "yolov8m.pt" if you prefer a different model

    # Train the model
    model.train(
        data=data_path,
        epochs=120,
        imgsz=640,
        batch=8,
        device="cuda",
        lr0=0.0005,
        half=True
    )

if __name__ == "__main__":
    train()
