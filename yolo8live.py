from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import cv2

model = YOLO("yolov8x.pt")

results = model.predict(source="2", show=True) #acept all formats

print(results)