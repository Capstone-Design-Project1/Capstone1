# yolo_detection.py
import torch

class YOLODetector:
    def __init__(self, model_path='yolov5s.pt'):
        # 모델 로드 (pre-trained 모델을 사용하거나, 직접 훈련한 모델을 지정)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    def detect_person(self, frame):
        results = self.model(frame)
        boxes = []
        for det in results.xyxy[0]:  # YOLOv5에서는 xyxy 형식으로 결과가 반환됨
            # 클래스가 'person'인 경우 (0이 사람 클래스라고 가정)
            if int(det[5]) == 0:
                x1, y1, x2, y2 = map(int, det[:4])
                boxes.append((x1, y1, x2, y2))
        return boxes