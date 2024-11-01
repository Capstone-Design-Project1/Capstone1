# main.py
import cv2
from yolo_detection import YOLODetector
from mediapipe_pose import MediaPipePose
from yolo_detection import YOLODetector

# 비디오 파일 열기
cap = cv2.VideoCapture('/home/bh/Desktop/3-2학기/캡스톤/train_data/원천데이터/동영상/Abnormal_Behavior_Falldown/inside/FD_In_H11H21H31_0007_20201015_11.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('/home/bh/Desktop/3-2학기/캡스톤/images/Yolov5s_midapipe_result2.mp4', fourcc, fps, (width, height))

# YOLO와 MediaPipePose 객체 생성
yolo_detector = YOLODetector(model_path='yolov5s.pt')
mediapipe_pose = MediaPipePose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO로 사람 감지
    person_boxes = yolo_detector.detect_person(frame)

    # 각 감지된 사람 영역에 대해 포즈 감지 적용
    for (x1, y1, x2, y2) in person_boxes:
        person_crop = frame[y1:y2, x1:x2]
        
        # MediaPipe 포즈 감지
        results = mediapipe_pose.detect_pose(person_crop)

        # 원본 이미지에 포즈 주석 그리기
        mediapipe_pose.draw_pose(frame[y1:y2, x1:x2], results)

    # 결과 프레임 저장
    out.write(frame)

# 자원 해제
cap.release()
out.release()
print("프로세스 완료 및 비디오 저장 완료")
