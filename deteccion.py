from ultralytics import YOLO
import cv2

#Cargamos el modelo YOLO
model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break
  
  results = model(frame)
  
  annotated_frame = results[0].plot()
  print(annotated_frame)

  cv2.imshow("YOLO Inference", annotated_frame)

  if cv2.waitKey(1) & 0xFF == 27:
    break
  
cap.release()
cv2.destroyAllWindows()
