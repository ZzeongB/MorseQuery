import cv2
import time

data_filename = "input.mp4"
duration_sec = 5
fps = 25
camera_id = 0

cap = cv2.VideoCapture(camera_id)
assert cap.isOpened(), "Camera not opened"

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(data_filename, fourcc, fps, (width, height))

start = time.time()
while time.time() - start < duration_sec:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()

print("saved:", data_filename)
