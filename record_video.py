import cv2
import time
import os

def record():
    video_path = "static/uploads/recorded_video.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    cap = cv2.VideoCapture(0)  # Open the webcam
    if not cap.isOpened():
        print("[ERROR] Webcam could not be opened.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height))

    print("[INFO] Recording...")
    start_time = time.time()

    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        cv2.imshow('Recording...', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return video_path
