import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch
import threading
from queue import Queue

# Load the YOLO Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n.pt').to(device)

# Function to Calculate the Center of a Bounding Box
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)

# Thread function to capture frames from each RTSP stream
def frame_capture_thread(rtsp_link, queue, stream_id):
    cap = cv2.VideoCapture(rtsp_link)
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Stream {stream_id} failed to capture frame.")
            break
        # Resize the frame and put it in the queue
        frame = cv2.resize(frame, (1480, 780))
        print(frame, flush=True)
        queue.put((stream_id, frame))

    cap.release()

# Main processing function that processes frames from all streams in batch
def process_stream(rtsp_links):
    # Set Desired Frame Rate
    dfps = 30
    f_time = 1 / dfps

    # Create queues for each stream
    frame_queues = [Queue(maxsize=1) for _ in rtsp_links]

    # Start threads for frame capture
    capture_threads = []
    for i, link in enumerate(rtsp_links):
        t = threading.Thread(target=frame_capture_thread, args=(link, frame_queues[i], i))
        t.start()
        capture_threads.append(t)

    confidence_threshold = 0.001

    frame_count = 0

    while True:
        start_time = time.time()

        # Collect frames from all queues (streams)
        frames = []
        valid_streams = []
        for i, q in enumerate(frame_queues):
            if not q.empty():
                stream_id, frame = q.get()
                frames.append(frame)
                valid_streams.append(stream_id)  # Track the valid stream

        # If no frames were collected, exit the loop
        if len(frames) == 0:
            print("No frames received from streams, stopping.")
            break

        # Skip frames to speed up processing
        frame_count += 1
        if frame_count % 3 != 0:
            continue

        # Run YOLO model on the batch of frames
        results = model(frames, device=device)

        # Process detections for each valid stream
        for stream_idx, result in zip(valid_streams, results):
            frame = frames[stream_idx]

            # Loop through detections
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()[0]
                cls = int(box.cls.cpu().numpy()[0])

                confidence = float(conf)
                class_name = model.names[cls]

                if confidence > confidence_threshold:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Display the frame for this stream
            cv2.imshow(f'Stream {stream_idx+1}', frame)

        elapsed_time = time.time() - start_time
        remaining_time = f_time - elapsed_time

        if remaining_time > 0:
            time.sleep(remaining_time * 0.5)

        if cv2.waitKey(int(f_time * 1000)) & 0xFF == ord('q'):
            break

    # Release all video resources
    for t in capture_threads:
        t.join()
    cv2.destroyAllWindows()

# List of 10 RTSP links
rtsp_links = [
    "rtsp://ITDT:ITDT123%@172.25.103.100:554/1/profile2/media.smp",
]

# Process streams in batch using multithreaded capture
process_stream(rtsp_links)
