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

def frame_capture_thread(rtsp_link, queue, stream_id, stop_event, max_retries=30, retry_delay=5):
    retries = 0
    cap = cv2.VideoCapture(rtsp_link)

    while retries < max_retries and not stop_event.is_set():
        if not cap.isOpened():
            print(f"Stream {stream_id} not opened, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)  # Delay before retrying
            cap = cv2.VideoCapture(rtsp_link)
            retries += 1
            continue

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"Stream {stream_id} failed to capture frame. Retrying in {retry_delay} seconds...")
                retries += 1
                time.sleep(retry_delay)  # Wait before retrying to capture
                cap.release()
                cap = cv2.VideoCapture(rtsp_link)
                break  # Exit current loop and try to reconnect

            retries = 0  # Reset retry count on successful frame capture
            frame = cv2.resize(frame, (720, 480))  # Resize the frame
            queue.put((stream_id, frame))  # Add the frame to the queue
            time.sleep(0.0)

    print(f"Stream {stream_id} failed after {max_retries} retries or stopped. Exiting.")
    cap.release()

# Main processing function that processes frames from all streams in batch
def process_stream(rtsp_links):
    # Set Desired Frame Rate
    dfps = 15
    f_time = 1 / dfps

    # Create queues for each stream
    frame_queues = [Queue(maxsize=10) for _ in rtsp_links]

    # Create stop events for each stream
    stop_events = [threading.Event() for _ in rtsp_links]

    # Start threads for frame capture
    capture_threads = []
    for i, link in enumerate(rtsp_links):
        print('Start ', i)
        t = threading.Thread(target=frame_capture_thread, args=(link, frame_queues[i], i, stop_events[i]), daemon=True)
        t.start()
        capture_threads.append(t)

    frame_count = 0

    while any(t.is_alive() for t in capture_threads):  # Continue while at least one thread is alive
        start_time = time.time()

        # Collect frames from all queues (streams)
        frames = []
        valid_streams = []
        for i, q in enumerate(frame_queues):
            if not q.empty():
                stream_id, frame = q.get()
                #print(frame)
                frames.append(frame.copy())
                valid_streams.append(stream_id)  # Track the valid stream
            else:
                print('fila vazia')
        print(f'Frames {len(frames)}')
        # If no frames were collected, exit the loop
        if len(frames) == 0 and all(not t.is_alive() for t in capture_threads):
            print("No frames received from streams and all threads are dead, stopping.")
            break

        # Skip frames to speed up processing
        # frame_count += 1
        # if frame_count % 3 != 0:
        #     continue

        for i, frame in enumerate(frames):
            print(frame)
            cv2.imshow(f"Stream {i}", np.array(frame, dtype = np.uint8 ) )

        elapsed_time = time.time() - start_time
        remaining_time = f_time - elapsed_time

        if remaining_time > 0:
            time.sleep(remaining_time * 0.5)

    print('Saindo')

    # Signal all threads to stop and wait for them to finish
    for event in stop_events:
        event.set()

    for t in capture_threads:
        t.join()

    cv2.destroyAllWindows()



