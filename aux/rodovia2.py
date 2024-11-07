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

def frame_capture_thread(rtsp_link, queue, stream_id, max_retries=30, retry_delay=5):
    retries = 0
    cap = cv2.VideoCapture(rtsp_link)

    while retries < max_retries:
        if not cap.isOpened():
            print(f"Stream {stream_id} not opened, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)  # Delay before retrying
            cap = cv2.VideoCapture(rtsp_link)
            retries += 1
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Stream {stream_id} failed to capture frame. Retrying in {retry_delay} seconds...")
                retries += 1
                time.sleep(retry_delay)  # Wait before retrying to capture
                cap.release()
                cap = cv2.VideoCapture(rtsp_link)
                break  # Exit current loop and try to reconnect

            retries = 0  # Reset retry count on successful frame capture
            frame = cv2.resize(frame, (640, 640))  # Resize the frame
            queue.put((stream_id, frame))  # Add the frame to the queue

    print(f"Stream {stream_id} failed after {max_retries} retries. Exiting.")
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

    # Dictionaries to Track Vehicle Positions and Stopped Time for each stream
    object_positions = [{} for _ in rtsp_links]
    object_stop_times = [{} for _ in rtsp_links]
    alert_sent = [{} for _ in rtsp_links]

    # Threshold Values
    frame_threshold = 10
    distance_threshold = 10
    stop_duration_threshold = 1
    confidence_threshold = 0.001

    frame_count = 0

    while True:
        start_time = time.time()

        # Collect frames from all queues (streams)
        frames = []
        valid_streams = []
        for i, q in enumerate(frame_queues):
            print('Start ', i)
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

        # # Run YOLO model on the batch of frames
        # results = model(frames, device=device)
        #
        # # Process detections for each valid stream
        # for stream_idx, result in zip(valid_streams, results):
        #     frame = frames[stream_idx]
        #
        #     # Loop through detections
        #     for box in result.boxes:
        #         x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        #         conf = box.conf.cpu().numpy()[0]
        #         cls = int(box.cls.cpu().numpy()[0])
        #
        #         confidence = float(conf)
        #         class_name = model.names[cls]
        #
        #         if confidence > confidence_threshold:
        #             center = get_center([x1, y1, x2, y2])
        #             object_id = f"{int(x1)}_{int(y1)}_{class_name}"
        #
        #             if object_id not in object_positions[stream_idx]:
        #                 object_positions[stream_idx][object_id] = []
        #
        #             object_positions[stream_idx][object_id].append(center)
        #
        #             if len(object_positions[stream_idx][object_id]) > frame_threshold:
        #                 object_positions[stream_idx][object_id] = object_positions[stream_idx][object_id][-frame_threshold:]
        #
        #             if len(object_positions[stream_idx][object_id]) == frame_threshold:
        #                 start_position = object_positions[stream_idx][object_id][0]
        #                 end_position = object_positions[stream_idx][object_id][-1]
        #                 distance = np.linalg.norm(np.array(end_position) - np.array(start_position))
        #
        #                 if distance < distance_threshold:
        #                     if object_id not in object_stop_times[stream_idx]:
        #                         object_stop_times[stream_idx][object_id] = time.time()
        #                     else:
        #                         stop_duration = time.time() - object_stop_times[stream_idx][object_id]
        #                         if stop_duration >= stop_duration_threshold:
        #                             if object_id not in alert_sent[stream_idx] or not alert_sent[stream_idx][object_id]:
        #                                 cv2.putText(frame, 'Stopped', (int(x1), int(y1) - 10),
        #                                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        #                                 print(f"Alert: Stopped {class_name} detected at {center} for {stop_duration:.2f} seconds")
        #                                 alert_sent[stream_idx][object_id] = True
        #                 else:
        #                     if object_id in object_stop_times[stream_idx]:
        #                         del object_stop_times[stream_idx][object_id]
        #                     if object_id in alert_sent[stream_idx]:
        #                         alert_sent[stream_idx][object_id] = False
        #
        #             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        #             label = f"{class_name} {confidence:.2f}"
        #             cv2.putText(frame, label, (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #
        #     # Display the frame for this stream
        #     cv2.imshow(f'Stream {stream_idx+1}', frame)

        elapsed_time = time.time() - start_time
        remaining_time = f_time - elapsed_time

        if remaining_time > 0:
            time.sleep(remaining_time * 0.5)

        # if cv2.waitKey(int(f_time * 1000)) & 0xFF == ord('q'):
        #     break

    print('Saindo')
    # Release all video resources
    for t in capture_threads:
        t.join()
    #cv2.destroyAllWindows()


