import cv2

cap = cv2.VideoCapture("rtsp://ITDT:ITDT123%@172.25.103.100:554/1/profile2/media.smp")
ret, img = cap.read()
frame = cv2.resize(img, (720, 640))
cv2.imwrite("first_frame.png", frame)

#
# def generate_frame(source_path: str):
#     # extract video frame
#     generator = sv.get_video_frames_generator(source_path)
#     iterator = iter(generator)
#     frame = next(iterator)
#
#     # save first frame
#     cv2.imwrite("first_frame.png", frame)
#
#
# while True:
#     ret, img = cap.read()
#     if ret:
#         frame = cv2.resize(img, (640, 640))  # Resize the frame
#         cv2.imshow('video output', frame)
#         k = cv2.waitKey(10) & 0xff
#         if k == 27:
#             break
# cap.release()
# cv2.destroyAllWindows()

# from ultralytics import YOLO
#
# # Load a pretrained YOLO11n model
# model = YOLO("yolov8n.pt")
#
# # Multiple streams with batched inference (e.g., batch-size 8 for 8 streams)
# source = "./list.streams"  # *.streams text file with one streaming address per line
#
# # Run inference on the source
# results = model(source, stream=True)  # generator of Results objects
#
# # Process results generator
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     print(result)