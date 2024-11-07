import cv2

from ultralytics import YOLO, solutions

# Load the pre-trained YOLOv8 model
model = YOLO("yolo11s.pt")

# Open the video file
cap = cv2.VideoCapture("rtsp://ITDT:ITDT123%@172.25.103.100:554/1/profile2/media.smp")
assert cap.isOpened(), "Error reading video file"

# Get video properties: width, height, and frames per second (fps)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define points for a line or region of interest in the video frame
line_points = [(277, 65), (605, 367)] # Line coordinates

# Specify classes to count, for example: person (0) and car (2)
classes_to_count = [2]  # Class IDs for person and car

# Initialize the Object Counter with visualization options and other parameters
counter = solutions.ObjectCounter(
    view_img=True,  # Display the image during processing
    reg_pts=line_points,  # Region of interest points
    names=model.names,  # Class names from the YOLO model
    draw_tracks=True,  # Draw tracking lines for objects
    line_thickness=2,  # Thickness of the lines drawn
)

# Process video frames in a loop
while cap.isOpened():
    success, img = cap.read()
    im0 = cv2.resize(img, (720, 640), interpolation=cv2.INTER_AREA)
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Perform object tracking on the current frame, filtering by specified classes
    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)

    # Use the Object Counter to count objects in the frame and get the annotated image
    im0 = counter.start_counting(im0, tracks)

    # Write the annotated frame to the output video
    #.imshow('Teste', im0)

# Release the video capture and writer objects
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()