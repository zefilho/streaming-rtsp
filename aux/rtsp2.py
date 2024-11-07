 from ultralytics import solutions
import cv2

speed_region = [(305, 121), (521, 347)]
def camera_streams(sources):
    speed = solutions.SpeedEstimator(model="yolo11n.pt", region=speed_region, show=True)

    stream_loader = LoadStreams(sources)
    for sources, imgs, _ in stream_loader:
        # Process and/or visualize the images here (e.g., using OpenCV)
        for j, img in enumerate(imgs):
            new_frame = cv2.resize(img, (720, 640), interpolation=cv2.INTER_AREA)
            im0 = speed.estimate_speed(new_frame)
            cv2.imshow(f"Camera Stream {sources[j]}", im0)  # Display the image using OpenCV

        # Apply further processing, if needed (e.g., object detection results)

        # Handle keyboard input for potential exit (optional)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break  # Exit the loop on pressing 'q'

    stream_loader.close()
    cv2.destroyAllWindows()  # Close all OpenCV windows after loop finishes

if __name__ == '__main__':
    rtsp_links = [
        "rtsp://ITDT:ITDT123%@172.25.103.100:554/1/profile2/media.smp",
    ]
    camera_streams('./list.streams')