# This is a sample Python script.
import cv2
from load_streams import LoadStreams

def camera_streams(sources):
    stream_loader = LoadStreams(sources)
    for sources, imgs in stream_loader:
        # Process and/or visualize the images here (e.g., using OpenCV)
        print(sources)
        for j, img in enumerate(imgs):
            new_frame = cv2.resize(img, (720, 640), interpolation=cv2.INTER_AREA)
            cv2.imshow(f"Camera Stream {sources[j]}", new_frame)  # Display the image using OpenCV

        # Apply further processing, if needed (e.g., object detection results)
        # Handle keyboard input for potential exit (optional)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break  # Exit the loop on pressing 'q'

    stream_loader.close()
    #cv2.destroyAllWindows()  # Close all OpenCV windows after loop finishes

def run():
    print('Iniciando Streams')
    # Process streams in batch using multithreaded capture
    camera_streams('./list.streams')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
