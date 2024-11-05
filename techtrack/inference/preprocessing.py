import cv2

def capture_video(filename, drop_rate):
    """
    Generator function that reads a video file and yields the remaining frames.

    Parameters:
    - filename (str): Path 
    - drop_rate (int): rate at which frames are dropped (example: drop_rate=10 yields every 10th frame).

    Yields:
    - frame (numpy.ndarray): The next frame in the video after dropping frames
    """
    # Open the video file
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {filename}")
        return

    frame_count = 0  # Total number of frames processed
    while True:
        ret, frame = cap.read()

        # If the frame was not read successfully, end of video is reached
        if not ret:
            break

        # Increment frame count
        frame_count += 1

        # Check if the current frame should be yielded based on drop_rate
        if frame_count % drop_rate == 0:
            yield frame

    # Release the video capture object
    cap.release()
