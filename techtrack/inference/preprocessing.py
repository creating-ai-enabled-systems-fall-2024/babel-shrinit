import cv2

def capture_video(source, drop_rate):
    """
    Generator function for video file or UDP stream, drops frames according to the drop_rate,
    gives the remaining frames.

    Parameters:
    - source (str): Path to the video file or UDP stream URL
    - drop_rate (int): The rate at which frames are dropped 

    Yields:
    - frame (numpy.ndarray): The next frame in the video
    """
    # Open 
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()

        # If poor exectuion, end of video or stream interruption
        if not ret:
            # try again 
            cap.open(source)
            continue

        # frame count
        frame_count += 1

        # current frame should be yielded based on drop_rate
        if frame_count % drop_rate == 0:
            yield frame

    # Release the video capture 
    cap.release()
