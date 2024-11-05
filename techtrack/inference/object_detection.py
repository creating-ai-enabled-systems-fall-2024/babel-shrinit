# object_detection_module.py

import cv2
import numpy as np

class Model:
    def __init__(self, config_path, weights_path, class_names_path):
        """
        Initializes the YOLO model.

        Parameters:
        - config_path (str): Path to the YOLO configuration file
        - weights_path (str): Path to the YOLO weights file
        - class_names_path (str): Path to the file containing class names
        """
        # Load YOLO network
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  

        
        with open(class_names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def predict(self, preprocessed_frame):
        """
        Outputs all (noisy) predictions of the model.

        Parameters:
        - preprocessed_frame (numpy.ndarray): The preprocessed frame.

        Returns:
        - outs (list): The raw outputs 
        """
       
        blob = cv2.dnn.blobFromImage(preprocessed_frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        
        outs = self.net.forward(self.output_layers)

        return outs

    def post_process(self, predict_output, score_threshold, nms_threshold=0.4):
        """
        Processes the output of the predict() function.

        Parameters:
        - predict_output (list): The output from the predict() function.
        - score_threshold (float): The minimum confidence score required
        - nms_threshold (float): Non-maximum suppression threshold

        Returns:
        - bboxes (List of tuples): Bounding boxes of detected objects
        - class_ids (List of ints): Class IDs of the detected objects
        - scores (List of floats): Confidence scores
        """
        frame_height, frame_width = self.frame_shape

        class_ids = []
        scores = []
        bboxes = []

       
        for out in predict_output:
            for detection in out:
                
                scores_array = detection[5:]
                class_id = np.argmax(scores_array)
                confidence = scores_array[class_id]

                if confidence > score_threshold:
                    # Object detected
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    bboxes.append((x, y, width, height))
                    class_ids.append(class_id)
                    scores.append(float(confidence))

        
        indices = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold, nms_threshold)

        filtered_bboxes = []
        filtered_class_ids = []
        filtered_scores = []

        if len(indices) > 0:
            for i in indices.flatten():
                filtered_bboxes.append(bboxes[i])
                filtered_class_ids.append(class_ids[i])
                filtered_scores.append(scores[i])

        return filtered_bboxes, filtered_class_ids, filtered_scores

    def set_frame_shape(self, frame_shape):
        """
        Sets the shape of the frame being processed.

        Parameters:
        - frame_shape (tuple): Shape of the frame in the form (height, width).
        """
        self.frame_shape = frame_shape[:2]  # (height, width)
