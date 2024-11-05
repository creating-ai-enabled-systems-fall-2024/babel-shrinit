import cv2

def filter(bboxes, class_ids, scores, nms_iou_threshold):
    """
    Filters bounding boxes using the Non-Maximum Suppression (NMS)

    Parameters:
    - bboxes (list of tuples): Bounding boxes in the format
    - class_ids (list of ints): Class IDs corresponding 
    - scores (list of floats): Confidence scores 
    - nms_iou_threshold (float): Intersection over Union (IoU) threshold for NMS

    Returns:
    - filtered_bboxes (list of tuples): Bounding boxes after NMS
    - filtered_class_ids (list of ints): Class IDs after NMS
    - filtered_scores (list of floats): Confidence scores after NMS
    """
    # Convert bboxes to the format required by cv2.dnn.NMSBoxes (i.e., [x, y, w, h])
    boxes = [list(map(int, bbox)) for bbox in bboxes]

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=nms_iou_threshold)

    filtered_bboxes = []
    filtered_class_ids = []
    filtered_scores = []

    if len(indices) > 0:
        for i in indices.flatten():
            filtered_bboxes.append(bboxes[i])
            filtered_class_ids.append(class_ids[i])
            filtered_scores.append(scores[i])

    return filtered_bboxes, filtered_class_ids, filtered_scores
