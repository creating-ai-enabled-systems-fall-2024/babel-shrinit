# hard_negative_mining_module.py

import os
import json
import numpy as np
from scipy.stats import entropy

# IoU and loss calculation 
def compute_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def compute_loss(prediction, annotation):
    """Compute total loss thru IoU for localization and BCE for classification."""
    total_loss = 0.0
    iou_threshold = 0.5

    for pred in prediction:
        pred_bbox = pred['bbox']
        pred_class = pred['class_id']
        
        best_iou = 0
        best_gt = None
        for gt in annotation:
            gt_bbox = gt['bbox']
            gt_class = gt['class_id']
            iou = compute_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
        
        localization_loss = 1 - best_iou if best_iou > iou_threshold else 1.0
        classification_loss = -(
            pred_class * np.log(pred_class + 1e-5) + (1 - pred_class) * np.log(1 - pred_class + 1e-5)
        ) if best_gt else 1.0
        total_loss += localization_loss + classification_loss

    return total_loss



def sample_hard_negatives(prediction_dir, annotation_dir, num_samples):
    """Samples the top-n hard negatives using computed loss."""
    losses = []

    for file_name in os.listdir(annotation_dir):
        # Construct path
        
        pred_path = os.path.join(prediction_dir, file_name.split('.')[0] + '.json')
        annot_path = os.path.join(annotation_dir, file_name.replace('.jpg','.txt'))
        # Load predictions 
        with open(pred_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # Load annotations 
        annotations = []
       
        with open(annot_path, 'r') as f:

            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                bbox = list(map(float, parts[1:])) 
                annotations.append({'bbox': bbox, 'class_id': class_id})
        
        # Compute loss, append it 
        loss = compute_loss(predictions, annotations)
        losses.append((file_name, loss))
    
    # Sort by loss, select top-n hard negatives
    losses.sort(key=lambda x: x[1], reverse=True)
    hard_negatives = losses[:num_samples]
    
    return hard_negatives



# Entropy difficulty measure
def compute_entropy(prediction):
    """ entropy of class probabilities per prediction for difficulty."""
    entropy_score = 0.0
    for pred in prediction:
        confidence_scores = pred['confidence_scores']
        entropy_score += entropy(confidence_scores)
    
    return entropy_score
