# System Report: Object Detection System Prototype

## System Design

### Inference Service
The Inference Service processes videos to detect objects, overlays bounding boxes on the frames, and stores the processed images along with prediction metadata (in YOLO format). It uses OpenCV's YOLO model for object detection, with functions for prediction, post-processing, and non-maximum suppression to improve detections.

### Rectification Service
The Rectification Service identifies and augments hard negatives from the dataset. It computes losses based on Intersection over Union (IoU) and Binary Cross-Entropy (BCE) to rank the hardest negative samples. Augmentation techniques like flipping, blurring, rotation, and brightness adjustment are used.

### Interface Service
The Interface Service processes video streams using the UDP protocol. The Inference Service runs within a Docker container, ensuring modularity and scalability, while the Rectification Service operates outside the container for efficiency and flexibility.

---

## Metrics Definition

### Offline Metrics
- **Mean Average Precision**: Measures the average proportion of true positive detections out of all positive predictions.
- **Recall**: Measures the proportion of true positives out of all actual positives.
- **IoU (Intersection over Union)**: Calculates the overlap between predicted and ground truth bounding boxes, which is imoprtant for detection quality.
- **F1 Score**: Balances the precision and recall.

### Online Metrics
- **Inference Time**: Tracks the time taken for the system to process each frame, allowing for real-time detection capability.
- **Throughput**: Measures the number of frames processed per second.
- **Detection Stability**: Calculates consistency in detections across frames in video streams.

**Monitoring Plan**:
- Use logging to track inference time and throughput during deployment.
- Store detection outputs for periodic review of false positives/negatives and stability metrics.

---

## Analysis of System Parameters and Configurations

### Significant Design Decisions
1. **Model Selection**:
   - **Significance**: Model 2 was chosen based on better precision (0.80) and recall (0.45), meaning better performance in detecting objects with fewer false detections.
   - **Analysis**: Performance metrics from `model_performance.ipynb` showed that Model 2 outperformed Model 1, especially in handling underrepresented classes like gloves. 

2. **Confidence and IoU Thresholds**:
   - **Significance**: These thresholds define the trade-off between detection sensitivity and false positive rates. This decision is important because the different evaluation metrics have different interpretations. 
   - **Analysis**: Using `post_process()` and non-maximum suppression, we set the thresholds (confidence: 0.5, IoU: 0.4). This configuration balances high recall with good precision for cluttered environments.We also primarily focused on IoU over other metrics because it provides a granular view.

3. **Augmentation Techniques**:
   - **Significance**: Augmentations improve dataset diversity, helping the model generalize to real-world scenarios.
   - **Analysis**: `augmentation_analysis.ipynb` evaluated the impact of augmentations like flipping, rotation, and brightness adjustment. Horizontal flipping improved performance on symmetrical objects, while rotation enhanced robustness for angled views. 

4. **Hard Negative Mining**:
   - **Significance**: Identifying hard negatives ensures the model learns from challenging cases, reducing misclassifications.
   - **Analysis**: `hard_mining_analysis.ipynb` ranked negatives based on computed losses (IoU and BCE). This allows for targeted augmentation of difficult samples, improving detection of underrepresented classes like gloves.

5. **Dockerized Inference Service**:
   - **Significance**: Containerizing the inference service makes it more portable and modular, which meakes deploying across different environments easier.
   - **Analysis**: The service was placed in a Docker container with few dependencies, isolating it from the rectification service. This makes it easier to allocate and is more scalable.

---

## Conclusion
This system integrates object detection techniques with robust rectification methods. We made design decisions based on empirical evidence, allowing for optimal performance on the TechTrack dataset. 

