#!/usr/bin/env python3
"""
Galaxea Perception Recruitment Assignment: Single-Image YOLO Service
Candidate Skeleton Code
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_srvs.srv import Trigger
from ultralytics import YOLO
import cv2
import os
import time


class YoloServiceNode(Node):
    def __init__(self):
        super().__init__('yolo_service_node')

        # Declare parameters
        self.declare_parameter('model_path', 'models/yolov8n.pt')
        self.declare_parameter('image_path', 'data/image_1.jpg')  # Path to the test image
        self.declare_parameter('target_color', 'red')  # Example target filtering attribute
        # Filter by YOLO class name; empty = keep all. dynamic_typing: empty list infers
        # BYTE_ARRAY in Humble, so allow overrides of any array type.
        self.declare_parameter('target_classes', [], ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('conf_thres', 0.5)  # Confidence threshold

        # Initialize YOLO Model
        self.get_logger().info('Loading YOLO model...')
        self.model = None
        target_classes_value = self.get_parameter('target_classes').value
        self.target_classes = list(target_classes_value) if target_classes_value else []
        self.conf_thres = self.get_parameter('conf_thres').get_parameter_value().double_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info(f'YOLO model loaded from {self.model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model from {self.model_path}: {e}')

        # ROS 2 Publishers & Services (No Camera Subscriber needed anymore)
        self.pub_detections = self.create_publisher(
            Detection2DArray,
            '/yolo/detections',
            10
        )
        self.srv_trigger = self.create_service(
            Trigger,
            '/yolo_detector/trigger_inference',
            self.trigger_callback
        )

        self.get_logger().info('YoloServiceNode is ready! Waiting for trigger service calls...')

    def trigger_callback(self, request, response):
        """
        Trigger Service Callback:
        Reads the target image from disk and executes YOLO inference.
        """
        image_path = self.get_parameter('image_path').get_parameter_value().string_value
        self.get_logger().info(f'Trigger requested! Processing image: {image_path}')

        if not os.path.exists(image_path):
            response.success = False
            response.message = f"Image not found at {image_path}"
            return response

        # Read the image using OpenCV
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            response.success = False
            response.message = f"Failed to read image at {image_path}"
            return response

        if self.model is None:
            response.success = False
            response.message = "YOLO model is not loaded. Check the 'model_path' parameter."
            return response
        
        start_time = time.perf_counter()
        # TODO 1: Perform YOLO Inference
        results = self.model.predict(
            source=cv_img, 
            conf=0.25, 
            verbose=False,
            device='cuda'
        )
        end_time = time.perf_counter()
        inference_ms = (end_time - start_time) * 1000.0
        self.get_logger().info(f"Inference latency: {inference_ms:.2f} ms")
        result = results[0]
        num_detections = len(result.boxes)
        self.get_logger().info(f"YOLO inference completed. Detected {num_detections} objects.")
        
        # TODO 2: Target Filtering Logic
        target_classes = None 
        min_confidence = 0.30
        
        filtered_boxes = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if conf > min_confidence:
                    filtered_boxes.append(box)

        self.get_logger().info(f"Filtered down to {len(filtered_boxes)} valid target detections.")

        # TODO 3: Construct and Publish Detection2DArray Message
        detection_array = Detection2DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = "camera_frame"

        for box in filtered_boxes:
            det = Detection2D()
            
            # Extract bounding box center (x, y) and size (w, h)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            det.bbox.center.position.x = float((x1 + x2) / 2.0)
            det.bbox.center.position.y = float((y1 + y2) / 2.0)
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)

            # Set Hypothesis (Class ID + Score)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(int(box.cls[0]))
            hypothesis.hypothesis.score = float(box.conf[0])
            det.results.append(hypothesis)

            detection_array.detections.append(det)

        # Publish result to /yolo/detections topic
        self.pub_detections.publish(detection_array)

        response.success = True
        response.message = f"Successfully published {len(filtered_boxes)} detections."
        return response


def main(args=None):
    rclpy.init(args=args)
    node = YoloServiceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
