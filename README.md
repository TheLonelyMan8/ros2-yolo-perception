# Language & Engine Choice

Language: Python (rclpy)

Inference Engine: PyTorch / Ultralytics API

Why: I selected Python and PyTorch/Ultralytics because it is simple, easy, fast enough, and reliable. I have explored the prototype C++ node using OpenCV's DNN module (cv::dnn) to reduce the runtime overhead, but the OpenCV 4.5.4's ONNX importer introduced parsing limitations on newer YOLO head architectures. So I just stick with PyTorch in Python due to time constrain.

 ---

# Optimization Strategies
-Standard Baseline Implementation:** No additional engine optimizations (e.g., TensorRT export, INT8/FP16 quantization, or multi-threading execution) were applied for this iteration. 
-Model Weight Persistence:** Kept the model loaded in memory upon node startup so inference callbacks avoid file read / cold-start overhead per request.
-Future Work:** Exporting the trained model to TensorRT / ONNX Runtime to bypass PyTorch execution overhead, and migrating post-processing to C++ once OpenCV ONNX parser constraints are resolved.

---

# Performance
Single-Frame Inference Latency: ~81.85 ms

Test Hardware:

CPU: Intel(R) Xeon(R) W-10855M CPU @ 2.80GHz

GPU: NVIDIA Quadro T2000 with Max-Q Design

---

# Setup Instructions

# 1. ROS 2 Dependencies
sudo apt update
sudo apt install ros-humble-vision-msgs ros-humble-cv-bridge

# 2. Python ML Dependencies
pip install ultralytics opencv-python
