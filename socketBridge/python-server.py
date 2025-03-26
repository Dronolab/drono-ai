import socket
import threading
import time
import argparse
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

class BalloonDetector:
    """
    Class to handle the TensorRT engine for balloon detection
    """
    def __init__(self, engine_path):
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load engine file
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate memory for inputs and outputs
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in range(self.engine.num_bindings):
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append to the lists
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})
    
    def detect(self, image):
        # Preprocess the image
        input_image = self.preprocess_image(image)
        
        # Copy data to input memory
        np.copyto(self.inputs[0]["host"], input_image.ravel())
        
        # Transfer to device memory
        for inp in self.inputs:
            cuda.memcpy_htod(inp["device"], inp["host"])
        
        # Run inference
        self.context.execute_v2(self.bindings)
        
        # Transfer from device to host memory
        for out in self.outputs:
            cuda.memcpy_dtoh(out["host"], out["device"])
        
        # Process and return detection results
        return self.process_output(self.outputs[0]["host"])
    
    def preprocess_image(self, image):
        # Resize and normalize the image
        # Adjust these values based on your model's requirements
        resized = cv2.resize(image, (640, 640))
        normalized = resized / 255.0  # Normalize to [0, 1]
        
        # Convert to the format expected by your model
        # (This might need adjustment based on your specific model)
        input_data = np.array(normalized, dtype=np.float32)
        
        # If your model expects NCHW format (common in TensorRT)
        if len(input_data.shape) == 3:
            input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW
            input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
        
        return input_data
    
    def process_output(self, output):
        # This function needs to be adjusted based on your model's output format
        # Here's a generic example for processing YOLO-style outputs
        
        # Assuming output format is [batch, num_detections, 6]
        # where each detection is [x1, y1, x2, y2, confidence, class]
        detections = np.reshape(output, (-1, 6))
        
        # Filter for balloon class and confidence threshold
        # Assuming class 0 is balloon, adjust as needed
        balloon_class = 0
        confidence_threshold = 0.5
        
        balloon_detections = []
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            
            if class_id == balloon_class and confidence > confidence_threshold:
                # Convert to integer coordinates for the C++ side
                balloon_detections.append({
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'confidence': float(confidence)
                })
        
        return balloon_detections

class DetectionServer:
    def __init__(self, port, engine_path, camera_id=0):
        """
        Initialize the detection server
        
        Args:
            port: Port to listen on
            engine_path: Path to the TensorRT engine file
            camera_id: Camera ID to use (default: 0)
        """
        self.port = port
        self.detector = BalloonDetector(engine_path)
        self.camera_id = camera_id
        self.camera = None
        self.running = False
        self.server_socket = None
    
    def start_camera(self):
        """Start capturing from the camera"""
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
    
    def stop_camera(self):
        """Stop and release the camera"""
        if self.camera:
            self.camera.release()
    
    def get_frame(self):
        """Get a frame from the camera"""
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                return frame
        return None
    
    def start_server(self):
        """Start the socket server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('127.0.0.1', self.port))
        self.server_socket.listen(5)
        
        self.running = True
        print(f"Server started on port {self.port}")
        
        self.start_camera()
        
        try:
            while self.running:
                # Accept client connections
                client_socket, addr = self.server_socket.accept()
                print(f"Connection from {addr}")
                
                # Handle client in a separate thread
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                client_thread.daemon = True
                client_thread.start()
        
        except KeyboardInterrupt:
            print("Server shutting down...")
        finally:
            self.stop_server()
    
    def stop_server(self):
        """Stop the server and clean up resources"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        self.stop_camera()
    
    def handle_client(self, client_socket):
        """Handle a client connection"""
        try:
            # Receive data from client
            data = client_socket.recv(1024).decode('utf-8')
            print(f"Received: {data}")
            
            # If client requests bounding boxes
            if "Request bounding boxes" in data:
                # Get a frame from the camera
                frame = self.get_frame()
                if frame is not None:
                    # Detect balloons in the frame
                    detections = self.detector.detect(frame)
                    
                    # Format detections for the C++ client
                    response = self.format_detections(detections)
                    
                    # Send response
                    client_socket.send(response.encode('utf-8'))
                else:
                    # If no frame, send empty response
                    client_socket.send("No frame available".encode('utf-8'))
        
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()
    
    def format_detections(self, detections):
        """Format detections for C++ client"""
        # Format: "x1|y1|x2|y2|confidence|"
        formatted = ""
        for det in detections:
            formatted += f"{det['x1']}|{det['y1']}|{det['x2']}|{det['y2']}|{det['confidence']}|"
        return formatted

def main():
    parser = argparse.ArgumentParser(description='Balloon Detection Server')
    parser.add_argument('--port', type=int, default=12345, help='Port to listen on')
    parser.add_argument('--engine', type=str, required=True, help='Path to TensorRT engine file')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID to use')
    
    args = parser.parse_args()
    
    server = DetectionServer(args.port, args.engine, args.camera)
    server.start_server()

if __name__ == "__main__":
    main()
