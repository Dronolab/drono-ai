import socket
import threading
import time
import argparse
import cv2
import numpy as np
import os
from ultralytics import YOLO

class UltralyticsDetector:
    """
    Balloon detector using Ultralytics YOLO with TensorRT engine
    """
    def __init__(self, engine_path):
        print(f"Loading TensorRT engine from {engine_path}...")
        self.model = YOLO(engine_path)
        print("TensorRT engine loaded successfully!")
    
    def detect(self, frame):
        # Run inference
        results = self.model(frame, imgsz=800, task="detect", conf=0.3)
        
        # Process results into our format
        balloon_detections = []
        
        if results and len(results) > 0:
            # Get the first result - typically there's only one
            result = results[0]
            
            # Convert boxes to our format
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for i in range(len(result.boxes)):
                    box = result.boxes[i]
                    
                    # Get coordinates (Ultralytics returns [x1, y1, x2, y2])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get confidence score
                    confidence = box.conf[0].item()  # Extract scalar value
                    
                    # Convert to integers for C++ client
                    balloon_detections.append({
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                        'confidence': float(confidence)
                    })
        
        print(f"Detected {len(balloon_detections)} balloons")
        return balloon_detections

class DetectionServer:
    def __init__(self, port, engine_path, image_path="test1.jpg"):
        """
        Initialize the detection server
        
        Args:
            port: Port to listen on
            engine_path: Path to the TensorRT engine file
            image_path: Path to the image to use (default: "test1.jpg")
        """
        self.port = port
        self.engine_path = engine_path
        self.detector = UltralyticsDetector(engine_path)
        self.image_path = image_path
        self.running = False
        self.server_socket = None
    
    # Camera functions are commented out since we'll use a fixed image
    
    """
    def start_camera(self):
        # Start capturing from the camera
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            print(f"Warning: Failed to open camera {self.camera_id}, using test image instead")
            self.use_test_image = True
            # Create a simple test image with a red background
            self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.test_image[:, :, 2] = 255  # Red color
            
            # Draw white circles for testing
            cv2.circle(self.test_image, (160, 120), 50, (255, 255, 255), -1)
            cv2.circle(self.test_image, (480, 360), 70, (255, 255, 255), -1)
        else:
            self.use_test_image = False
            print(f"Camera {self.camera_id} opened successfully")
    
    def stop_camera(self):
        # Stop and release the camera
        if self.camera and not self.use_test_image:
            self.camera.release()
    """
    
    def start_camera(self):
        """Load the fixed image instead of starting a camera"""
        print(f"Using fixed image: {self.image_path}")
        
        # Check if the image exists
        if os.path.exists(self.image_path):
            self.test_image = cv2.imread(self.image_path)
            if self.test_image is None:
                print(f"Error loading image from {self.image_path}, using fallback test image")
                self.create_fallback_test_image()
        else:
            print(f"Image not found at {self.image_path}, using fallback test image")
            self.create_fallback_test_image()
    
    def create_fallback_test_image(self):
        """Create a fallback test image with simulated balloons"""
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_image[:, :, 2] = 255  # Red background
        
        # Draw white circles for testing
        cv2.circle(self.test_image, (160, 120), 50, (255, 255, 255), -1)
        cv2.circle(self.test_image, (480, 360), 70, (255, 255, 255), -1)
    
    def stop_camera(self):
        """No camera to stop when using fixed image"""
        pass
    
    def get_frame(self):
        """Return the fixed image"""
        return self.test_image.copy()
    
    def start_server(self):
        """Start the socket server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Try to bind to all interfaces instead of just localhost
            print(f"Attempting to bind to port {self.port}...")
            self.server_socket.bind(('0.0.0.0', self.port))
            print(f"Successfully bound to port {self.port}")
            
            self.server_socket.listen(5)
            print(f"Listening for connections...")
            
            self.running = True
            print(f"Balloon detection server started on port {self.port}")
            
            self.start_camera()
            
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
        except Exception as e:
            print(f"Error in server: {e}")
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
                # Get a frame
                frame = self.get_frame()
                
                if frame is not None:
                    # Detect balloons in the frame
                    print("Detecting balloons in frame...")
                    detections = self.detector.detect(frame)
                    
                    # Format detections for the C++ client
                    response = self.format_detections(detections)
                    
                    # Send response
                    print(f"Sending response: '{response}'")
                    bytes_sent = client_socket.send(response.encode('utf-8'))
                    print(f"Sent {bytes_sent} bytes")
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
    parser = argparse.ArgumentParser(description='Ultralytics YOLO TensorRT Server')
    parser.add_argument('--port', type=int, default=12345, help='Port to listen on')
    parser.add_argument('--engine', type=str, required=True, help='Path to TensorRT engine file')
    parser.add_argument('--image', type=str, default="test1.jpg", help='Path to image file to use')
    
    args = parser.parse_args()
    
    server = DetectionServer(args.port, args.engine, args.image)
    server.start_server()

if __name__ == "__main__":
    main()
