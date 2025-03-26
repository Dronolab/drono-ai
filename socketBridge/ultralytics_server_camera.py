import socket
import threading
import time
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

class BalloonDetectionServer:
    def __init__(self, port, engine_path, camera_id=0):
        self.port = port
        self.camera_id = camera_id
        self.engine_path = engine_path
        self.running = False
        self.server_socket = None
        
        # Initialize model
        print(f"Loading TensorRT engine from {engine_path}...")
        self.model = YOLO(engine_path)
        print("TensorRT engine loaded successfully!")
        
        # Initialize camera
        self.camera = None
    
    def start(self):
        """Start the server and the display loop"""
        try:
            # Start the socket server in a separate thread
            server_thread = threading.Thread(target=self.run_server)
            server_thread.daemon = True
            server_thread.start()
            
            # Start the display loop in the main thread
            self.run_display()
            
        except KeyboardInterrupt:
            print("Server shutting down...")
        finally:
            self.stop()
    
    def run_server(self):
        """Run the socket server in a separate thread"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            print(f"Binding to port {self.port}...")
            self.server_socket.bind(('0.0.0.0', self.port))
            print(f"Successfully bound to port {self.port}")
            
            self.server_socket.listen(5)
            print(f"Listening for connections...")
            
            self.running = True
            
            while self.running:
                try:
                    # Accept with timeout to allow clean shutdown
                    self.server_socket.settimeout(1.0)
                    client_socket, addr = self.server_socket.accept()
                    print(f"Connection from {addr}")
                    
                    # Handle client in a separate thread
                    client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                    client_thread.daemon = True
                    client_thread.start()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"Error accepting connection: {e}")
                    continue
        
        except Exception as e:
            print(f"Error in server thread: {e}")
    
    def run_display(self):
        """Run the camera display loop"""
        print(f"Opening camera {self.camera_id}...")
        self.camera = cv2.VideoCapture(self.camera_id)
        
        if not self.camera.isOpened():
            print(f"Failed to open camera {self.camera_id}")
            return
        
        print("Camera opened successfully")
        print("Press 'q' to quit")
        
        # Initialize display window
        window_name = "Balloon Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        last_detection_time = 0
        latest_results = None
        
        try:
            while self.running or self.running is None:  # Allow running before server starts
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                # Run detection every 0.5 seconds
                current_time = time.time()
                if current_time - last_detection_time > 0.5:
                    try:
                        results = self.model(frame, imgsz=800, task="detect", conf=0.3)
                        latest_results = results[0] if results and len(results) > 0 else None
                        last_detection_time = current_time
                    except Exception as e:
                        print(f"Error during detection: {e}")
                
                # Display frame with detections
                display_frame = frame.copy()
                
                if latest_results is not None:
                    # Plot results on image
                    display_frame = latest_results.plot()
                    
                    # Show number of detections
                    boxes = latest_results.boxes if hasattr(latest_results, 'boxes') else []
                    num_detections = len(boxes)
                    cv2.putText(display_frame, f"Detections: {num_detections}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the image
                cv2.imshow(window_name, display_frame)
                
                # Check for key press (wait 1ms)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("User pressed 'q'. Exiting...")
                    self.running = False
                    break
        
        except Exception as e:
            print(f"Error in display loop: {e}")
        
        finally:
            # Clean up
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            print("Display closed")
    
    def handle_client(self, client_socket):
        """Handle a client connection"""
        try:
            # Receive data from client
            data = client_socket.recv(1024).decode('utf-8')
            print(f"Received: {data}")
            
            # If client requests bounding boxes
            if "Request bounding boxes" in data:
                try:
                    # Get frame
                    if not self.camera or not self.camera.isOpened():
                        print("Camera not available")
                        client_socket.send("".encode('utf-8'))
                        return
                    
                    ret, frame = self.camera.read()
                    if not ret:
                        print("Failed to read frame")
                        client_socket.send("".encode('utf-8'))
                        return
                    
                    # Detect balloons
                    print("Detecting balloons...")
                    results = self.model(frame, imgsz=800, task="detect", conf=0.3)
                    
                    # Process detections
                    detections = []
                    if results and len(results) > 0:
                        result = results[0]
                        if hasattr(result, 'boxes') and len(result.boxes) > 0:
                            for i in range(len(result.boxes)):
                                box = result.boxes[i]
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                confidence = box.conf[0].item()
                                
                                detections.append({
                                    'x1': int(x1),
                                    'y1': int(y1),
                                    'x2': int(x2),
                                    'y2': int(y2),
                                    'confidence': float(confidence)
                                })
                    
                    # Format response
                    response = ""
                    for det in detections:
                        response += f"{det['x1']}|{det['y1']}|{det['x2']}|{det['y2']}|{det['confidence']}|"
                    
                    # Send response
                    print(f"Sending {len(detections)} detections")
                    client_socket.send(response.encode('utf-8'))
                
                except Exception as e:
                    print(f"Error handling detection request: {e}")
                    client_socket.send("".encode('utf-8'))
        
        except Exception as e:
            print(f"Error handling client: {e}")
        
        finally:
            client_socket.close()
    
    def stop(self):
        """Stop the server and release resources"""
        self.running = False
        
        if self.server_socket:
            try:
                self.server_socket.close()
                print("Server socket closed")
            except Exception as e:
                print(f"Error closing server socket: {e}")
        
        if self.camera:
            try:
                self.camera.release()
                print("Camera released")
            except Exception as e:
                print(f"Error releasing camera: {e}")
        
        # Ensure OpenCV windows are closed
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Simplified Balloon Detection Server with Display')
    parser.add_argument('--port', type=int, default=12345, help='Port to listen on')
    parser.add_argument('--engine', type=str, required=True, help='Path to TensorRT engine file')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID to use')
    
    args = parser.parse_args()
    
    server = BalloonDetectionServer(args.port, args.engine, args.camera)
    server.start()

if __name__ == "__main__":
    main()
