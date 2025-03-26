import socket
import time
import threading
import random
import argparse

class SimpleDetectionServer:
    def __init__(self, port):
        """
        Initialize a simple detection server that generates random bounding boxes
        
        Args:
            port: Port to listen on
        """
        self.port = port
        self.running = False
        self.server_socket = None
    
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
            print(f"Simple test server started on port {self.port}")
            
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
    
    def handle_client(self, client_socket):
        """Handle a client connection with random bounding boxes"""
        try:
            # Receive data from client
            data = client_socket.recv(1024).decode('utf-8')
            print(f"Received: {data}")
            
            # If client requests bounding boxes, generate random ones
            if "Request bounding boxes" in data:
                # Generate 1-3 random bounding boxes
                num_boxes = random.randint(1, 3)
                detections = []
                
                for _ in range(num_boxes):
                    # Generate random coordinates (making sure x2 > x1, y2 > y1)
                    x1 = random.randint(0, 400)
                    y1 = random.randint(0, 300)
                    x2 = x1 + random.randint(50, 200)
                    y2 = y1 + random.randint(50, 200)
                    confidence = random.uniform(0.7, 0.99)
                    
                    detections.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'confidence': confidence
                    })
                
                # Format detections for the C++ client
                response = self.format_detections(detections)
                
                # Send response
                print(f"Sending response: '{response}'")
                bytes_sent = client_socket.send(response.encode('utf-8'))
                print(f"Sent {bytes_sent} bytes")
                
                # Add a small delay to ensure data is sent
                time.sleep(0.1)
        
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
    parser = argparse.ArgumentParser(description='Simple Detection Test Server')
    parser.add_argument('--port', type=int, default=12345, help='Port to listen on')
    
    args = parser.parse_args()
    
    server = SimpleDetectionServer(args.port)
    server.start_server()

if __name__ == "__main__":
    main()
