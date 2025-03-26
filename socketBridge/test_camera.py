import cv2
import time
import argparse

def test_camera(camera_id=0):
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_id}")
        return False
    
    print("Camera opened successfully!")
    print("Attempting to read frames...")
    
    # Try to read and display 100 frames (about 3-4 seconds)
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            print(f"ERROR: Failed to read frame {i}")
            break
        
        # Display info
        height, width = frame.shape[:2]
        print(f"Frame {i}: size={width}x{height}")
        
        # Add frame number to the image
        cv2.putText(frame, f"Frame: {i}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Camera Test", frame)
        
        # Check for key press to exit
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("User pressed 'q' to quit")
            break
        
        # Small delay
        time.sleep(0.03)
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test completed")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test camera display")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID to test")
    
    args = parser.parse_args()
    test_camera(args.camera)
