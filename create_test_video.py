"""
Create a test video file for testing the video restoration pipeline.
"""
import cv2
import os
import numpy as np

# Create output directory
os.makedirs('inputs', exist_ok=True)

# Video parameters
height, width = 480, 640
fps = 5
output_path = 'inputs/test_video.mp4'

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Create frames
for i in range(10):
    # Create a gray frame
    frame = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    # Add a rectangle
    cv2.rectangle(frame, (100, 100), (width-100, height-100), (0, 0, 255), 3)
    
    # Add text
    cv2.putText(frame, f'Frame {i+1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add some noise to simulate old video
    noise = np.random.normal(0, 15, frame.shape).astype(np.int8)
    frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
    
    # Write frame to video
    video.write(frame)

# Release the video writer
video.release()

print(f'Created test video: {output_path}')
