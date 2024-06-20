import cv2
from ultralytics import YOLO
import time
import numpy as np

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = r"C:\Users\iamma\OneDrive\Desktop\SURGE\head_on.mp4"
cap = cv2.VideoCapture(video_path)

# Get frames per second (fps)
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the number of frames to skip to achieve approximately 125 ms frame rate
skip_frames = round(fps * 0.125)

# Get frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter objects for the output videos
video_annotate = cv2.VideoWriter_fourcc(*'mp4v')
out_annotate = cv2.VideoWriter(r"D:\ComputerVision\multiPed_annotate.mp4", 
                               video_annotate, 
                               fps, 
                               (frame_width, frame_height))

video_traj = cv2.VideoWriter_fourcc(*'mp4v')
out_traj = cv2.VideoWriter(r"D:\ComputerVision\multiPed_circle.mp4", 
                           video_traj, 
                           fps, 
                           (frame_width, frame_height))

# Arrays to store positions and time
x_pos_1 = []
y_pos_1 = []
x_pos_2 = []
y_pos_2 = []
times = []

start_time = time.time()
frame_counter = 0

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame_counter += 1

        # Only process frames at the specified interval
        if frame_counter % skip_frames == 0:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, classes=0)
            
            # Check if both agents are detected
            agent_1_detected = False
            agent_2_detected = False

            for track in results[0].boxes:
                id = track.id
                if id != 2:
                    agent_1_detected = True
                elif id == 2:
                    agent_2_detected = True

            print(f"Agent 1 detected: {agent_1_detected}, Agent 2 detected: {agent_2_detected}")

            # Proceed only if both agents are detected
            if agent_1_detected and agent_2_detected:
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                
                current_time = time.time() - start_time

                for track in results[0].boxes:
                    id = track.id
                    det = track.xyxy[0].numpy()
                    x1, y1, x2, y2 = map(int, det[:4])
                    center_x = (x1 + x2) // 2
                    center_y = y2

                    if id != 2:
                        x_pos_1.append(center_x)
                        y_pos_1.append(center_y)
                    elif id == 2:
                        x_pos_2.append(center_x)
                        y_pos_2.append(center_y)
                        times.append(current_time)

                    center = (center_x, center_y)
                    radius = 20
                    color = (0, 0, 255)
                    thickness = 2
                    frame = cv2.circle(frame, center, radius, color, thickness)

               

                # Write the frames to the output video files
                out_annotate.write(annotated_frame)
                out_traj.write(frame)
                
                # Display the frames
                cv2.imshow("Annotated Frame", annotated_frame)
                cv2.imshow("Trajectory Frame", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

# Release the video writer and video capture objects
out_annotate.release()
out_traj.release()
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
print(x_pos_1)

# Save the positions and times to numpy files
np.save("x_pos_1.npy", np.array(x_pos_1))
np.save("y_pos_1.npy", np.array(y_pos_1))
np.save("x_pos_2.npy", np.array(x_pos_2))
np.save("y_pos_2.npy", np.array(y_pos_2))
np.save("times.npy", np.array(times))
