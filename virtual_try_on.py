import cv2
import mediapipe as mp
import numpy as np

def virtual_try_on(clothing_path):
    # Load the clothing image with alpha channel
    clothing_img = cv2.imread(clothing_path, cv2.IMREAD_UNCHANGED)
    if clothing_img is None:
        print(f"Error: Could not load image from {clothing_path}")
        return
    
    # Ensure the image has an alpha channel
    if clothing_img.shape[2] == 3:
        clothing_img = cv2.cvtColor(clothing_img, cv2.COLOR_BGR2BGRA)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.7,  # Increased confidence threshold
        min_tracking_confidence=0.7
    )

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Pose
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                # Extract keypoints
                h, w, _ = frame.shape
                landmarks = results.pose_landmarks.landmark

                # Get shoulder and hip keypoints
                left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                                int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h))
                right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                                 int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h))
                left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                            int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h))

                # Calculate shirt width (distance between shoulders)
                shoulder_distance = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
                shirt_width = int(shoulder_distance * 1.5)  # Adjust multiplier as needed

                # Calculate shirt height (distance from shoulder to hip)
                shoulder_hip_distance = np.linalg.norm(np.array(left_shoulder) - np.array(left_hip))
                shirt_height = int(shoulder_hip_distance * 1.2)  # Adjust multiplier as needed

                # Skip invalid dimensions
                if shirt_width <= 10 or shirt_height <= 10:
                    continue

                # Resize the clothing image dynamically
                resized_clothing = cv2.resize(clothing_img, (shirt_width, shirt_height))

                # Calculate the CENTER position between shoulders
                center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
                center_y = int((left_shoulder[1] + right_shoulder[1]) / 2)

                # Adjust the shirt position relative to the torso
                x_offset = center_x - int(shirt_width * 0.5)  # Center horizontally
                y_offset = center_y - int(shirt_height * 0.2)  # Adjust vertical offset

                # Boundary checks
                x1 = max(x_offset, 0)
                y1 = max(y_offset, 0)
                x2 = min(x_offset + shirt_width, frame_width)
                y2 = min(y_offset + shirt_height, frame_height)

                # Skip if out of bounds
                if x1 >= frame_width or y1 >= frame_height:
                    continue

                # Crop the clothing image to fit within the frame
                crop_x1 = max(-x_offset, 0)
                crop_y1 = max(-y_offset, 0)
                crop_x2 = shirt_width - max((x_offset + shirt_width) - frame_width, 0)
                crop_y2 = shirt_height - max((y_offset + shirt_height) - frame_height, 0)

                # Skip invalid crops
                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    continue

                # Extract alpha channel and crop
                alpha = resized_clothing[crop_y1:crop_y2, crop_x1:crop_x2, 3] / 255.0
                alpha = cv2.merge([alpha, alpha, alpha])  # Convert to 3-channel alpha

                # Overlay the clothing
                for c in range(0, 3):
                    frame[y1:y2, x1:x2, c] = \
                        resized_clothing[crop_y1:crop_y2, crop_x1:crop_x2, c] * alpha[:, :, c] + \
                        frame[y1:y2, x1:x2, c] * (1.0 - alpha[:, :, c])

            # Display the frame
            cv2.imshow('Live Virtual Dressing Room', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()