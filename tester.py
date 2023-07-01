import cv2
import dlib
import numpy as np

# Load pre-trained models
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Load virtual objects (mustache and glasses)
mustache_img = cv2.imread("mustache.png", -1)
glasses_img = cv2.imread("glasses.png", -1)

def scale_object(obj, landmarks, left_idx, right_idx, top_idx, bottom_idx):
    left = landmarks.part(left_idx).x
    right = landmarks.part(right_idx).x
    top = landmarks.part(top_idx).y
    bottom = landmarks.part(bottom_idx).y

    width = int(right - left)
    height = int(bottom - top)

    scaled_obj = cv2.resize(obj, (width, height))

    return scaled_obj

def apply_lens(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Scale mustache
        scaled_mustache = scale_object(mustache_img, landmarks, 31, 35, 29, 33)

        # Overlay mustache on the frame
        for i in range(scaled_mustache.shape[0]):
            for j in range(scaled_mustache.shape[1]):
                if scaled_mustache[i, j, 3] != 0:
                    y = landmarks.part(33).y + int(i * 0.8)
                    x = landmarks.part(33).x - int(scaled_mustache.shape[1] / 2) + j
                    if y < frame.shape[0] and x < frame.shape[1]:
                        frame[y, x, :] = scaled_mustache[i, j, :3]

        # Scale glasses
        scaled_glasses = scale_object(glasses_img, landmarks, 17, 26, 24, 29)

        # Calculate the center between the two eyeballs
        eye_center_x = int((landmarks.part(36).x + landmarks.part(45).x) / 2)
        eye_center_y = int((landmarks.part(36).y + landmarks.part(45).y) / 2)

        # Calculate the eye angle
        eye_angle = np.arctan2(landmarks.part(45).y - landmarks.part(36).y,
                               landmarks.part(45).x - landmarks.part(36).x)
        eye_angle = np.degrees(eye_angle)

        # Adjust the position of the glasses
        y_offset = -10  # Adjust this value to raise or lower the glasses
        x_offset = 0  # Adjust this value to move the glasses horizontally

        # Overlay glasses on the frame with tilt adjustment
        for i in range(scaled_glasses.shape[0]):
            for j in range(scaled_glasses.shape[1]):
                if scaled_glasses[i, j, 3] != 0:
                    y = eye_center_y - int(scaled_glasses.shape[0] / 2) + y_offset + i
                    x = eye_center_x - int(scaled_glasses.shape[1] / 2) + x_offset + j

                    # Apply tilt adjustment
                    tilt_y = int(np.sin(np.radians(eye_angle)) * (j - scaled_glasses.shape[1] / 2))
                    y += tilt_y

                    if y < frame.shape[0] and x < frame.shape[1]:
                        frame[y, x, :] = scaled_glasses[i, j, :3]

    return frame

# Open the webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Set the width of the captured frame
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Set the height of the captured frame

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Apply the lens effect
    frame_with_lens = apply_lens(frame)

    # Display the frame
    cv2.imshow('Snapchat Lens', frame_with_lens)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
