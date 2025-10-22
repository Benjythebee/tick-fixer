import cv2
import mediapipe as mp
import csv
import time
import numpy as np
from collections import deque
from win11toast import notify
# Landmark name mapping for key facial landmarks
LANDMARK_NAMES = {
    # 10: "forehead_top",
    # 151: "forehead_center",
    # 9: "eyebrows_center", 
    # 175: "chin_center",
    # 400: "chin_right",
    # 133: "left_eye_inner",
    # 33: "left_eye_outer",
    # 159: "left_eye_upper",
    # 145: "left_eye_lower",
    # 362: "right_eye_inner",
    # 263: "right_eye_outer",
    # 386: "right_eye_upper",
    # 374: "right_eye_lower",
    # 2: "nose_tip",
    # 19: "nose_bridge_upper",
    # 5: "nose_bridge_lower",
    # 51: "nose_left",
    # 281: "nose_right",
    61: "mouth_left",
    291: "mouth_right",
    13: "mouth_upper",
    14: "mouth_lower",
    # 55: "left_eyebrow_inner",
    # 70: "left_eyebrow_middle",
    # 63: "left_eyebrow_outer",
    # 285: "right_eyebrow_inner", 
    # 296: "right_eyebrow_middle",
    # 293: "right_eyebrow_outer",
    172: "left_cheek",
    397: "right_cheek"
}
vertical_landmarks = ['mouth_upper', 'mouth_lower']
tick_landmarks = ['mouth_left', 'mouth_right']
DRAW_FULL_MESH = False  # Set to False for better performance
IS_DEBUG = False
# Tick detection parameters
HISTORY_SIZE = 30  # Number of frames to keep in history (about 1 second at 30fps)
HORIZONTAL_THRESHOLD = 1.10  # 10% increase in mouth width ratio (normalized, scale-independent)
TICK_DURATION_FRAMES = 10  # Minimum frames the condition must persist

# History buffers for distances
horizontal_distances = deque(maxlen=HISTORY_SIZE)
tick_counter = 0
tick_detected = False
last_tick_time = 0
TICK_COOLDOWN = 2.0  # Seconds between tick detections

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points (normalized coordinates)"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_normalized_ratio(mouth_dist, reference_dist):
    """Calculate mouth distance as a ratio of reference distance (face width)"""
    if reference_dist == 0:
        return 0
    return mouth_dist / reference_dist

def detect_tick(horizontal_dist,vertical_dist=None):
    """Detect if current distances indicate a tick"""
    global tick_counter, tick_detected, last_tick_time
    
    if len(horizontal_distances) < HISTORY_SIZE:
        return False
    
    # Calculate baseline (median of history to be robust to outliers)
    baseline_horizontal = np.median(horizontal_distances)
    
    # Check if current distances match tick pattern
    is_tick_pattern = (
        horizontal_dist > baseline_horizontal * HORIZONTAL_THRESHOLD and
        (True if vertical_dist is None else vertical_dist < 0.1)
    )

    current_time = time.time()
    
    if is_tick_pattern:
        tick_counter += 1
        print(f"Tick detected!")
        # Tick must persist for minimum duration and respect cooldown
        if tick_counter >= TICK_DURATION_FRAMES and (current_time - last_tick_time) > TICK_COOLDOWN:
            if not tick_detected:
                tick_detected = True
                last_tick_time = current_time
                return True
    else:
        tick_counter = 0
        tick_detected = False
    
    return False

# Initialize Mediapipe FaceMesh and Drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Open webcam
cap = cv2.VideoCapture(1)
prev_time = 0
fps_display = 0
# get video dimensions for info display
info_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - 10

def getFPS():
    global prev_time, fps_display
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    fps_display = int(fps)
    return fps_display

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,      # includes iris and lips detail
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
    frame_count = 0
    PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue

        if frame_count > 10000:
            frame_count = 0

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect the face mesh
        results = face_mesh.process(rgb_frame)

        # Convert back to BGR for OpenCV display
        frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Draw the mesh on the frame
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if DRAW_FULL_MESH:
                    # Draw the mesh connections
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
                    )
                    
                # Extract key landmark positions (NORMALIZED coordinates)
                h, w, _ = frame.shape
                mouth_positions_norm = {}  # Normalized (0-1)
                vertical_distance = 0
                
                # Store normalized coordinates for distance calculations
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in LANDMARK_NAMES and LANDMARK_NAMES[idx] in tick_landmarks:
                        # Normalized coordinates for distance calculations
                        mouth_positions_norm[LANDMARK_NAMES[idx]] = (landmark.x, landmark.y)
                
                # Get reference landmarks (cheeks) for face width normalization
                left_cheek = face_landmarks.landmark[172]  # left_cheek
                right_cheek = face_landmarks.landmark[397]  # right_cheek

                upper_mouth = face_landmarks.landmark[13]  # mouth_upper
                lower_mouth = face_landmarks.landmark[14]  # mouth_lower
                
                # Calculate reference distance (face width at cheeks) in normalized coords
                face_width = calculate_distance(
                    (left_cheek.x, left_cheek.y),
                    (right_cheek.x, right_cheek.y)
                )

                vertical_distance = calculate_distance(
                    (upper_mouth.x, upper_mouth.y),
                    (lower_mouth.x, lower_mouth.y)
                )
                
                # Calculate distances if all landmarks are present
                if len(mouth_positions_norm) == 2 and face_width > 0:
                    # Horizontal distance (mouth_left to mouth_right) - NORMALIZED
                    horiz_dist_norm = calculate_distance(
                        mouth_positions_norm['mouth_left'], 
                        mouth_positions_norm['mouth_right']
                    )
                    
                    # Calculate ratios relative to face width (scale-independent)
                    horiz_ratio = calculate_normalized_ratio(horiz_dist_norm, face_width)
                    
                    
                    # Detect tick using normalized ratios
                    tick_happening = detect_tick(horiz_ratio,vertical_distance)

                    if not tick_detected:
                        # print(f"Adding horiz ratio to history: {horiz_ratio:.3f}")
                        # Add RATIO to history (not raw distance)
                        horizontal_distances.append(horiz_ratio)

                    if IS_DEBUG:
                        # Display distances and tick status
                        info_y = 30
                        cv2.putText(frame, f"Mouth Width Ratio: {horiz_ratio:.3f}", 
                                (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (5, 5, 5), 2)
                        cv2.putText(frame, f"Face Width: {face_width:.3f}", 
                                (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (5, 5, 5), 1)
                        

                        if len(horizontal_distances) >= HISTORY_SIZE:
                            baseline_h = np.median(horizontal_distances)
                            cv2.putText(frame, f"Baseline Ratio: {baseline_h:.3f}", 
                                    (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (5, 5, 5), 1)
                        
                        # Display tick detection status
                        if tick_counter > 0:
                            cv2.putText(frame, f"Tick Detected ({tick_counter}/{TICK_DURATION_FRAMES})", 
                                    (10, info_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                            if tick_counter == 10:
                                notify('Tick detected!', audio={'silent': 'true'}, button='Dismiss', duration='short')
                    
                        # Display FPS
                        fps = getFPS()
                        cv2.putText(frame, f"FPS: {int(fps)}", 
                                (info_x-75, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (5, 225, 2), 2)
                    elif tick_counter > 0:
                        if tick_counter == 10:
                                notify('Tick detected!', audio={'silent': 'true'}, button='Dismiss', duration='short')
                            
                # Draw landmark labels with names and indices
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if not IS_DEBUG: break
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    if(LANDMARK_NAMES.get(idx) not in tick_landmarks and LANDMARK_NAMES.get(idx) not in vertical_landmarks): continue
                    # Check if this landmark has a name mapping
                    if idx in LANDMARK_NAMES:
                        label = f"{LANDMARK_NAMES[idx]} ({idx})"
                        # Draw the label with background for better visibility
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                        cv2.rectangle(frame, (x-2, y-text_h-2), (x+text_w+2, y+2), (0, 0, 0), -1)
                        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    else:
                        # For other landmarks, just show the index
                        label = f"{idx}"
                        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1)

        # Display the annotated frame
        if IS_DEBUG:
            cv2.imshow('Face Mesh', frame)

        # Exit on ESC key
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()