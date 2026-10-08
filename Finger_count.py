import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def count_fingers(hand_landmarks, handedness):
    """
    Return how many fingers are up (0–5).
    handedness: 'Left' or 'Right'
    """

    lm = hand_landmarks.landmark
    fingers_up = 0

    # -----------------------------
    # Thumb
    # -----------------------------
    thumb_tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = lm[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = lm[mp_hands.HandLandmark.THUMB_MCP]

    # Check if thumb is extended.
    # We use both distance and direction so a closed
    # thumb is less likely to be counted accidentally.

    thumb_extended = False

    if handedness == "Right":
        if thumb_tip.x < thumb_ip.x:
            if abs(thumb_tip.x - thumb_mcp.x) > abs(thumb_ip.x - thumb_mcp.x):
                thumb_extended = True

    else:
        if thumb_tip.x > thumb_ip.x:
            if abs(thumb_tip.x - thumb_mcp.x) > abs(thumb_ip.x - thumb_mcp.x):
                thumb_extended = True

    if thumb_extended:
        fingers_up += 1

    # -----------------------------
    # Other four fingers
    # -----------------------------

    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]

    finger_pips = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP,
    ]

    for tip_id, pip_id in zip(finger_tips, finger_pips):

        if lm[tip_id].y < lm[pip_id].y:
            fingers_up += 1

    return fingers_up


def main():

    # -----------------------------
    # ESP32-CAM
    # -----------------------------

    cap = cv2.VideoCapture(
        "http://192.168.1.12:81/stream"
    )

    # Keep buffer small to reduce delay
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Cannot connect to ESP32-CAM")
        return

    print("ESP32-CAM connected!")

    # -----------------------------
    # MediaPipe
    # -----------------------------

    with mp_hands.Hands(
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("Failed to receive frame")
                continue

            # Mirror image
            frame = cv2.flip(frame, 1)

            # BGR → RGB
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # MediaPipe
            results = hands.process(rgb)

            # -----------------------------
            # Hand detection
            # -----------------------------

            if results.multi_hand_landmarks:

                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):

                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # Left / Right
                    label = handedness.classification[0].label

                    # Count fingers
                    num_fingers = count_fingers(
                        hand_landmarks,
                        label
                    )

                    print(
                        f"Hand: {label}, "
                        f"Fingers up: {num_fingers}"
                    )

                    # Display result
                    cv2.putText(
                        frame,
                        f"{label}: {num_fingers}",
                        (10, 60 if label == "Right" else 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )

            # -----------------------------
            # Show camera
            # -----------------------------

            cv2.imshow(
                "Finger Count (0-5)",
                frame
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()