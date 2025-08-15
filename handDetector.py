from ultralytics import YOLO
import cv2
import supervision as sv

import findPokerHand

# Config camera
cap = cv2.VideoCapture(1) #Webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Load the YOLO model
model = YOLO("weights.pt")
class_names = model.names if hasattr(model, 'names') else None
print(f"Class names: {class_names}")

# Annotation 
bbox_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1,
        color=sv.Color.white()
)

label_annotator = sv.LabelAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1,
        text_padding=5,
        color=sv.Color.white()
)

FONT = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame from camera.")
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    labels = []
    hand = []
    for class_id, confidence in zip(detections.class_id, detections.confidence):
        class_id = int(class_id)
        confidence = float(confidence)

        name = class_names.get(class_id, str(class_id)) if isinstance(class_names, dict) else class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
        label = f"{name} {confidence:.2f}"
        labels.append(label)

        if confidence >= 0.5:
            hand.append(name)

        # Draw bounding boxes and labels
        frame = bbox_annotator.annotate(
            scene=frame,
            detections=detections
        )

        frame = label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        # Remove duplicates
        seen = set()
        hand_unique = [c for c in hand if not (c in seen or seen.add(c))]

        if len(hand_unique) == 5:
            score, name = findPokerHand.find_poker_hand(hand_unique)
            cv2.putText(frame, f"Hand: {name}", (30, 70), FONT, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"{hand_unique}", (30, 110), FONT, 0.8, (0, 255, 0), 2, cv2.LINE_AA)            

        # Visualize the results
        cv2.imshow("Poker Hand Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
