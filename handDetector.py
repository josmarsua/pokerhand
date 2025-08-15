from ultralytics import YOLO
import cv2
import supervision as sv

import findPokerHand  # find_poker_hand(list[str]) -> (score:int, name:str)

# --- Config cámara ---
cap = cv2.VideoCapture(0)  # Webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# (opcional) fuerza un backend de GUI si tu build lo soporta
# cv2.namedWindow("Poker Hand Detection", cv2.WINDOW_NORMAL)

# --- Modelo YOLO ---
model = YOLO("weights.pt")
class_names = model.names if hasattr(model, 'names') else None
#print(f"Class names: {class_names}")

# --- Annotators supervision ---
try:
    bbox_annotator = sv.BoundingBoxAnnotator()
except Exception:
    bbox_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=1, text_padding=6)

CONF_TH = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX

def get_class_name(class_names, class_id: int) -> str:
    if isinstance(class_names, dict):
        return class_names.get(class_id, str(class_id))
    if class_names and 0 <= class_id < len(class_names):
        return class_names[class_id]
    return str(class_id)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame from camera.")
        break

    # Inferencia
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Construye labels y cartas detectadas
    labels = []
    hand = []
    for class_id, confidence in zip(detections.class_id, detections.confidence):
        class_id = int(class_id)
        confidence = float(confidence)
        cname = get_class_name(class_names, class_id)
        labels.append(f"{cname} {confidence:.2f}")
        if confidence >= CONF_TH:
            hand.append(cname)

    # Dibuja una sola vez por frame
    frame = bbox_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # Quita duplicados conservando orden
    seen = set()
    hand_unique = [c for c in hand if not (c in seen or seen.add(c))]

    # Evalúa mano si hay 5 cartas
    if len(hand_unique) == 5:
        score, hand_name = findPokerHand.find_poker_hand(hand_unique)
        cv2.putText(frame, f"Hand: {hand_name}", (30, 70), FONT, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"{hand_unique}", (30, 110), FONT, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Muestra SIEMPRE (aunque no haya detecciones)
    cv2.imshow("Poker Hand Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):  # 'q' o ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
