# Poker Hand Detector 🎴 (YOLO + OpenCV + supervision)

Detector de cartas de póker en tiempo real con **YOLO** (Ultralytics), **OpenCV** y anotación con **supervision**.  
Cuando el modelo detecta **5 cartas únicas** con suficiente confianza, se evalúa la mano usando `findPokerHand.py`, clasificando desde **High Card** hasta **Royal Flush** (incluye el caso **A-2-3-4-5**).

---

## 🚀 Características

- Detección de cartas con **YOLO** a partir de `weights.pt`.
- Anotación de **cajas** y **etiquetas** con `supervision`.
- Evaluación robusta de la mano:
  - Soporta **escalera con As bajo** (A-2-3-4-5).
  - Devuelve `(score, name)` para uso posterior (comparaciones, ranking, etc.).
- Visualización en tiempo real con OpenCV.

---

## 🗂️ Estructura

```
.
├── handDetector.py        # Webcam + YOLO + supervision + overlay
├── findPokerHand.py       # Lógica de evaluación de la mano de póker
├── weights.pt             # Pesos YOLO
└── README.md
```

> El script lee los nombres de clase desde `model.names` en `weights.pt` (52 cartas: AC, AD, AH, AS, …, QS).

---

## ⚙️ Configuración rápida

1. Coloca `weights.pt` en la raíz del proyecto.
2. Ajusta la cámara si es necesario en `handDetector.py`:
   ```python
   cap = cv2.VideoCapture(0)  # Prueba con 1, 2… si 0 no funciona
   ```
3. (Opcional) Ajusta el umbral de confianza:
   ```python
   CONF_TH = 0.5
   ```
4. Ejecuta:
   ```bash
   python handDetector.py
   ```

- Se abrirá la ventana **“Poker Hand Detection”** con las detecciones.
- Al haber **5 cartas únicas** por encima de `CONF_TH`, verás el nombre de la mano en el overlay.
- Sal con **q** o **ESC**.

---

## 🧠 Lógica de evaluación (findPokerHand.py)

Funciones principales:

- `parse_card(card: str) -> (rank:int, suit:str)`  
  Convierte `"10H"`, `"AC"`, etc. a valores numéricos y palo.

- `is_straight(ranks) -> (bool, high_rank:int|None)`  
  Detecta escaleras, incluido **A-2-3-4-5** (As bajo).

- `find_poker_hand(hand: list[str]) -> (score:int, name:str)`  
  Devuelve el tipo de mano y su puntuación:
  ```
  10: Royal Flush
   9: Straight Flush
   8: Four of a Kind
   7: Full House
   6: Flush
   5: Straight
   4: Three of a Kind
   3: Two Pair
   2: Pair
   1: High Card
  ```

### Tests
Ejecuta pruebas básicas incluidas en el propio archivo:
```bash
python findPokerHand.py
```

---

## 🔍 Cómo funciona el pipeline

1. **Captura** un frame de la webcam.
2. **Inferencia** con `YOLO("weights.pt")` → `results`.
3. **Conversión** a `sv.Detections` y **anotación** de cajas/etiquetas con `supervision`.
4. **Filtrado** por confianza (`CONF_TH`) y **deduplicación** para quedarnos con 5 cartas únicas.
5. **Evaluación** de la mano con `find_poker_hand`.
6. **Overlay** del resultado y **visualización** con `cv2.imshow`.

---

## 🛠️ Consejos y solución de problemas

- **Cámara no abre / índice incorrecto**  
  Cambia el índice de `VideoCapture`:
  ```python
  cap = cv2.VideoCapture(1)  # o 2, 3…
  ```
  En Linux, lista dispositivos con `ls -lah /dev/video*` o `v4l2-ctl --list-devices`.

- **No aparece ventana**  
  Evita entornos **headless** (WSL sin servidor X, SSH sin X-forwarding).  
  Puedes crear la ventana de forma explícita:
  ```python
  cv2.namedWindow("Poker Hand Detection", cv2.WINDOW_NORMAL)
  ```

- **Windows (DirectShow)**  
  ```python
  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  ```

- **Cámara ocupada**  
  Cierra apps que usen la webcam (Zoom, Teams, navegador, etc.).

- **Falsos positivos**  
  Sube `CONF_TH` (p. ej., `0.6–0.7`) o reduce resolución para mejorar velocidad/estabilidad.

