#!/usr/bin/env python3
"""
test_pipeline.py
Pipeline de prueba: extracción de keypoints con MediaPipe Tasks
Proyecto: Reconocimiento de LSA — Tesis UNSTA 2026
Equipo  : Bloj · Domfrocht · Petrelli · [Manu]

Landmarkers utilizados:
    HandLandmarker  → 21 kps × 2 manos  = 42 puntos
    PoseLandmarker  → 9 kps upper body   = 9 puntos
    FaceLandmarker  → 27 kps no-manuales = 27 puntos

Vector de features por frame: 126 (manos) + 36 (pose) + 75 (cara) = 237 valores

Uso:
    python test_pipeline.py                        # webcam
    python test_pipeline.py --source video.mp4     # archivo de video
    python test_pipeline.py --source 0 --save      # webcam + guardar .npy
    python test_pipeline.py --source seña.mp4 --save --output mi_seña.npy
"""

import argparse
import os
import sys
import urllib.request
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

MODEL_URLS = {
    "hand": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "face": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
}
MODEL_DIR = "models"

# Subset de pose: nariz + hombros + codos + muñecas + caderas
# Índices según PoseLandmarker (BlazePose 33 kps)
POSE_UPPER_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]

# Landmarks faciales para rasgos no-manuales discriminativos
# Referencia: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
FACE_ROI_IDX = [
    # Nariz (punto de referencia para normalización)
    1, 4, 6,
    # Comisuras oculares (izq, der)
    33, 263,
    # Comisuras bucales (izq, der)
    61, 291,
    # Labio inferior
    17, 314,
    # Labio superior
    0, 267,
    # Ceja izquierda (5 puntos)
    70, 63, 105, 66, 107,
    # Ceja derecha (5 puntos)
    336, 296, 334, 293, 300,
    # Apertura párpado izquierdo (sup + inf)
    159, 145,
    # Apertura párpado derecho (sup + inf)
    386, 374,
]  # Total: 25 landmarks → 25×3 = 75 features

# Conexiones de mano para dibujo (21 puntos, sin depender de legacy solutions)
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),           # pulgar
    (0, 5),  (5, 6),  (6, 7),  (7, 8),            # índice
    (5, 9),  (9, 10), (10, 11),(11, 12),           # medio
    (9, 13), (13, 14),(14, 15),(15, 16),           # anular
    (13, 17),(0, 17), (17, 18),(18, 19),(19, 20),  # meñique
]

# Conexiones upper body para pose
POSE_CONNECTIONS = [
    (11, 12),  # hombros
    (11, 13), (13, 15),  # brazo izq
    (12, 14), (14, 16),  # brazo der
    (11, 23), (12, 24),  # torso
]

# Colores (BGR)
C_HAND_L = (255, 180,  30)   # amarillo-naranja → mano izquierda
C_HAND_R = ( 30, 200, 255)   # cian             → mano derecha
C_POSE   = ( 50, 255, 100)   # verde
C_FACE   = (220,  80, 220)   # violeta
C_TEXT   = (255, 255, 255)
C_DIM    = (160, 160, 160)


# ══════════════════════════════════════════════════════════════════════════════
# DESCARGA DE MODELOS
# ══════════════════════════════════════════════════════════════════════════════

def _reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
    bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    print(f"\r    [{bar}] {pct:5.1f}%", end="", flush=True)


def ensure_models() -> dict[str, str]:
    """Descarga los modelos .task si no están en ./models/."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    paths = {}
    print("── Modelos ─────────────────────────────────────────────────────")
    for name, url in MODEL_URLS.items():
        dst = os.path.join(MODEL_DIR, f"{name}_landmarker.task")
        if os.path.exists(dst):
            print(f"  [{name:4s}] ✓ ya descargado  ({dst})")
        else:
            print(f"  [{name:4s}] descargando …")
            urllib.request.urlretrieve(url, dst, reporthook=_reporthook)
            print()  # newline tras la barra
        paths[name] = dst
    print()
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def extract_hands(result) -> np.ndarray:
    """
    Devuelve array (2, 21, 3) float32 con coordenadas [x, y, z] ∈ [0,1].
    Índice 0 → mano izquierda, índice 1 → mano derecha.
    Zeros si la mano no se detecta.
    """
    out = np.zeros((2, 21, 3), dtype=np.float32)
    if not result.hand_landmarks:
        return out
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        side = 0 if handedness[0].category_name == "Left" else 1
        for i, lm in enumerate(landmarks):
            out[side, i] = [lm.x, lm.y, lm.z]
    return out  # (2, 21, 3) → flatten = 126 features


def extract_pose(result) -> np.ndarray:
    """
    Devuelve array (9, 4) float32 con [x, y, z, visibility] para POSE_UPPER_IDX.
    Zeros si no hay pose detectada.
    """
    out = np.zeros((len(POSE_UPPER_IDX), 4), dtype=np.float32)
    if not result.pose_landmarks:
        return out
    lms = result.pose_landmarks[0]
    for i, idx in enumerate(POSE_UPPER_IDX):
        lm = lms[idx]
        vis = getattr(lm, "visibility", 1.0)
        out[i] = [lm.x, lm.y, lm.z, vis]
    return out  # (9, 4) → flatten = 36 features


def extract_face(result) -> np.ndarray:
    """
    Devuelve array (27, 3) float32.
    Coordenadas normalizadas relativas a la nariz y escaladas por
    el ancho intercantal (ojo izq ↔ ojo der).

    Normalización:
        coord_rel = (coord_raw - nariz) / ancho_intercantal
    Esto hace las features invariantes a posición y escala del señante.
    """
    out = np.zeros((len(FACE_ROI_IDX), 3), dtype=np.float32)
    if not result.face_landmarks:
        return out

    lms = result.face_landmarks[0]

    # Punto de referencia: punta de la nariz (idx=1)
    nose = np.array([lms[1].x, lms[1].y, lms[1].z], dtype=np.float32)

    # Escala: distancia intercantal (canto interno ojo izq=33, der=263)
    scale = abs(lms[33].x - lms[263].x)
    if scale < 1e-6:
        scale = 1.0  # fallback si no hay detección confiable

    for i, idx in enumerate(FACE_ROI_IDX):
        lm  = lms[idx]
        raw = np.array([lm.x, lm.y, lm.z], dtype=np.float32)
        out[i] = (raw - nose) / scale

    return out  # (25, 3) → flatten = 75 features


def build_feature_vector(h: np.ndarray,
                          p: np.ndarray,
                          f: np.ndarray) -> np.ndarray:
    """
    Concatena los arrays de hands, pose y face en un vector 1-D por frame.

    Estructura del vector (237 features en total):
        [0:126]   → hands  (2 × 21 × 3)
        [126:162] → pose   (9 × 4)
        [162:237] → face   (25 × 3)
    """
    return np.concatenate([h.flatten(), p.flatten(), f.flatten()])


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def draw_all(frame: np.ndarray,
             hand_res, pose_res, face_res,
             fvec: np.ndarray) -> np.ndarray:
    """Dibuja landmarks de los tres modelos sobre el frame."""
    h, w = frame.shape[:2]

    # ── Manos ─────────────────────────────────────────────────────────────────
    if hand_res.hand_landmarks:
        for landmarks, handedness in zip(hand_res.hand_landmarks,
                                         hand_res.handedness):
            color = C_HAND_L if handedness[0].category_name == "Left" else C_HAND_R
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, _px(landmarks[a], w, h),
                                _px(landmarks[b], w, h), color, 1, cv2.LINE_AA)
            for lm in landmarks:
                cv2.circle(frame, _px(lm, w, h), 3, color, -1, cv2.LINE_AA)

    # ── Pose (upper body) ──────────────────────────────────────────────────────
    if pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks[0]
        for a, b in POSE_CONNECTIONS:
            # solo dibujar si ambos pertenecen al subset
            if a in POSE_UPPER_IDX and b in POSE_UPPER_IDX:
                cv2.line(frame, _px(lms[a], w, h),
                                _px(lms[b], w, h), C_POSE, 2, cv2.LINE_AA)
        for idx in POSE_UPPER_IDX:
            cv2.circle(frame, _px(lms[idx], w, h), 5, C_POSE, -1, cv2.LINE_AA)

    # ── Cara (ROI no-manuales) ─────────────────────────────────────────────────
    if face_res.face_landmarks:
        lms = face_res.face_landmarks[0]
        for idx in FACE_ROI_IDX:
            cv2.circle(frame, _px(lms[idx], w, h), 2, C_FACE, -1, cv2.LINE_AA)

    # ── HUD ───────────────────────────────────────────────────────────────────
    n_hands = len(hand_res.hand_landmarks) if hand_res.hand_landmarks else 0
    n_pose  = 1 if pose_res.pose_landmarks else 0
    n_face  = 1 if face_res.face_landmarks else 0

    cv2.putText(frame,
                f"Manos: {n_hands}  Pose: {n_pose}  Cara: {n_face}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"Feature dim: {fvec.shape[0]}  "
                f"(hands=126 pose=36 face=75)",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_DIM, 1, cv2.LINE_AA)

    # Leyenda colores
    for i, (label, color) in enumerate([
        ("Mano izq", C_HAND_L),
        ("Mano der", C_HAND_R),
        ("Pose",     C_POSE),
        ("Cara",     C_FACE),
    ]):
        y = h - 20 - i * 22
        cv2.circle(frame, (14, y), 7, color, -1)
        cv2.putText(frame, label, (26, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run(source: str, save_features: bool, output_path: str) -> None:

    # 1) Modelos
    paths = ensure_models()

    # 2) Crear landmarkers en modo VIDEO
    RunningMode = mp_vision.RunningMode

    hand_det = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=paths["hand"]),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    pose_det = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=paths["pose"]),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    face_det = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=paths["face"]),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    # 3) Fuente de video
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[ERROR] No se pudo abrir la fuente: '{source}'")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"── Fuente : {source}")
    print(f"   Tamaño : {width}×{height}  FPS: {fps:.1f}")
    print(f"   Feature dim esperada: {2*21*3 + len(POSE_UPPER_IDX)*4 + len(FACE_ROI_IDX)*3}")
    print("\n   [q] para salir  |  [s] para guardar frame actual como .png\n")

    all_frames = []
    frame_idx  = 0
    t0         = time.perf_counter()

    # 4) Loop principal
    with hand_det, pose_det, face_det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Timestamp ms monotónicamente creciente (obligatorio en VIDEO mode)
            ts_ms = int(frame_idx * (1000.0 / fps))

            # Convertir a RGB y crear mp.Image
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Inferencia
            hand_res = hand_det.detect_for_video(mp_img, ts_ms)
            pose_res = pose_det.detect_for_video(mp_img, ts_ms)
            face_res = face_det.detect_for_video(mp_img, ts_ms)

            # Extraer features
            h_kps = extract_hands(hand_res)
            p_kps = extract_pose(pose_res)
            f_kps = extract_face(face_res)
            fvec  = build_feature_vector(h_kps, p_kps, f_kps)
            all_frames.append(fvec)

            # Visualizar
            vis = draw_all(frame.copy(), hand_res, pose_res, face_res, fvec)

            # FPS real
            elapsed = time.perf_counter() - t0
            real_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            cv2.putText(vis, f"FPS: {real_fps:.1f}",
                        (width - 100, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_TEXT, 1, cv2.LINE_AA)

            cv2.imshow("LSA — Pipeline de prueba  (q = salir)", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"frame_{frame_idx:05d}.png"
                cv2.imwrite(fname, vis)
                print(f"  Frame guardado: {fname}")

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # 5) Guardar features
    if save_features and all_frames:
        arr = np.array(all_frames, dtype=np.float32)  # (T, 243)
        np.save(output_path, arr)

        print("\n── Features guardados ───────────────────────────────────────────")
        print(f"   Archivo : {output_path}")
        print(f"   Shape   : {arr.shape}  (frames × features)")
        print(f"   Memoria : {arr.nbytes / 1024:.1f} KB")
        print()
        print("   Estructura del vector (eje=1):")
        print(f"     [0   : 126]  hands  — 2 manos × 21 kps × 3 coords")
        print(f"     [126 : 162]  pose   — {len(POSE_UPPER_IDX)} kps × 4 (x,y,z,vis)")
        print(f"     [162 : 237]  face   — {len(FACE_ROI_IDX)} kps × 3 coords (norm. nariz)")
        print()
        print("   Para cargar:")
        print(f"     import numpy as np")
        print(f"     X = np.load('{output_path}')  # shape {arr.shape}")

    elif not all_frames:
        print("\n[AVISO] No se procesó ningún frame.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LSA — Pipeline de prueba con MediaPipe Tasks",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source", default="0",
        help="Fuente de video:\n"
             "  '0'          → webcam por defecto\n"
             "  'video.mp4'  → archivo de video"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Guardar el array de features en formato .npy al terminar"
    )
    parser.add_argument(
        "--output", default="features_test.npy",
        help="Ruta de salida para el .npy  (default: features_test.npy)"
    )
    args = parser.parse_args()

    run(source=args.source,
        save_features=args.save,
        output_path=args.output)
