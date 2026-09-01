#!/usr/bin/env python3
"""
record_sign.py
Grabación de corpus LSA — MediaPipe Tasks
Proyecto: Reconocimiento de LSA — Tesis UNSTA 2026
Equipo  : Bloj · Domfrocht · Petrelli · [Manu]

Graba N repeticiones de una seña y guarda cada una como .npy
con shape (T_frames, 237).

Uso:
    python record_sign.py --label hola --signer signer01
    python record_sign.py --label gracias --signer signer02 --reps 20 --frames 40

Controles durante la grabación:
    SPACE  → iniciar grabación de la siguiente repetición
    R      → repetir la última (descartarla y volver a grabar)
    Q      → salir (guarda lo que haya hasta ese momento)
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES (deben coincidir con test_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = "models"

POSE_UPPER_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]

FACE_ROI_IDX = [
    1, 4, 6,
    33, 263,
    61, 291,
    17, 314,
    0, 267,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    159, 145,
    386, 374,
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
]

# Colores (BGR)
C_HAND_L  = (255, 180,  30)
C_HAND_R  = ( 30, 200, 255)
C_POSE    = ( 50, 255, 100)
C_FACE    = (220,  80, 220)
C_WHITE   = (255, 255, 255)
C_BLACK   = (  0,   0,   0)
C_RED     = (  0,  50, 220)
C_GREEN   = ( 40, 200,  80)
C_ORANGE  = ( 30, 150, 255)
C_OVERLAY = ( 20,  20,  20)   # fondo de texto semitransparente

FEATURE_DIM = 237


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE FEATURES (idéntico a test_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

def extract_hands(result) -> np.ndarray:
    out = np.zeros((2, 21, 3), dtype=np.float32)
    if not result.hand_landmarks:
        return out
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        side = 0 if handedness[0].category_name == "Left" else 1
        for i, lm in enumerate(landmarks):
            out[side, i] = [lm.x, lm.y, lm.z]
    return out


def extract_pose(result) -> np.ndarray:
    out = np.zeros((len(POSE_UPPER_IDX), 4), dtype=np.float32)
    if not result.pose_landmarks:
        return out
    lms = result.pose_landmarks[0]
    for i, idx in enumerate(POSE_UPPER_IDX):
        lm  = lms[idx]
        vis = getattr(lm, "visibility", 1.0)
        out[i] = [lm.x, lm.y, lm.z, vis]
    return out


def extract_face(result) -> np.ndarray:
    out = np.zeros((len(FACE_ROI_IDX), 3), dtype=np.float32)
    if not result.face_landmarks:
        return out
    lms   = result.face_landmarks[0]
    nose  = np.array([lms[1].x, lms[1].y, lms[1].z], dtype=np.float32)
    scale = abs(lms[33].x - lms[263].x) or 1.0
    for i, idx in enumerate(FACE_ROI_IDX):
        lm  = lms[idx]
        raw = np.array([lm.x, lm.y, lm.z], dtype=np.float32)
        out[i] = (raw - nose) / scale
    return out


def build_feature_vector(h, p, f) -> np.ndarray:
    return np.concatenate([h.flatten(), p.flatten(), f.flatten()])


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def draw_landmarks(frame, hand_res, pose_res, face_res):
    h, w = frame.shape[:2]

    if hand_res.hand_landmarks:
        for landmarks, handedness in zip(hand_res.hand_landmarks,
                                         hand_res.handedness):
            color = C_HAND_L if handedness[0].category_name == "Left" else C_HAND_R
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, _px(landmarks[a], w, h),
                                _px(landmarks[b], w, h), color, 1, cv2.LINE_AA)
            for lm in landmarks:
                cv2.circle(frame, _px(lm, w, h), 3, color, -1, cv2.LINE_AA)

    if pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks[0]
        for a, b in POSE_CONNECTIONS:
            if a in POSE_UPPER_IDX and b in POSE_UPPER_IDX:
                cv2.line(frame, _px(lms[a], w, h),
                                _px(lms[b], w, h), C_POSE, 2, cv2.LINE_AA)
        for idx in POSE_UPPER_IDX:
            cv2.circle(frame, _px(lms[idx], w, h), 5, C_POSE, -1, cv2.LINE_AA)

    if face_res.face_landmarks:
        lms = face_res.face_landmarks[0]
        for idx in FACE_ROI_IDX:
            cv2.circle(frame, _px(lms[idx], w, h), 2, C_FACE, -1, cv2.LINE_AA)

    return frame


def put_text_bg(frame, text, pos, font_scale=0.7, color=C_WHITE,
                thickness=1, bg=C_OVERLAY, padding=6):
    """Texto con fondo semitransparente para legibilidad."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # rectángulo de fondo
    cv2.rectangle(frame,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding),
                  bg, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness,
                cv2.LINE_AA)


def draw_progress_bar(frame, current, total, color):
    h, w = frame.shape[:2]
    bar_x  = 10
    bar_y  = h - 18
    bar_w  = w - 20
    bar_h  = 10
    filled = int(bar_w * current / total)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                  color, -1)


def overlay_state(frame, state: str, label: str, signer: str,
                  rep_done: int, rep_total: int,
                  rec_frame: int = 0, rec_total: int = 30,
                  countdown: int = 0):
    """Dibuja el HUD completo según el estado actual."""
    h, w = frame.shape[:2]

    # ── Encabezado: seña + señante ─────────────────────────────────────────
    put_text_bg(frame,
                f"Seña: '{label}'   Señante: {signer}",
                (10, 28), font_scale=0.65, color=C_WHITE)

    # ── Progreso de repeticiones ───────────────────────────────────────────
    put_text_bg(frame,
                f"Repetición: {rep_done}/{rep_total}",
                (10, 58), font_scale=0.65, color=C_ORANGE)

    # ── Estado central ─────────────────────────────────────────────────────
    if state == "IDLE":
        msg   = "Presiona  SPACE  para grabar"
        color = C_WHITE
        put_text_bg(frame, msg,
                    (w // 2 - 200, h // 2 + 10),
                    font_scale=0.9, color=color,
                    bg=(40, 40, 120), padding=12)

    elif state == "COUNTDOWN":
        cv2.rectangle(frame, (0, 0), (w, h), (30, 30, 30), 8)
        num = str(countdown)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(num, font, 5.0, 8)
        cv2.putText(frame, num,
                    (w // 2 - tw // 2, h // 2 + th // 2),
                    font, 5.0, C_ORANGE, 8, cv2.LINE_AA)
        put_text_bg(frame, "¡Prepárate!",
                    (w // 2 - 80, h // 2 + th // 2 + 50),
                    font_scale=0.8, color=C_ORANGE, bg=(40, 40, 20))

    elif state == "RECORDING":
        # borde rojo parpadeante
        blink = int(time.time() * 4) % 2 == 0
        if blink:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), C_RED, 5)
        put_text_bg(frame, f"● GRABANDO  {rec_frame}/{rec_total}",
                    (w // 2 - 130, 28),
                    font_scale=0.8, color=C_RED, bg=(20, 0, 0))
        draw_progress_bar(frame, rec_frame, rec_total, C_RED)

    elif state == "SAVED":
        cv2.rectangle(frame, (0, 0), (w, h), C_GREEN, 6)
        put_text_bg(frame, f"✓  Repetición {rep_done} guardada",
                    (w // 2 - 180, h // 2),
                    font_scale=0.9, color=C_GREEN,
                    bg=(10, 30, 10), padding=12)

    elif state == "DONE":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), C_BLACK, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        put_text_bg(frame, f"¡Listo! {rep_total} repeticiones guardadas.",
                    (w // 2 - 240, h // 2 - 20),
                    font_scale=0.9, color=C_GREEN,
                    bg=(10, 30, 10), padding=14)
        put_text_bg(frame, "Presiona  Q  para salir",
                    (w // 2 - 140, h // 2 + 30),
                    font_scale=0.7, color=C_WHITE,
                    bg=(40, 40, 40), padding=10)

    # ── Controles (esquina inferior izq) ───────────────────────────────────
    if state in ("IDLE", "SAVED"):
        put_text_bg(frame, "[SPACE] grabar   [R] repetir última   [Q] salir",
                    (10, h - 12),
                    font_scale=0.45, color=(180, 180, 180), bg=(30, 30, 30))

    return frame


# ══════════════════════════════════════════════════════════════════════════════
# GRABACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def record(label: str, signer: str, n_reps: int, n_frames: int,
           source: str, countdown_secs: int, output_dir: str):

    # ── Directorio de salida ──────────────────────────────────────────────
    sign_dir = os.path.join(output_dir, label)
    os.makedirs(sign_dir, exist_ok=True)

    # Detectar repeticiones ya grabadas para no sobreescribir
    existing = [f for f in os.listdir(sign_dir)
                if f.startswith(signer) and f.endswith(".npy")]
    rep_start = len(existing)
    if rep_start > 0:
        print(f"[INFO] Se encontraron {rep_start} repetición(es) previas de "
              f"'{signer}' para '{label}'. Continuando desde rep {rep_start + 1}.")

    # ── Modelos ───────────────────────────────────────────────────────────
    def model_path(name):
        p = os.path.join(MODEL_DIR, f"{name}_landmarker.task")
        if not os.path.exists(p):
            sys.exit(f"[ERROR] Modelo '{p}' no encontrado. "
                     f"Ejecutá primero test_pipeline.py para descargarlo.")
        return p

    RunningMode = mp_vision.RunningMode

    hand_det = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path("hand")),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    pose_det = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path("pose")),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    face_det = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path("face")),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    # ── Video source ─────────────────────────────────────────────────────
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[ERROR] No se pudo abrir la fuente: '{source}'")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n── Configuración ───────────────────────────────────────────────")
    print(f"   Seña       : {label}")
    print(f"   Señante    : {signer}")
    print(f"   Repeticiones: {n_reps}  (ya grabadas: {rep_start})")
    print(f"   Frames/rep : {n_frames}  (~{n_frames/fps:.1f}s por seña)")
    print(f"   Salida     : {sign_dir}/")
    print(f"   Fuente     : {source}  ({width}×{height} @ {fps:.0f}fps)\n")

    # ── Máquina de estados ────────────────────────────────────────────────
    # IDLE → COUNTDOWN → RECORDING → SAVED → IDLE (× n_reps) → DONE
    state        = "IDLE"
    rep_done     = rep_start         # repeticiones completadas
    rec_buffer   = []                # frames de la repetición en curso
    countdown_t  = 0.0              # timestamp en que empezó el countdown
    saved_t      = 0.0              # timestamp en que se guardó (para mostrar brevemente)
    frame_idx    = 0

    with hand_det, pose_det, face_det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ts_ms = int(frame_idx * (1000.0 / fps))

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_res = hand_det.detect_for_video(mp_img, ts_ms)
            pose_res = pose_det.detect_for_video(mp_img, ts_ms)
            face_res = face_det.detect_for_video(mp_img, ts_ms)

            # ── Lógica de estados ─────────────────────────────────────────
            if state == "COUNTDOWN":
                elapsed   = time.time() - countdown_t
                remaining = countdown_secs - elapsed
                countdown_val = max(1, int(np.ceil(remaining)))
                if elapsed >= countdown_secs:
                    state      = "RECORDING"
                    rec_buffer = []

            elif state == "RECORDING":
                fvec = build_feature_vector(
                    extract_hands(hand_res),
                    extract_pose(pose_res),
                    extract_face(face_res),
                )
                rec_buffer.append(fvec)

                if len(rec_buffer) >= n_frames:
                    # ── Guardar repetición ─────────────────────────────
                    rep_done += 1
                    seq  = np.array(rec_buffer, dtype=np.float32)  # (T, 237)
                    fname = f"{signer}_rep{rep_done:03d}.npy"
                    fpath = os.path.join(sign_dir, fname)
                    np.save(fpath, seq)
                    print(f"  ✓  [{rep_done:02d}/{rep_done + (n_reps - rep_done)}] "
                          f"guardado: {fpath}  shape={seq.shape}")

                    state   = "SAVED"
                    saved_t = time.time()

            elif state == "SAVED":
                if time.time() - saved_t > 1.2:          # mostrar "guardado" 1.2s
                    if rep_done >= rep_start + n_reps:
                        state = "DONE"
                    else:
                        state = "IDLE"

            # ── Dibujar ───────────────────────────────────────────────────
            vis = draw_landmarks(frame.copy(), hand_res, pose_res, face_res)
            vis = overlay_state(
                vis,
                state    = state,
                label    = label,
                signer   = signer,
                rep_done = rep_done - rep_start,
                rep_total= n_reps,
                rec_frame= len(rec_buffer),
                rec_total= n_frames,
                countdown= countdown_val if state == "COUNTDOWN" else 0,
            )

            cv2.imshow(f"LSA — Grabando: '{label}'", vis)
            key = cv2.waitKey(1) & 0xFF

            # ── Controles ─────────────────────────────────────────────────
            if key == ord("q"):
                break

            elif key == ord(" ") and state == "IDLE":
                state       = "COUNTDOWN"
                countdown_t = time.time()
                countdown_val = countdown_secs

            elif key == ord("r") and state == "IDLE" and rep_done > rep_start:
                # Descartar la última repetición
                fname_last = f"{signer}_rep{rep_done:03d}.npy"
                fpath_last = os.path.join(sign_dir, fname_last)
                if os.path.exists(fpath_last):
                    os.remove(fpath_last)
                    print(f"  ✗  Repetición {rep_done} descartada.")
                rep_done -= 1
                state     = "IDLE"

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    total = rep_done - rep_start
    print(f"\n── Sesión finalizada ───────────────────────────────────────────")
    print(f"   Repeticiones grabadas esta sesión : {total}")
    print(f"   Total en disco para '{label}/{signer}': {rep_done}")
    print(f"   Directorio: {sign_dir}/")
    if total > 0:
        ejemplo = os.path.join(sign_dir, f"{signer}_rep001.npy")
        if os.path.exists(ejemplo):
            arr = np.load(ejemplo)
            print(f"\n   Shape de cada .npy: {arr.shape}  "
                  f"({arr.shape[0]} frames × {arr.shape[1]} features)")
            print(f"\n   Para cargar todas las repeticiones:")
            print(f"     import numpy as np, glob")
            print(f"     files = sorted(glob.glob('{sign_dir}/{signer}_*.npy'))")
            print(f"     X = np.stack([np.load(f) for f in files])  "
                  f"# shape ({total}, {arr.shape[0]}, {arr.shape[1]})")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LSA — Grabación de corpus",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--label",    required=True,
                        help="Nombre de la seña (ej: hola, gracias, por_favor)")
    parser.add_argument("--signer",   required=True,
                        help="ID del señante (ej: signer01, ivan, julian)")
    parser.add_argument("--reps",     type=int, default=15,
                        help="Número de repeticiones a grabar (default: 15)")
    parser.add_argument("--frames",   type=int, default=30,
                        help="Frames por repetición (default: 30 ≈ 1 segundo)")
    parser.add_argument("--source",   default="0",
                        help="Fuente de video: '0' para webcam (default: 0)")
    parser.add_argument("--countdown",type=int, default=3,
                        help="Segundos de cuenta regresiva (default: 3)")
    parser.add_argument("--output",   default="corpus",
                        help="Directorio raíz del corpus (default: corpus/)")
    args = parser.parse_args()

    record(
        label         = args.label,
        signer        = args.signer,
        n_reps        = args.reps,
        n_frames      = args.frames,
        source        = args.source,
        countdown_secs= args.countdown,
        output_dir    = args.output,
    )
