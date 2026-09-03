#!/usr/bin/env python3
"""
augment_mirror.py
Data augmentation por espejo horizontal — Corpus LSA
Proyecto: Reconocimiento de LSA — Tesis UNSTA 2026
Equipo  : Bloj · Domfrocht · Petrelli · [Manu]

Para cada .npy del corpus genera una versión espejo que convierte
señas de mano derecha en mano izquierda y viceversa.

La operación tiene tres partes:
  1. Manos  → swap izq↔der  +  flip x (x → 1−x)
  2. Pose   → swap pares L/R +  flip x (x → 1−x)
  3. Cara   → swap pares L/R +  negar x (x → −x, coords relativas)

Uso:
    python augment_mirror.py                    # procesa corpus/ completo
    python augment_mirror.py --corpus corpus/   # ruta explícita
    python augment_mirror.py --dry-run          # muestra qué haría, no escribe
    python augment_mirror.py --verify           # verifica mirror(mirror(x)) == x
"""

import argparse
import os
import glob
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES — deben coincidir con record_sign.py
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_DIM = 237

# ── Estructura del vector (237 features) ──────────────────────────────────────
#   [0   : 126]  hands  — izq[0:63] + der[63:126]  (2 × 21 × 3)
#   [126 : 162]  pose   — 9 landmarks × 4 (x,y,z,vis)
#   [162 : 237]  face   — 25 landmarks × 3 (x,y,z relativas a nariz)

HAND_IZQ_SLICE = slice(0,   63)
HAND_DER_SLICE = slice(63, 126)

# Índices en POSE_UPPER_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]
# Pares (izquierdo, derecho) para hacer swap
POSE_BASE = 126
POSE_LR_PAIRS = [
    (1, 2),   # hombro  izq(11) ↔ hombro  der(12)
    (3, 4),   # codo    izq(13) ↔ codo    der(14)
    (5, 6),   # muñeca  izq(15) ↔ muñeca  der(16)
    (7, 8),   # cadera  izq(23) ↔ cadera  der(24)
]
POSE_STRIDE = 4   # (x, y, z, vis) por landmark

# Posiciones en FACE_ROI_IDX = [1,4,6, 33,263, 61,291, 17,314, 0,267,
#                                70,63,105,66,107, 336,296,334,293,300,
#                                159,145, 386,374]
# Pares (izquierdo, derecho) para hacer swap
FACE_BASE = 162
FACE_LR_PAIRS = [
    (3,  4),   # comisuras oculares     izq(33)  ↔ der(263)
    (5,  6),   # comisuras bucales      izq(61)  ↔ der(291)
    (7,  8),   # labio inferior         izq(17)  ↔ der(314)
    (9,  10),  # labio superior         izq(0)   ↔ der(267)
    (11, 16),  # ceja, punto 1          izq(70)  ↔ der(336)
    (12, 17),  # ceja, punto 2          izq(63)  ↔ der(296)
    (13, 18),  # ceja, punto 3          izq(105) ↔ der(334)
    (14, 19),  # ceja, punto 4          izq(66)  ↔ der(293)
    (15, 20),  # ceja, punto 5          izq(107) ↔ der(300)
    (21, 23),  # párpado sup            izq(159) ↔ der(386)
    (22, 24),  # párpado inf            izq(145) ↔ der(374)
]
FACE_STRIDE = 3   # (x, y, z) por landmark
N_FACE_LM   = 25  # total de landmarks faciales


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN DE ESPEJO
# ══════════════════════════════════════════════════════════════════════════════

def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Genera la versión espejo de una secuencia de features.

    Args:
        seq: array float32 de shape (T, 237)

    Returns:
        mir: array float32 de shape (T, 237) con la seña espejada

    Nota sobre frames sin detección:
        Cuando MediaPipe no detecta una mano o pose, el extractor guarda
        ceros. Si aplicáramos x → 1-x sobre esos ceros, obtendríamos x=1,
        lo que parecería un landmark en el borde derecho de la imagen.
        Para evitarlo, el flip de x se aplica frame a frame SOLO cuando
        el bloque correspondiente tiene datos reales (al menos un valor ≠ 0).
        Las coordenadas de cara usan x → -x, y -0 = 0, así que no sufren
        este problema.
    """
    assert seq.ndim == 2 and seq.shape[1] == FEATURE_DIM, (
        f"Shape esperado (T, {FEATURE_DIM}), recibido {seq.shape}"
    )

    mir = seq.copy()
    T   = seq.shape[0]

    # ── 1. MANOS ──────────────────────────────────────────────────────────────
    # Swap: bloque izquierdo ↔ bloque derecho
    izq = seq[:, HAND_IZQ_SLICE].copy()
    der = seq[:, HAND_DER_SLICE].copy()
    mir[:, HAND_IZQ_SLICE] = der
    mir[:, HAND_DER_SLICE] = izq

    # Flip x frame a frame, solo si la mano fue detectada (bloque ≠ 0)
    for t in range(T):
        if np.any(mir[t, HAND_IZQ_SLICE] != 0):   # mano izq (ahora contiene der orig)
            mir[t, 0:63:3] = 1.0 - mir[t, 0:63:3]
        if np.any(mir[t, HAND_DER_SLICE] != 0):   # mano der (ahora contiene izq orig)
            mir[t, 63:126:3] = 1.0 - mir[t, 63:126:3]

    # ── 2. POSE ───────────────────────────────────────────────────────────────
    # Swap pares L/R
    for a, b in POSE_LR_PAIRS:
        a0 = POSE_BASE + a * POSE_STRIDE
        b0 = POSE_BASE + b * POSE_STRIDE
        tmp = mir[:, a0:a0 + POSE_STRIDE].copy()
        mir[:, a0:a0 + POSE_STRIDE] = mir[:, b0:b0 + POSE_STRIDE]
        mir[:, b0:b0 + POSE_STRIDE] = tmp

    # Flip x de pose frame a frame, solo si pose fue detectada
    n_pose_lm = (FACE_BASE - POSE_BASE) // POSE_STRIDE   # = 9
    for t in range(T):
        if np.any(mir[t, POSE_BASE:FACE_BASE] != 0):
            for i in range(n_pose_lm):
                x_idx = POSE_BASE + i * POSE_STRIDE
                mir[t, x_idx] = 1.0 - mir[t, x_idx]

    # ── 3. CARA ───────────────────────────────────────────────────────────────
    # Swap pares L/R
    for a, b in FACE_LR_PAIRS:
        a0 = FACE_BASE + a * FACE_STRIDE
        b0 = FACE_BASE + b * FACE_STRIDE
        tmp = mir[:, a0:a0 + FACE_STRIDE].copy()
        mir[:, a0:a0 + FACE_STRIDE] = mir[:, b0:b0 + FACE_STRIDE]
        mir[:, b0:b0 + FACE_STRIDE] = tmp

    # Negar x en todos los landmarks de cara
    # (coords relativas al centro → espejo = negar x)
    for i in range(N_FACE_LM):
        x_idx = FACE_BASE + i * FACE_STRIDE
        mir[:, x_idx] = -mir[:, x_idx]

    return mir


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def verify_mirror(seq: np.ndarray, tol: float = 1e-5) -> bool:
    """
    Verifica que mirror(mirror(seq)) ≈ seq.
    Si el espejo es correcto, aplicarlo dos veces devuelve la secuencia original.
    """
    double = mirror_sequence(mirror_sequence(seq))
    ok = np.allclose(seq, double, atol=tol)
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO DEL CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def process_corpus(corpus_dir: str, dry_run: bool, run_verify: bool) -> None:

    # Buscar todos los .npy del corpus que NO sean ya espejos
    pattern = os.path.join(corpus_dir, "**", "*.npy")
    all_files = sorted(glob.glob(pattern, recursive=True))
    source_files = [f for f in all_files if "_mirror" not in f]

    if not source_files:
        print(f"[AVISO] No se encontraron archivos .npy en '{corpus_dir}'")
        return

    print(f"── Augmentation por espejo ─────────────────────────────────────")
    print(f"   Corpus       : {corpus_dir}")
    print(f"   Archivos base: {len(source_files)}")
    print(f"   Modo         : {'DRY RUN (no escribe)' if dry_run else 'ESCRITURA'}")
    print()

    generated  = 0
    skipped    = 0
    errors     = 0
    verify_ok  = 0
    verify_fail= 0

    for fpath in source_files:
        mirror_path = fpath.replace(".npy", "_mirror.npy")
        fname       = os.path.relpath(fpath, corpus_dir)

        # Saltar si el espejo ya existe
        if os.path.exists(mirror_path):
            print(f"  ~ ya existe  {fname}")
            skipped += 1
            continue

        # Cargar
        try:
            seq = np.load(fpath).astype(np.float32)
        except Exception as e:
            print(f"  ✗ error al leer {fname}: {e}")
            errors += 1
            continue

        # Validar shape
        if seq.ndim != 2 or seq.shape[1] != FEATURE_DIM:
            print(f"  ✗ shape inválido {seq.shape}  →  {fname}")
            errors += 1
            continue

        # Espejo
        mir = mirror_sequence(seq)

        # Verificación opcional
        if run_verify:
            ok = verify_mirror(seq)
            if ok:
                verify_ok += 1
            else:
                print(f"  ⚠ verificación FALLÓ en {fname}")
                verify_fail += 1

        # Guardar
        mirror_rel = os.path.relpath(mirror_path, corpus_dir)
        if dry_run:
            print(f"  → [dry] generaría  {mirror_rel}  shape={mir.shape}")
        else:
            np.save(mirror_path, mir)
            print(f"  ✓ generado  {mirror_rel}  shape={mir.shape}")

        generated += 1

    # ── Resumen ───────────────────────────────────────────────────────────────
    print()
    print(f"── Resumen ─────────────────────────────────────────────────────")
    print(f"   Generados  : {generated}")
    print(f"   Ya existían: {skipped}")
    print(f"   Errores    : {errors}")
    if run_verify:
        print(f"   Verificados OK   : {verify_ok}")
        print(f"   Verificados FAIL : {verify_fail}")

    if not dry_run and generated > 0:
        total = len(source_files) + generated
        print()
        print(f"   El corpus pasó de {len(source_files)} → {total} archivos")
        print(f"   (×{total/len(source_files):.1f} aumento de datos)")
        print()

        # Contar por seña
        labels = set()
        for f in source_files:
            parts = os.path.normpath(f).split(os.sep)
            if len(parts) >= 2:
                labels.add(parts[-2])
        print(f"   Señas en el corpus: {sorted(labels)}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LSA — Data augmentation por espejo horizontal",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--corpus", default="corpus",
        help="Ruta al directorio raíz del corpus (default: corpus/)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Mostrar qué se haría sin escribir ningún archivo"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verificar que mirror(mirror(x)) ≈ x para cada archivo"
    )
    args = parser.parse_args()

    process_corpus(
        corpus_dir = args.corpus,
        dry_run    = args.dry_run,
        run_verify = args.verify,
    )
