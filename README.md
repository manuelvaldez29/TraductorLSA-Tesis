# Reconocimiento de LSA — Tesis UNSTA 2026

Sistema de reconocimiento automático de Lengua de Señas Argentina (LSA)
para mostradores de atención al público en Tucumán.

**Equipo:** Bloj Iván · Domfrocht Julián · Petrelli Francesco · Valdez Manuel  
**Institución:** UNSTA — Ingeniería en Informática

---

## Estructura del proyecto

```
lsa-tesis/
├── test_pipeline.py      # Prueba los tres landmarkers en tiempo real
├── record_sign.py        # Graba repeticiones de una seña al corpus
├── augment_mirror.py     # Genera versiones espejo del corpus
├── requirements.txt      # Dependencias Python
├── models/               # Modelos MediaPipe (se descargan automáticamente, no van en git)
└── corpus/               # Grabaciones .npy (se comparte por OneDrive, no va en git)
```

---

## Setup inicial (hacer una sola vez)

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/lsa-tesis.git
cd lsa-tesis
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Python recomendado:** 3.10 o 3.11

### 3. Descargar la carpeta `corpus/` desde OneDrive

Pedir acceso a Manu. Descargar la carpeta `corpus/` y ubicarla
dentro de la carpeta del repo, de forma que quede así:

```
lsa-tesis/
└── corpus/
    └── hola/
        ├── manu1_rep001.npy
        └── ...
```

### 4. Verificar que todo funciona

```bash
python test_pipeline.py
```

Debería abrir la webcam con los landmarks de manos, pose y cara superpuestos.
La primera vez descarga automáticamente los modelos en `models/` (~30 MB en total).

---

## Grabar señas

Cada integrante usa su propio signer ID al grabar:

| Integrante | Signer ID |
|---|---|
| Manu Valdez | `manu1` |
| Iván Bloj | `ivan1` |
| Julián Domfrocht | `julian1` |
| Francesco Petrelli | `fran1` |

### Comando de grabación

```bash
python record_sign.py --label NOMBRE_SEÑA --signer TU_ID
```

**Ejemplo:**
```bash
python record_sign.py --label hola --signer ivan1
```

Esto graba 15 repeticiones por defecto. Para cambiar la cantidad:

```bash
python record_sign.py --label hola --signer ivan1 --reps 20
```

### Controles durante la grabación

| Tecla | Acción |
|---|---|
| `SPACE` | Iniciar grabación de la siguiente repetición |
| `R` | Descartar la última repetición y repetirla |
| `Q` | Salir (guarda lo grabado hasta ese momento) |

### Dónde se guardan

```
corpus/
└── hola/
    ├── ivan1_rep001.npy    ← (30 frames × 237 features)
    ├── ivan1_rep002.npy
    └── ...
```

---

## Generar versiones espejo

Después de grabar, correr el script de augmentation para duplicar el corpus
generando automáticamente la versión de mano contraria de cada seña:

```bash
# Ver qué va a generar sin escribir nada
python augment_mirror.py --dry-run

# Generar espejos con verificación matemática
python augment_mirror.py --verify
```

Cada `ivan1_rep001.npy` genera un `ivan1_rep001_mirror.npy` con la seña espejada.

---

## Subir grabaciones al OneDrive compartido

Una vez grabadas las señas y generados los espejos, subir la carpeta `corpus/`
(o solo las subcarpetas nuevas) al OneDrive compartido del equipo.

**Ruta en OneDrive:** `TESIS/corpus/`

---

## Estructura del vector de features

Cada frame se representa como un vector de **237 valores float32**:

```
[0   : 126]  Manos   — 2 manos × 21 keypoints × (x, y, z)
[126 : 162]  Pose    — 9 keypoints upper body × (x, y, z, visibilidad)
[162 : 237]  Cara    — 25 keypoints no-manuales × (x, y, z) normalizados
```

Cada repetición grabada tiene shape `(30, 237)` — 30 frames por seña.

---

## Cargar el corpus para entrenamiento

```python
import numpy as np
import glob

X, y = [], []
for label in ["hola", "gracias"]:   # agregar señas según avance el corpus
    for fpath in sorted(glob.glob(f"corpus/{label}/*.npy")):
        X.append(np.load(fpath))    # (30, 237)
        y.append(label)

X = np.array(X)   # (N, 30, 237)
print(f"Dataset: {X.shape}  —  {len(set(y))} señas")
```

---

## Dependencias

| Librería | Versión mínima | Para qué |
|---|---|---|
| `mediapipe` | 0.10.14 | Hand, Pose y Face Landmarker |
| `opencv-python` | 4.9.0 | Captura de video y visualización |
| `numpy` | 1.26.0 | Manejo de arrays y guardado .npy |
