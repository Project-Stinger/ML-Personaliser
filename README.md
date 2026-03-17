# Stinger ML Personaliser

Browser-based training tool for the [Stinger](https://github.com/Project-Stinger/Firmware) foam dart blaster's ML idle prediction system. Trains personalized models from your IMU data and uploads them to the blaster — no Python, no reflashing.

**Live:** [project-stinger.github.io/ML-Personaliser](https://project-stinger.github.io/ML-Personaliser/)

## What it does

The Stinger V2 has a 6-axis IMU (accel + gyro). The ML system learns your pre-shot motion pattern — the characteristic aim-stabilize-tense sequence you do before pulling the trigger — and pre-spins the flywheels ~100ms before you fire.

This web app handles the full personalization loop:

1. **Pull** binary IMU log from the blaster over Web Serial (`MLDUMP` protocol)
2. **Parse** the 17-byte/sample binary format (timestamp + 6-axis IMU + trigger)
3. **Detect shots** via trigger rising edges, filter bursts (1s minimum gap)
4. **Extract training windows** — 500ms windows ending 100ms before each trigger pull
5. **Sample negatives** — non-shot windows with motion, balanced against augmented positives
6. **Featurize** — summary stats (mean/std/absmax per axis) for LR, rich stats (+ magnitude) for MLP
7. **Train** both Logistic Regression and MLP (64→32→1) with Adam optimizer
8. **Evaluate** on a held-out stratified split with precision/recall/F1 metrics
9. **Upload** MLMD binary model files back to the blaster with CRC32 validation

No server, no install — everything runs in your browser using Web Serial (Chrome/Edge).

## Architecture

```
index.html          Entry point, UI layout
app.js              Serial I/O, UI rendering, plot drawing, upload/download flow
train.js            ML pipeline: parsing, shot detection, featurization, training, MLMD export
style.css           Dark theme + data quality widget styles
how.html            User-facing explanation page
```

### Data flow

```
Blaster (USB Serial)                    Browser
─────────────────                       ───────
                    ← "MLDUMP\n"
"MLDUMP1 <size>\n" →
<size bytes binary> →
"\nMLDUMP_DONE\n"   →
                                        parseBinaryLog()
                                        detectShots()
                                        extractDataset()    ← window jitter augmentation (5x positives)
                                        featurizeSummary()  ← mean/std/absmax per axis (18 features)
                                        featurizeRich()     ← + absmean + magnitude stats (30 features)
                                        fitScaler()         ← StandardScaler (mean/std normalization)
                                        trainLogReg()       ← gradient descent with adaptive L2
                                        trainMLP()          ← Adam optimizer, He init, ReLU, sigmoid output
                                        buildMlmdLR()       ← 24-byte header + float payload + CRC32
                                        buildMlmdMLP()

                    ← "MLMODEL_PUT_LR <size> <crc32hex>\n"
"MLMODEL_READY\n"  →
                    ← <size bytes binary>
"MLMODEL_OK\n"     →
                    ← "MLMODEL_LOAD_LR\n"
"MLMODEL_LOADED\n" →
```

### Binary formats

**MlSample** (17 bytes, logged at 100Hz):
```
u32 timestamp_ms    (little-endian)
i16 ax, ay, az      (raw accelerometer)
i16 gx, gy, gz      (raw gyroscope)
u8  trigger          (0 or 1)
```

**MLMD model file** (header + payload):
```
Header (24 bytes):
  char[4]  magic          "MLMD"
  u16      version        1
  u16      windowSamples  50
  u8       modelType      0=LR, 1=MLP
  u8       reserved       0
  u16      features       18 (LR) or 30 (MLP)
  u16      h1             0 (LR) or 64 (MLP)
  u16      h2             0 (LR) or 32 (MLP)
  u32      payloadBytes
  u32      payloadCrc32

Payload (float32 little-endian):
  LR:  scalerMean[18] + scalerScale[18] + coef[18] + intercept[1]
  MLP: scalerMean[30] + scalerScale[30] + W1[64×30] + B1[64]
       + W2[32×64] + B2[32] + W3[32] + B3[1]
```

### Feature extraction

**LR (18 features)** — `featurizeSummary()`:
Per-axis (ax, ay, az, gx, gy, gz): mean, standard deviation, absolute max.

**MLP (30 features)** — `featurizeRich()`:
LR features + per-axis absolute mean + accel magnitude (mean/std/max) + gyro magnitude (mean/std/max).

Both match the firmware implementation in `src/mlInfer.cpp` exactly.

### Training

**Logistic Regression**: Gradient descent with decaying learning rate and adaptive L2 regularization (stronger for small datasets). 5000 iterations.

**MLP (30→64→32→1)**: He initialization, ReLU activations, sigmoid output, Adam optimizer (β1=0.9, β2=0.999), adaptive L2, optional Gaussian noise injection on small datasets. 600-700 iterations with interactive weight visualization.

### Dataset construction

- **Positive windows**: 500ms of IMU data ending 100ms before each trigger pull. Augmented 5x with ±1/±2 sample jitter.
- **Negative windows**: Sampled from non-shot regions with minimum 250ms separation. Count matches augmented positives for class balance.
- **Shot filtering**: Only first shot in a burst accepted (1000ms minimum gap between shots).

## Import / combine datasets

The "Import .bin log files" button accepts multiple `.bin` files. They are concatenated before training, enabling:

- **Session accumulation**: Record multiple sessions, download each `.bin`, import all together
- **Community datasets**: Share `.bin` files and train on combined data from multiple users

## Run locally

Web Serial requires a secure context. Serve this folder:

```sh
python3 -m http.server 8080
```

Open `http://localhost:8080` in Chrome or Edge.

## Firmware compatibility

Requires Stinger V2 firmware with `USE_ML_LOG` enabled (default in `env:v2`). The firmware provides:

- `mlLogLoop()` / `mlLogSlowLoop()` — 100Hz IMU logging to LittleFS
- `mlInferLoop()` / `mlInferSlowLoop()` — real-time inference on sliding window
- Serial commands: `MLDUMP`, `MLMODEL_PUT_LR/MLP`, `MLMODEL_LOAD_LR/MLP`, `MLMODEL_INFO`, `MLMODEL_DELETE`

Source: [Project-Stinger/Firmware](https://github.com/Project-Stinger/Firmware) (`src/mlLog.cpp`, `src/mlInfer.cpp`)
