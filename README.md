# Real-Time Vishing Defence (prototype)

A Deep Learning college project that flags **possibly synthetic (AI-generated or cloned) voices** in a call
recording. It can also combine that with a simple **scam-language check** on a transcript, to give the listener
an advisory warning.

> **Important:** this prototype analyses **audio recordings**: an uploaded file, a demo clip, or a short take
> recorded from this computer's microphone in the browser. It does **not** connect to, record or intercept phone
> calls or telephone-line audio. "Real-time" means the recording is processed in short sliding windows, the way a
> live stream would be. The app is advisory only and never blocks anything.

## Features

- **Three voice anti-spoofing models**, trained and compared on ASVspoof 2019 LA (`vishing_experiments.py`):
  MFCC + SVM, a small CNN on log-mel spectrograms, and frozen Wav2Vec2 + logistic regression.
- **Window-length and latency study** for the CNN (1-4 s windows).
- **Streamlit dashboard** (`DL_Lab_ESE/realtime_demo.py`) using the trained CNN:
  - sliding-window scoring of a WAV/FLAC file, with a timeline chart and adjustable window, step and smoothing
  - short recordings (≥ 1 s) are wrap-padded exactly like the training data
  - invalid input (undecodable, empty, silent, NaN samples, < 1 s) is rejected with a message and never scored
  - optional transcript → toy scam-language score → hand-weighted combined score
  - all outputs are labelled as **uncalibrated scores**, not probabilities (see *Reading the outputs*)

## Folder structure

```
Real Time Vishing Defence/
├── README.md                       this file
├── requirements.txt                pinned Python dependencies
├── .streamlit/config.toml          dashboard theme (used when launched from this folder)
├── PROJECT_AUDIT.md                audit of the original project (Phase 0)
├── vishing_experiments.py          training + evaluation script (Google Colab, GPU)
├── Real_Time_Vishing_Defence.ipynb Colab notebook that runs the script and builds the demo clips
├── results_in_domain.csv           EER / accuracy / F1 of the three models (1000-clip eval subset)
├── results_chunk.csv               CNN accuracy / EER / GPU latency for 1-4 s windows
├── demo_clips/                     sample audio from the ASVspoof 2019 LA eval split (16 kHz mono)
│   ├── real_0..2.wav               genuine speech (2.0-3.2 s)
│   ├── fake_0..2.wav               spoofed speech (2.4-8.3 s)
│   └── call_real_then_fake.wav     22.6 s "call": 4 genuine utterances, then 4 spoofed ones
├── DL_Lab_ESE/
│   ├── realtime_demo.py            Streamlit dashboard (UI)
│   ├── vishing_core.py             model, audio loading/validation, windowing, scoring (no UI)
│   ├── scam_text.py                TF-IDF + logistic-regression scam-language classifier (toy data)
│   ├── test_vishing_core.py        unit tests for vishing_core.py
│   └── cnn.pt                      trained CNN weights (state_dict, 23,585 parameters)
├── _backup_phase1_original/        untouched copy of the original realtime_demo.py + checksums
└── _backup_phase2_ui/              Phase 1 versions of the files changed by the UI redesign + checksums
```

The ASVspoof 2019 LA dataset is **not** included. It is only needed to retrain (see *Retraining*).

## Installation (Windows)

You need Python **3.12 or newer**. Check with `py --version`. Open **Command Prompt** in the project folder:

```bat
cd "%USERPROFILE%\OneDrive\Desktop\Real Time Vishing Defence"
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`torch` and `torchaudio` are pinned to the **same release** (2.11.0). Mismatched versions fail at import, so
don't upgrade one without the other.

## Running the dashboard

With the virtual environment active:

```bat
streamlit run "DL_Lab_ESE\realtime_demo.py"
```

Streamlit opens `http://localhost:8501` in your browser. Launch from the project folder (as above) so the
charcoal/off-white theme in `.streamlit/config.toml` is applied; from anywhere else the app still works with
Streamlit's default theme. The model file is found next to `realtime_demo.py`,
so the command works from any folder. If `cnn.pt` is missing, the page shows the expected path instead of
crashing.

Optional self-check (about 1 second, no dataset needed):

```bat
python "DL_Lab_ESE\test_vishing_core.py" -v
```

## Testing with the sample audio

1. Start the dashboard and click **Browse files** under *01 Call recording*.
2. Pick a file from `demo_clips\`:
   - `call_real_then_fake.wav`: the timeline should stay low for the first ~12 s and rise in the second half.
     This is the best demo.
   - `real_*.wav` / `fake_*.wav`: single utterances. Most are shorter than the default 3 s window, so the page
     says they were wrap-padded and scores one window.
3. Optionally paste a transcript in *02 Transcript* (for example *"This is your bank, share the OTP now"*) and press
   Ctrl+Enter. The scam-language and combined score cards fill in, and the voice-weight slider becomes active.
4. Try the sliders in *03 Analysis settings*. Under *04 Results*, the timeline can show the 0.4 / 0.7 heuristic band
   lines, and the expanders hold window-by-window scores and technical details.

What the dashboard showed for the clips during Phase 1 testing (3 s window, 1 s step, alpha 0.4, no transcript):

| Clip | Windows | Latest smoothed voice score |
|---|---|---|
| real_0 (2.0 s, padded) | 1 | 0.02 |
| real_2 (3.15 s) | 1 | 0.05 |
| fake_0 (2.39 s, padded) | 1 | 0.96 |
| fake_1 (8.26 s) | 6 | 0.51 |
| call_real_then_fake (22.6 s) | 20 | 0.84 |

## Microphone demo (record a take)

Choose **Record from microphone** under *01 Call recording*, press the microphone button, speak for 4-8 s, then
press stop. The take is analysed after recording stops, through exactly the same checks and scoring as an uploaded
file (minimum 1 s, silence and corruption checks, 3 s windows, 0.5 boundary, 0.4 / 0.7 bands). It uses Streamlit's
built-in `st.audio_input`; nothing extra is installed.

- Open the app at **http://localhost:8501** on the same computer. Browsers only allow the microphone on
  `localhost` or HTTPS, so the "Network URL" opened from another device will not get microphone access.
- Allow microphone access when the browser asks, and in Windows Settings > Privacy & security > Microphone.
- This is **not** continuous streaming and **not** phone-call monitoring: one short take is recorded, then scored.
- **Expect unreliable scores.** Laptop-microphone audio (room noise, browser echo cancellation / noise suppression,
  no leading silence) is far from the ASVspoof recordings the model was trained on. In a simulated browser test a
  *genuine* clip scored 0.74 (HIGH) through the microphone path, versus 0.02 when uploaded. Treat a microphone
  score as a demonstration of the pipeline, not a verdict.

## Reading the outputs

- **Voice score (0-1)** is the CNN's sigmoid output. It is **uncalibrated**: it ranks audio by how much it
  resembles ASVspoof synthetic speech, but 0.8 does **not** mean "80 % chance of fraud".
- The only threshold with measured performance is the model's **0.5 decision boundary**. At that point the CNN
  had 86.7 % accuracy and 8.4 % EER on 4 s windows (87.4 % / 8.7 % on 3 s) on a 1000-clip ASVspoof eval
  subset. The dashboard shows the matching `results_chunk.csv` row for the chosen window.
- The **LOW / MEDIUM / HIGH bands (0.4 / 0.7)** come from the original prototype. They were hand-picked and
  **not validated**, and the dashboard says so.
- The **scam-language score** comes from a toy model trained on 24 self-written sentences. It is weakly
  separating (about 0.4-0.6 on most inputs).
- The **combined score** = `w × smoothed voice + (1 − w) × text`. It is a heuristic, not a probability.

## Retraining (not needed to run the dashboard)

`vishing_experiments.py` is written for **Google Colab with a GPU**. Put ASVspoof 2019 LA under `/content/LA`
and run it through `Real_Time_Vishing_Defence.ipynb`. It writes the two result CSVs and `cnn.pt`; copy `cnn.pt`
into `DL_Lab_ESE\`. The supplied results came from one run with seed 0.

## Known limitations

- **No live-call interception.** Only recordings are analysed (uploads, demo clips, or a short microphone take
  recorded in the browser). There is no telephony input and no continuous streaming.
- **No speech-to-text.** The transcript must be typed or pasted.
- **Uncalibrated scores and unvalidated bands** (see above). Calibrating on the ASVspoof `dev` split is future work.
- **The CNN misses about 25 % of spoofed clips at the 0.5 boundary.** This is inferred from the reported
  accuracy/F1; it rarely flags genuine speech. Example: `fake_1.wav` only reaches ~0.5.
- **Domain gap:** trained and tested only on ASVspoof 2019 LA (clean, 2019-era TTS/VC). Phone-line audio,
  noise and modern voice-cloning tools were not tested. The planned cloned-voice test (`results_out_domain.csv`)
  has not been run.
- **Short clips** (1 s up to one window) are wrap-padded, so the same audio is scored several times over.
  Results on them are less reliable. Clips under 1 s are rejected.
- The last partial window of a recording (shorter than one step) is not scored.
- Only the CNN is used in the dashboard. Wav2Vec2 scored best (5.4 % EER) but isn't deployed.
- Results come from a single seed on balanced subsets (3000 train / 1000 eval), not the full ASVspoof eval set.
- The latencies in `results_chunk.csv` are GPU (T4) timings. On CPU the CNN took about 3-8 ms per window in testing.
