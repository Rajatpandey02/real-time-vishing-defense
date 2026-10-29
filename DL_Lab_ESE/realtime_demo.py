# realtime_demo.py
# Streamlit dashboard: simulates a live call by scoring an uploaded recording in sliding windows.
# It analyses uploaded files, demo clips or a short microphone recording made in the browser.
# It does not connect to, capture or intercept phone calls or telephone-line audio.
#
# Run from the project root so that .streamlit/config.toml (theme) is picked up:
#   pip install -r requirements.txt
#   streamlit run DL_Lab_ESE/realtime_demo.py
# Keep cnn.pt, scam_text.py, vishing_core.py and this file in the same folder.
#
# Phase 1 (backup: _backup_phase1_original/): short-clip padding, invalid-audio rejection,
#   model path relative to this file, honest score labelling.
# Phase 5 (backup: _backup_phase5_validation/): input checks for near-silent, constant (DC) and truncated audio,
#   via vishing_core.inspect_audio. Scores, thresholds and the model are unchanged.
# Phase 6 (backup: _backup_phase6_ui_refresh/): presentation only - demo-clip picker with playback, threshold labels
#   on every score, per-window verdict strip, always-visible limitations. inspect_audio checks, windowing,
#   scores, smoothing, fusion and thresholds are unchanged.
# Phase 7 (backup: _backup_phase7_mic/): third input mode "Record from microphone" (st.audio_input, built into
#   Streamlit; no new package). The recording goes through the same inspect_audio -> make_windows -> score_windows
#   -> ema path as an upload. Upload and demo modes, scores and thresholds are unchanged.
# Phase 8 (backup: _backup_phase8_ui_polish/): presentation only - compact header, result summary beside the
#   settings (first screen), help tooltips on the three score cards, consistent 2-decimal scores, chart axis/legend
#   tidy-up, clearer limitations. The analysis block and every score, threshold and formula are unchanged.
# Phase 2 (backup: _backup_phase2_ui/): presentation only. Detection, preprocessing, windowing,
#   scores, thresholds, weights and advice logic are unchanged; all of it lives in vishing_core.py
#   and in the "analysis" block below, which uses the same calls and formulas as Phase 1.
import html
import io
from pathlib import Path

import altair as alt
import pandas as pd
import soundfile as sf
import streamlit as st

from scam_text import scam_score
from vishing_core import (DECISION_BOUNDARY, HEURISTIC_BANDS, MIN_SECONDS, SR, AudioError, ModelLoadError,
                          ema, heuristic_band, inspect_audio, load_model, make_windows, score_windows)

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "cnn.pt"
CHUNK_RESULTS = HERE.parent / "results_chunk.csv"   # evaluation evidence from vishing_experiments.py (read-only)
DEMO_DIR = HERE.parent / "demo_clips"
# Demo catalogue: ground-truth SOURCE labels from the ASVspoof 2019 LA protocol (verified in Phase 2D).
# These describe where each clip comes from; they are not model results, which are always computed live.
DEMOS = [
    ("real_0.wav", "Genuine speech", "Bona fide utterance from the ASVspoof 2019 LA evaluation set."),
    ("real_2.wav", "Genuine speech", "Bona fide utterance from the ASVspoof 2019 LA evaluation set."),
    ("fake_0.wav", "Synthetic speech", "Spoofed utterance, attack A16 (waveform-concatenation TTS)."),
    ("fake_1.wav", "Synthetic speech", "Spoofed utterance, attack A15 (TTS/VC hybrid), an attack type the robustness "
                                       "audit found hard for this model."),
    ("call_real_then_fake.wav", "Combined call", "Four genuine utterances followed by four synthetic ones "
                                                 "(attacks A17, A16, A08, A07), about 22.6 s."),
]


class NamedAudio(io.BytesIO):
    """Demo clip read from disk, shaped like a Streamlit upload (name, size, read/seek)."""
    def __init__(self, path):
        data = Path(path).read_bytes()
        super().__init__(data)
        self.name, self.size = Path(path).name, len(data)

# ----------------------------------------------------------------------------- look & feel
INK, MUTED, LINE, PAPER, CARD = "#1C1D20", "#6A6862", "#E3DFD6", "#F6F4EF", "#FFFFFF"
ACCENT, RED, AMBER, SLATE = "#D4582A", "#B3362B", "#B7791F", "#4A6658"
BAND_COLOR = {"LOW": SLATE, "MEDIUM": AMBER, "HIGH": RED}
SERIES = ["Raw window", "Smoothed voice", "Combined"]
SERIES_COLOR = ["#B8B2A7", ACCENT, INK]
SANS = "IBM Plex Sans, Segoe UI, system-ui, sans-serif"   # chart fonts, with fallbacks
MONO = "IBM Plex Mono, Consolas, ui-monospace, monospace"

st.set_page_config(page_title="Synthetic Speech Detection | Vishing Defence", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [data-testid="stApp"] {{ font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
.block-container {{ max-width: 1180px; padding-top: 1.1rem; padding-bottom: 4rem; }}
h1, h2, h3 {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; letter-spacing: -0.01em; }}
[data-testid="stExpander"] details {{ border-color: {LINE}; border-radius: 6px; }}

.vd-header {{ border-bottom: 1px solid {LINE}; padding-bottom: .9rem; margin-bottom: 1.3rem; }}
.vd-eyebrow {{ font: 500 .74rem/1 'IBM Plex Mono', ui-monospace, monospace; letter-spacing: .14em;
              text-transform: uppercase; color: {ACCENT}; margin-bottom: .55rem; }}
.vd-title {{ font-size: clamp(1.55rem, 2.6vw, 2rem); font-weight: 700; color: {INK}; line-height: 1.15; margin: 0; padding: 0; }}
.vd-lede {{ color: {MUTED}; font-size: .97rem; max-width: 54rem; margin: .45rem 0 .75rem; line-height: 1.55; }}
.vd-tags {{ display: flex; flex-wrap: wrap; gap: .4rem; }}
.vd-tag {{ font: 500 .72rem/1 'IBM Plex Mono', ui-monospace, monospace; color: {INK}; background: #EDE9E1;
          border-radius: 3px; padding: .38rem .55rem; }}

.vd-step {{ display: flex; align-items: baseline; gap: .6rem; margin: .2rem 0 .45rem; }}
.vd-step b {{ font: 500 .78rem/1 'IBM Plex Mono', ui-monospace, monospace; color: {ACCENT}; }}
.vd-step span {{ font-weight: 600; font-size: 1.02rem; color: {INK}; }}
.vd-hint {{ color: {MUTED}; font-size: .84rem; margin: -.15rem 0 .5rem; line-height: 1.45; }}

.vd-file {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(105px, 1fr)); gap: .1rem .9rem;
           border: 1px solid {LINE}; background: {CARD}; border-radius: 6px; padding: .7rem .9rem; margin: .5rem 0 1.3rem; }}
.vd-file div {{ min-width: 0; }}
.vd-file div:first-child {{ grid-column: span 2; }}
.vd-file small {{ display: block; font: 500 .66rem/1.6 'IBM Plex Mono', ui-monospace, monospace;
                 letter-spacing: .08em; text-transform: uppercase; color: {MUTED}; }}
.vd-file span {{ font-size: .9rem; color: {INK}; overflow-wrap: anywhere; }}

.vd-verdict {{ border-left: 4px solid var(--c); background: {CARD}; border-radius: 0 6px 6px 0;
              padding: .85rem 1.1rem; margin: .1rem 0 1rem; box-shadow: 0 1px 0 {LINE}; }}
.vd-verdict .k {{ font: 500 .68rem/1.5 'IBM Plex Mono', ui-monospace, monospace; letter-spacing: .05em;
                 text-transform: uppercase; color: {MUTED}; }}
.vd-verdict .h {{ font-size: 1.7rem; font-weight: 700; color: var(--c); margin: .2rem 0 .3rem; letter-spacing: .02em; }}
.vd-verdict p {{ margin: .15rem 0; color: {INK}; font-size: .93rem; line-height: 1.5; }}
.vd-verdict p.m {{ color: {MUTED}; font-size: .82rem; }}

.vd-note {{ background: #FBF3E6; border: 1px solid #EED9B5; color: #6B4A12; border-radius: 6px;
           padding: .65rem .9rem; font-size: .88rem; margin-bottom: 1rem; line-height: 1.5; }}
.vd-error {{ background: #FBEDEA; border: 1px solid #EBC3BC; border-left: 4px solid {RED}; border-radius: 0 6px 6px 0;
            padding: .9rem 1.1rem; margin: .4rem 0 1rem; }}
.vd-error .h {{ font-weight: 700; color: {RED}; margin-bottom: .25rem; }}
.vd-error p {{ margin: .15rem 0; color: {INK}; font-size: .9rem; line-height: 1.5; }}
.vd-empty {{ border: 1px dashed #CFC9BD; border-radius: 6px; padding: 1.6rem 1.2rem; text-align: center; color: {MUTED}; }}
.vd-empty b {{ color: {INK}; }}
.vd-empty code {{ font-size: .82rem; }}

.vd-card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 6px; padding: 1rem 1.1rem 1.05rem;
           height: 100%; }}
@media (min-width: 900px) {{ .vd-card {{ min-height: 282px; }} }}
.vd-card .k {{ font: 500 .7rem/1 'IBM Plex Mono', ui-monospace, monospace; letter-spacing: .12em;
              text-transform: uppercase; color: {MUTED}; }}
.vd-card .v {{ font: 500 2.35rem/1.1 'IBM Plex Mono', ui-monospace, monospace; color: {INK}; margin: .55rem 0 .15rem; }}
.vd-card .v.off {{ color: #B9B4AA; font-size: 1.15rem; font-family: 'IBM Plex Sans', system-ui, sans-serif;
                  font-weight: 600; margin: .9rem 0 .4rem; }}
.vd-card .u {{ font-size: .72rem; color: {MUTED}; }}
.vd-card p {{ font-size: .84rem; color: {INK}; margin: .3rem 0 0; line-height: 1.45; }}
.vd-card p.m {{ color: {MUTED}; }}
.vd-bar {{ position: relative; height: 6px; background: #ECE8E0; border-radius: 3px; margin: .7rem 0 .2rem; }}
.vd-bar i {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; background: var(--c); }}
.vd-bar em {{ position: absolute; top: -4px; bottom: -4px; width: 2px; background: {INK}; opacity: .55; }}
.vd-scale {{ display: flex; justify-content: space-between; font: .64rem/1 'IBM Plex Mono', ui-monospace, monospace;
            color: #A7A196; }}
.vd-section {{ margin: 1.9rem 0 .7rem; padding-top: 1.1rem; border-top: 1px solid {LINE}; }}
.vd-help {{ display: inline-block; width: 15px; height: 15px; border: 1px solid #BDB6AA; border-radius: 50%;
           font: 500 .62rem/13px 'IBM Plex Mono', ui-monospace, monospace; text-align: center; color: {MUTED};
           letter-spacing: 0; text-transform: none; cursor: help; margin-left: .4rem; vertical-align: 1px; }}
.vd-card .d {{ font-size: .8rem; color: {MUTED}; margin: .3rem 0 0; line-height: 1.4; }}
.vd-distinct {{ font-size: .86rem; color: {INK}; background: #EFEBE3; border-radius: 4px; padding: .55rem .85rem;
               margin: .1rem 0 .9rem; line-height: 1.5; }}
.vd-placeholder {{ border: 1px dashed #CFC9BD; border-radius: 6px; padding: .9rem 1rem; color: {MUTED};
                  font-size: .88rem; line-height: 1.45; }}
.vd-placeholder b {{ color: {INK}; }}
.vd-src {{ border-left: 3px solid {LINE}; padding: .35rem .7rem; margin: .3rem 0 .6rem; font-size: .86rem; color: {INK}; }}
.vd-src small {{ display: block; font: 500 .66rem/1.6 'IBM Plex Mono', ui-monospace, monospace; letter-spacing: .08em;
               text-transform: uppercase; color: {MUTED}; }}
.vd-strip {{ display: flex; flex-wrap: wrap; gap: 3px; margin: .35rem 0 .3rem; }}
.vd-strip span {{ flex: 1 1 26px; min-width: 26px; max-width: 64px; height: 26px; border-radius: 3px; position: relative;
                 font: 500 .62rem/26px 'IBM Plex Mono', ui-monospace, monospace; text-align: center; color: #fff; }}
.vd-strip span.lo {{ background: #A9B8AE; }}
.vd-strip span.hi {{ background: {ACCENT}; }}
.vd-strip-legend {{ font-size: .78rem; color: {MUTED}; margin-bottom: 1rem; }}
.vd-strip-legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 .3rem 0 .8rem;
                     vertical-align: -1px; }}
.vd-limits {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1.4rem;
             border-top: 1px solid {LINE}; margin: 2.2rem 0 1.4rem; padding-top: 1.1rem; }}
.vd-limits h4 {{ font: 600 .9rem/1.3 'IBM Plex Sans', system-ui, sans-serif; color: {INK}; margin: 0 0 .35rem; }}
.vd-limits p {{ font-size: .84rem; color: #5B5953; line-height: 1.55; margin: 0 0 .35rem; }}
.vd-limits b {{ color: {INK}; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


def esc(s):
    return html.escape(str(s))


def step_heading(num, title, hint=None):
    st.markdown(f'<div class="vd-step">{f"<b>{num}</b>" if num else ""}<span>{esc(title)}</span></div>'
                + (f'<div class="vd-hint">{esc(hint)}</div>' if hint else ""), unsafe_allow_html=True)


def error_panel(title, message, fix=None):
    st.markdown(f'<div class="vd-error"><div class="h">{esc(title)}</div><p>{esc(message)}</p>'
                + (f'<p>{esc(fix)}</p>' if fix else "") + "</div>", unsafe_allow_html=True)


def score_bar(value, color):
    """0-1 track with the model's decision boundary marked. Purely a picture of the number."""
    pct = max(0.0, min(1.0, value)) * 100
    b = DECISION_BOUNDARY * 100
    return (f'<div class="vd-bar" style="--c:{color}"><i style="width:{pct:.1f}%"></i>'
            f'<em style="left:calc({b:.0f}% - 1px)" title="decision boundary {DECISION_BOUNDARY}"></em></div>'
            f'<div class="vd-scale"><span>0</span><span>{DECISION_BOUNDARY}</span><span>1</span></div>')


def card(label, value=None, color=INK, lines=(), off_text=None, measures=None, help_text=None):
    tip = (f'<span class="vd-help" title="{html.escape(help_text, quote=True)}">?</span>' if help_text else "")
    head = f'<div class="k">{esc(label)}{tip}</div>' + (f'<p class="d">{esc(measures)}</p>' if measures else "")
    if value is None:
        body = f'<div class="v off">{esc(off_text)}</div>'
    else:
        body = (f'<div class="v">{value:.2f}<span class="u"> / 1</span></div>' + score_bar(value, color))
    body += "".join(f'<p class="{"m" if muted else ""}">{txt}</p>' for txt, muted in lines)
    return f'<div class="vd-card">{head}{body}</div>'


def fmt_size(n):
    if n < 1024:
        return f"{n} bytes"
    return f"{n / 1024:.0f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:.1f} MB"


def file_panel(f, x=None, orig_sr=None):
    """Selected-file facts. Format/channels come from the file header (display only)."""
    items = [("File", esc(f.name)), ("Size", fmt_size(f.size))]
    if x is not None:
        try:
            f.seek(0); info = sf.info(f); f.seek(0)
            items += [("Duration", f"{len(x) / SR:.2f} s"), ("Sample rate", f"{orig_sr:,} Hz"),
                      ("Channels", "mono" if info.channels == 1 else f"{info.channels} (mixed to mono)"),
                      ("Format", esc(info.format))]
        except Exception:
            items += [("Duration", f"{len(x) / SR:.2f} s"), ("Sample rate", f"{orig_sr:,} Hz")]
    else:
        items += [("Status", "not analysed (see Result summary)")]
    return '<div class="vd-file">' + "".join(f"<div><small>{k}</small><span>{v}</span></div>" for k, v in items) + "</div>"


def timeline(df_long, show_heuristic):
    t0, t1 = float(df_long["time"].min()), float(df_long["time"].max())
    y = alt.Y("score:Q", title="score (uncalibrated)", scale=alt.Scale(domain=[0, 1]),
              axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1], format=".2f"))
    x = alt.X("time:Q", title="time in recording (s, window end)", axis=alt.Axis(tickMinStep=1, format="d"),
              scale=alt.Scale(domain=[t0 - 0.5, t1 + 0.5], nice=False, zero=False))
    present = [s for s in SERIES if s in set(df_long["series"])]          # legend lists only plotted series
    idx = [SERIES.index(s) for s in present]
    color = alt.Color("series:N", title=None, sort=present,
                      scale=alt.Scale(domain=present, range=[SERIES_COLOR[i] for i in idx]),
                      legend=alt.Legend(orient="top", labelLimit=0, columns=2 if len(present) > 2 else len(present),
                                        columnPadding=14, labelFontSize=12))
    size = alt.Size("series:N", scale=alt.Scale(domain=present, range=[[1.2, 2.6, 2.2][i] for i in idx]), legend=None)
    tip = [alt.Tooltip("time:Q", title="time (s)", format=".1f"), alt.Tooltip("series:N", title="series"),
           alt.Tooltip("score:Q", format=".2f")]
    lines = alt.Chart(df_long).mark_line(interpolate="linear").encode(x=x, y=y, color=color, size=size)
    pts = alt.Chart(df_long).mark_circle(size=34, opacity=0.9).encode(x=x, y=y, color=color, tooltip=tip)
    rule_df = pd.DataFrame({"y": [DECISION_BOUNDARY], "label": [f"decision boundary {DECISION_BOUNDARY}"]})
    rule = alt.Chart(rule_df).mark_rule(color=INK, strokeDash=[2, 3], opacity=0.55).encode(y="y:Q")
    rule_txt = alt.Chart(rule_df).mark_text(align="left", dx=4, dy=-6, fontSize=10, color=MUTED).encode(
        y="y:Q", x=alt.value(0), text="label:N")
    layers = [rule, rule_txt, lines, pts]
    if show_heuristic:
        h = pd.DataFrame({"y": list(HEURISTIC_BANDS), "label": [f"heuristic {v}" for v in HEURISTIC_BANDS]})
        layers.insert(0, alt.Chart(h).mark_rule(color="#C9C3B8", strokeDash=[1, 4]).encode(y="y:Q"))
        layers.insert(1, alt.Chart(h).mark_text(align="left", dx=4, dy=-6, fontSize=10, color="#A7A196").encode(
            y="y:Q", x=alt.value(0), text="label:N"))
    return (alt.layer(*layers).properties(height=300)
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#EEEAE2", domainColor=LINE,
                            tickColor=LINE, labelFont=MONO, titleFont=SANS,
                            titleFontWeight=500, labelFontSize=11, titleFontSize=12, titlePadding=8)
            .configure_legend(labelFont=SANS, labelColor=INK, symbolStrokeWidth=3))


@st.cache_resource
def get_model(path_str):
    return load_model(path_str)


@st.cache_data
def chunk_evidence(path_str):
    try:
        return pd.read_csv(path_str).set_index("Window (s)")
    except Exception:
        return None


H_VOICE = 'CNN output for the latest smoothed window: how closely the audio resembles the synthetic speech in ASVspoof 2019 LA. Uncalibrated, so not a probability, and not evidence of scam intent.'
H_TEXT = 'Toy TF-IDF + logistic-regression model trained on 24 self-written sentences. Scores the wording only; it is not a validated fraud classifier.'
H_COMB = 'Hand-weighted average: w x voice + (1 - w) x transcript. It drives the 0.4 / 0.7 bands, which were hand-picked and not validated. Not a fraud probability.'

# ----------------------------------------------------------------------------- header
st.markdown("""
<div class="vd-header">
  <div class="vd-eyebrow">Vishing Defence &middot; Deep Learning prototype</div>
  <h1 class="vd-title">Synthetic Speech Detection</h1>
  <p class="vd-lede">Analyses audio recordings for synthetic-speech patterns with a CNN trained on
  ASVspoof&nbsp;2019&nbsp;LA. An optional transcript gets a separate check for scam-style wording.
  Results are advisory: verify any unusual request through a trusted channel.</p>
  <div class="vd-tags"><span class="vd-tag">Recording analysis, not live phone-call monitoring</span>
  <span class="vd-tag">Scores are uncalibrated</span><span class="vd-tag">Advisory only</span></div>
</div>
""", unsafe_allow_html=True)

try:
    model = get_model(str(MODEL_PATH))
except ModelLoadError as e:
    error_panel("The voice model could not be loaded", str(e),
                "Nothing can be analysed until cnn.pt is in place. Restart the app after copying it.")
    st.stop()

# ----------------------------------------------------------------------------- inputs
left, right = st.columns([1.45, 1], gap="large")
with left:
    step_heading("01", "Audio input", f"Upload a WAV or FLAC file, pick a demo clip, or record a short take. At least "
                 f"{MIN_SECONDS:.0f} s; stereo is mixed to mono and audio is resampled to 16 kHz, as in training.")
    demos = [d for d in DEMOS if (DEMO_DIR / d[0]).is_file()]
    modes = ["Upload a recording"] + (["Use a demo clip"] if demos else []) + ["Record from microphone"]
    source_mode = st.radio("Input source", modes, horizontal=True, label_visibility="collapsed")
    mic_mode = source_mode == "Record from microphone"
    if source_mode == "Use a demo clip" and demos:
        names = [d[0] for d in demos]
        pick = st.selectbox("Demo clip", names, format_func=lambda n: f"{n}  -  {dict((d[0], d[1]) for d in demos)[n]}")
        meta = {d[0]: d for d in demos}[pick]
        st.markdown(f'<div class="vd-src"><small>Source label (from the dataset, not a model result)</small>'
                    f'<b>{esc(meta[1])}</b>: {esc(meta[2])}</div>', unsafe_allow_html=True)
        f = NamedAudio(DEMO_DIR / pick)
    elif mic_mode:
        st.markdown('<div class="vd-src"><small>Microphone recording, not a phone call</small>'
                    'Records a short take from <b>this computer&#39;s microphone</b> in the browser and analyses it '
                    'after you press stop, exactly like an uploaded file. It does <b>not</b> connect to, capture or '
                    'intercept telephone calls or telephone-line audio. Laptop-microphone audio differs from the '
                    'ASVspoof recordings the model was trained on, so treat the score as a demonstration, not a '
                    f'verdict. Speak for 4-8 s (at least {MIN_SECONDS:.0f} s).</div>', unsafe_allow_html=True)
        f = st.audio_input("Microphone recording", sample_rate=SR, label_visibility="collapsed")
    else:
        f = st.file_uploader("Call recording", type=["wav", "flac"], label_visibility="collapsed")
    file_slot = st.empty()
    player_slot = st.empty()
    step_heading("02", "Transcript (optional)", "What the caller said, typed or pasted. Used for the "
                 "scam-language score; press Ctrl+Enter or click outside the box to apply.")
    transcript = st.text_area("Transcript", height=96, label_visibility="collapsed",
                              placeholder="e.g. This is your bank. Your account will be blocked unless you share the OTP.")
has_text = bool(transcript.strip())

with right:
    step_heading("03", "Analysis settings")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        win = c1.slider("Window length (s)", 1, 4, 3,
                        help="Seconds of audio per score. Clips shorter than this are wrap-padded to one window.")
        step = c2.slider("Window step (s)", 1, 2, 1, help="How far the window moves between scores.")
        alpha = st.slider("Smoothing (EMA alpha)", 0.1, 1.0, 0.4,
                          help="Weight of the newest window in the smoothed score. 1.0 = no smoothing.")
        w_audio = st.slider("Voice weight in combined score", 0.0, 1.0, 0.7, disabled=not has_text,
                            help="combined = w x smoothed voice + (1 - w) x scam-language score. "
                                 "Only used when a transcript is provided.")
        if not has_text:
            st.caption("Voice weight is inactive until a transcript is entered.")
    step_heading("", "Result summary")
    verdict_slot = st.empty()                      # filled below, after the analysis (display only)

# ----------------------------------------------------------------------------- results
st.markdown('<div class="vd-section"></div>', unsafe_allow_html=True)
step_heading("04", "Score breakdown", "Three separate measurements, then the window-by-window detail.")

if f is None:
    verdict_slot.markdown('<div class="vd-placeholder"><b>No result yet.</b> Choose audio in step 01; the heuristic '
                          'band and the score it is based on appear here.</div>', unsafe_allow_html=True)
if f is None and mic_mode:
    file_slot.empty()
    st.markdown('<div class="vd-empty"><b>No recording yet.</b><br>Press the microphone button in step 01, speak for '
                '4-8 seconds, then press stop. The take is analysed as soon as recording stops.</div>',
                unsafe_allow_html=True)
elif f is None:
    file_slot.empty()
    st.markdown('<div class="vd-empty"><b>No recording yet.</b><br>Upload a WAV or FLAC file in step 01, or switch to '
                '<b>Use a demo clip</b>. The combined-call demo is the clearest: genuine speech for the first half, '
                'synthetic speech after.</div>', unsafe_allow_html=True)
else:
    analysis = None
    with st.spinner("Decoding audio and scoring windows..."):
        try:
            # --- analysis: identical calls and formulas to Phase 1 (Phase 5: inspect_audio adds input checks) ---
            check = inspect_audio(f)
            x, orig_sr, audio_warnings = check.audio, check.orig_sr, check.warnings
            windows, times, padded = make_windows(x, win, step)
            raw = list(score_windows(model, windows))
            smooth = ema(raw, alpha)
            t_score = scam_score(transcript) if has_text else None
            combined = [w_audio * a + (1 - w_audio) * t_score for a in smooth] if t_score is not None else smooth
            last_voice, last = smooth[-1], combined[-1]
            band = heuristic_band(last)
            analysis = True
        except AudioError as e:
            file_slot.markdown(file_panel(f), unsafe_allow_html=True)
            st.markdown('<div class="vd-empty">No scores: the input was rejected. The reason is shown under '
                        '<b>Result summary</b>.</div>', unsafe_allow_html=True)
            with verdict_slot.container():        # shown under "Result summary", visible on the first screen
                if mic_mode:
                    error_panel("This recording cannot be analysed, so no score is shown", str(e),
                                "Record again: allow microphone access, speak close to the microphone for 4-8 "
                                "seconds, then press stop.")
                else:
                    error_panel("This file cannot be analysed, so no score is shown", str(e),
                                "Check that the file plays in a media player and contains speech, then upload it "
                                "again.")

    if analysis:
        file_slot.markdown(file_panel(f, x, orig_sr), unsafe_allow_html=True)
        try:
            if not mic_mode:                        # st.audio_input already shows its own playback control
                f.seek(0); player_slot.audio(f.read(), format="audio/flac" if f.name.lower().endswith(".flac") else "audio/wav")
                f.seek(0)
        except Exception:
            pass                                    # playback is a convenience only
        dur = len(x) / SR
        lo, hi = HEURISTIC_BANDS
        source = "combined" if t_score is not None else "smoothed voice"
        advice = {"HIGH": "The model flags this audio strongly. Do not share OTPs or send money; hang up and "
                          "call back on a number you trust.",
                  "MEDIUM": "Some warning signs. Be cautious and verify the caller.",
                  "LOW": "No strong signal from the model, but still verify unusual requests."}[band]
        summary = verdict_slot.container()
        summary.markdown(
            f'<div class="vd-verdict" style="--c:{BAND_COLOR[band]}"><div class="k">Heuristic band &middot; '
            f'latest {source} score {last:.2f}</div><div class="h">{band}</div><p>{esc(advice)}</p>'
            f'<p class="m">Bands at {lo} / {hi} were hand-picked and are not validated. '
            'This is not a fraud probability.</p></div>', unsafe_allow_html=True)

        for wmsg in audio_warnings:
            summary.markdown(f'<div class="vd-note"><b>Possibly damaged file.</b> {esc(wmsg)} Treat the result with '
                        'extra caution.</div>', unsafe_allow_html=True)
        if padded:
            summary.markdown(f'<div class="vd-note"><b>Short recording.</b> This clip is {dur:.2f} s, shorter than the '
                        f'{win} s window, so it was wrap-padded (repeated from the start) to {win} s, the same '
                        'padding the training script used. Only one window was scored; treat the result with '
                        'extra caution.</div>', unsafe_allow_html=True)

        side = "leans synthetic" if last_voice > DECISION_BOUNDARY else "leans genuine"
        st.markdown('<div class="vd-distinct"><b>Voice and wording are measured separately.</b> The voice score asks '
                    'whether the speech sounds synthetic; the transcript score asks whether the wording sounds like a '
                    'scam. A high voice score does not prove the caller is a scammer, and a genuine voice can still '
                    'run a scam.</div>', unsafe_allow_html=True)
        cols = st.columns(3, gap="medium")
        cols[0].markdown(card(
            "Voice: synthetic-speech score", last_voice, ACCENT,
            [(f"Latest smoothed score. Model <b>{side}</b> "
              f"({'above' if last_voice > DECISION_BOUNDARY else 'at or below'} the {DECISION_BOUNDARY} boundary).", False),
             (f"{len(raw)} window(s) of {win} s &middot; latest raw window {raw[-1]:.2f}", True),
             (f"Marker = model decision boundary {DECISION_BOUNDARY} (the only threshold with measured performance).", True)],
            measures="Does the speech sound synthetic?", help_text=H_VOICE),
            unsafe_allow_html=True)
        if t_score is not None:
            cols[1].markdown(card("Transcript: scam-language score", t_score, AMBER,
                                  [("From the transcript text only; says nothing about the voice.", False),
                                   ("Toy text model, 24 training sentences, uncalibrated; not a validated fraud "
                                    "classifier. No validated threshold; the 0.5 marker is for reference only.", True)],
                                  measures="Does the wording sound like a scam?", help_text=H_TEXT),
                             unsafe_allow_html=True)
            cols[2].markdown(card("Combined: heuristic score", last, INK,
                                  [(f"{w_audio:.1f} &times; voice {last_voice:.2f} + {1 - w_audio:.1f} &times; "
                                    f"text {t_score:.2f}", False),
                                   (f"Hand-weighted average of two uncalibrated scores. Feeds the hand-picked "
                                    f"{HEURISTIC_BANDS[0]} / {HEURISTIC_BANDS[1]} bands; not a fraud probability.", True)],
                                  measures="Both scores blended by a fixed weight", help_text=H_COMB),
                             unsafe_allow_html=True)
        else:
            cols[1].markdown(card("Transcript: scam-language score", off_text="No transcript",
                                  lines=[("Add the caller's words in step 02 to get this score.", True)],
                                  measures="Does the wording sound like a scam?", help_text=H_TEXT),
                             unsafe_allow_html=True)
            cols[2].markdown(card("Combined: heuristic score", off_text="Needs a transcript",
                                  lines=[("Without a transcript, the heuristic band uses the smoothed voice "
                                          "score alone.", True)],
                                  measures="Both scores blended by a fixed weight", help_text=H_COMB),
                             unsafe_allow_html=True)

        st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
        step_heading("", "Window by window",
                     f"One block per {win} s window, in time order (label = window end, in seconds). Orange = raw "
                     f"voice score above the {DECISION_BOUNDARY} boundary; grey-green = at or below it.")
        above = [v > DECISION_BOUNDARY for v in raw]
        st.markdown('<div class="vd-strip">' + "".join(
            f'<span class="{"hi" if a else "lo"}" title="window ending {t:.1f} s: raw score {v:.2f}">{t:.0f}</span>'
            for t, v, a in zip(times, raw, above)) + "</div>", unsafe_allow_html=True)
        first = next((t for t, a in zip(times, above) if a), None)
        st.markdown(f'<div class="vd-strip-legend"><b>{sum(above)} of {len(raw)}</b> window(s) above the boundary'
                    + (f'; first one ends at {first:.1f} s.' if first is not None else '.')
                    + f'<span style="white-space:nowrap"><i style="background:{ACCENT}"></i>above {DECISION_BOUNDARY}</span>'
                    + '<span style="white-space:nowrap"><i style="background:#A9B8AE"></i>at or below</span></div>',
                    unsafe_allow_html=True)
        step_heading("", "Voice score over time",
                     "Each point is one window, plotted at the time its window ends. The smoothed line is what the "
                     "cards and band use. Higher means more like the synthetic speech the model was trained on.")
        if len(raw) > 1:
            data = {"time": times, SERIES[0]: raw, SERIES[1]: smooth}
            if t_score is not None:
                data[SERIES[2]] = combined
            long = pd.DataFrame(data).melt("time", var_name="series", value_name="score")
            show_h = st.toggle("Show heuristic band lines (0.4 / 0.7, not validated)", value=False)
            st.altair_chart(timeline(long, show_h), width="stretch")
        else:
            st.caption("Only one window was scored, so there is no timeline to plot.")

        ev = chunk_evidence(str(CHUNK_RESULTS))
        with st.expander("Window-by-window scores"):
            table = pd.DataFrame({"Window end (s)": times, "Raw voice score": raw, "Smoothed voice score": smooth})
            if t_score is not None:
                table["Combined score"] = combined
            table[f"Raw vs {DECISION_BOUNDARY} boundary"] = ["above" if v > DECISION_BOUNDARY else "at/below" for v in raw]
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.2f")
                                        for c in ("Raw voice score", "Smoothed voice score", "Combined score")} |
                         {"Window end (s)": st.column_config.NumberColumn(format="%.1f")})
        with st.expander("Technical details"):
            ref = ""
            if ev is not None and win in ev.index:
                r = ev.loc[win]
                ref = (f"\n* **Measured reference for {win} s windows:** {r['Accuracy']:.1f} % accuracy, "
                       f"{r['EER']:.1f} % EER on the 1000-clip ASVspoof 2019 LA eval subset (`results_chunk.csv`).")
            st.markdown(f"""
* **Input:** {dur:.2f} s analysed, original sample rate {orig_sr} Hz, resampled to {SR} Hz mono.
* **Windows:** {len(raw)} x {win} s, step {step} s{', wrap-padded' if padded else ''}; a final partial step is not scored. Default settings (3 s, 1 s, alpha 0.4, weight 0.7) match the original prototype.
* **Features:** 64-band log-mel (n_fft 512, hop 160), normalised per window; CNN with 3 conv blocks (23,585 parameters).
* **Score:** sigmoid of the CNN output; above the {DECISION_BOUNDARY} decision boundary the model leans synthetic.
* **Smoothing:** exponential moving average, alpha {alpha:.2f}.{ref}
* **Model file:** `{MODEL_PATH.parent.name}/{MODEL_PATH.name}`
""")

st.markdown("""
<div class="vd-limits">
  <div><h4>Benchmark-based detection</h4><p>Trained and tested only on ASVspoof&nbsp;2019&nbsp;LA, clean studio-style
  speech. On a 1,000-clip evaluation subset: <b>86.7&nbsp;% accuracy, 8.4&nbsp;% EER</b>. Phone-line audio, noise and
  modern voice-cloning tools were not tested.</p><p><b>Microphone takes have no measured accuracy</b>; the figures
  above come from the benchmark subset only.</p></div>
  <div><h4>Unseen attacks and shortcuts</h4><p>Some attack types are mostly missed (A15 and A17: <b>roughly 70&ndash;90&nbsp;%</b>
  missed in Phases 3&ndash;4). The model also relies on <b>leading silence</b>: genuine clips in the dataset start
  with more silence, and removing it raised the EER to 23.6&nbsp;% in a diagnostic test.</p></div>
  <div><h4>Voice spoofing is not scam intent</h4><p>The voice score asks <b>whether speech sounds synthetic</b>, not whether
  the caller is a fraudster. A real person can run a scam, and synthetic speech can be harmless. The transcript score
  is a separate toy model (24 training sentences), not a validated fraud classifier.</p></div>
</div>
""", unsafe_allow_html=True)

with st.expander("How to read these numbers"):
    lo, hi = HEURISTIC_BANDS
    st.markdown(f"""
* **Voice score** is the CNN's sigmoid output (0-1). It is an **uncalibrated model score, not a probability**
  that the voice is synthetic or that the call is a fraud. The model was never calibrated.
* The only threshold with measured performance is the model's **decision boundary at {DECISION_BOUNDARY}**.
  On a 1000-clip ASVspoof 2019 LA eval subset (4 s windows) it gave 86.7 % accuracy and 8.4 % EER
  (`results_in_domain.csv`). That is studio-quality benchmark speech; phone audio and modern voice clones were **not** tested.
* The **LOW / MEDIUM / HIGH bands ({lo} / {hi})** come from the original prototype. They were chosen by hand and
  **have not been validated**.
* The **scam-language score** comes from a toy text model trained on 24 self-written sentences; it is also uncalibrated.
* The **combined score** is a hand-weighted average of two uncalibrated scores. It is not a fraud probability.
* **Microphone takes** go through the same checks and scoring, but no accuracy has been measured for them. Laptop
  microphone audio differs from the benchmark recordings, so treat those scores as a demonstration only.
""")
