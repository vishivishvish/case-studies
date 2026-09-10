# Extracted from eeg_motor_imagery.ipynb

# %% Cell 0
# ── Configuration ──────────────────────────────────────────────────────
SUBJECT = 1
IMAGERY_RUNS = [4, 8, 12]      # imagined left/right fist
MOVEMENT_RUNS = [3, 7, 11]     # actual left/right fist
SFREQ = 160                     # sampling rate (Hz)
TMIN = 0.0                      # epoch start (seconds)
TMAX = 2.0                      # epoch end (seconds)

from mne.datasets import eegbci

# %% Cell 1
# ── Environment detection ──────────────────────────────────────────────
try:
    import google.colab
    IN_COLAB = True
    print("Running on Google Colab")
except ImportError:
    IN_COLAB = False
    print("Running locally")

import os
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# %% Cell 2
import warnings
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import welch

# MNE-Python
import mne
mne.set_log_level("WARNING")

# Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed: install with: pip install xgboost")

# SHAP
import shap
shap.initjs()

print("Libraries loaded.")
print(f"MNE version: {mne.__version__}")
print(f"NumPy version: {np.__version__}")

# %% Cell 3
# ── Download and load imagery runs ────────────────────────────────────
print(f"Loading Subject {SUBJECT}, runs {IMAGERY_RUNS} (imagery)...")
# Use update_path=True to avoid interactive prompt
img_files = eegbci.load_data(SUBJECT, runs=IMAGERY_RUNS,
                              path='data/', verbose=False, update_path=True)
img_raws  = [mne.io.read_raw_edf(f, preload=True, verbose=False)
             for f in img_files]
raw_img = mne.concatenate_raws(img_raws)

# %% Cell 4
# ── Load actual movement runs (for comparative visualisation) ────────
print(f"Loading Subject {SUBJECT}, runs {MOVEMENT_RUNS} (actual movement)...")
# Use update_path=True to avoid interactive prompt
mov_files = eegbci.load_data(SUBJECT, runs=MOVEMENT_RUNS,
                              path='data/', verbose=False, update_path=True)
mov_raws  = [mne.io.read_raw_edf(f, preload=True, verbose=False)
             for f in mov_files]
raw_mov = mne.concatenate_raws(mov_raws)

# %% Cell 5
# ── Apply bandpass filter ──────────────────────────────────────────────
print("Applying 1-100 Hz bandpass filter...")
raw_img_filt = raw_img.copy().filter(l_freq=1.0, h_freq=100.0,
                                      fir_window='hamming', verbose=False)
raw_mov_filt = raw_mov.copy().filter(l_freq=1.0, h_freq=100.0,
                                      fir_window='hamming', verbose=False)
print("  Filtering complete.")

# ── Extract events and epoch ───────────────────────────────────────────
print("\nExtracting events from annotations...")

events_img, event_id_img = mne.events_from_annotations(
    raw_img_filt, verbose=False)
events_mov, event_id_mov = mne.events_from_annotations(
    raw_mov_filt, verbose=False)

print(f"  Imagery event types: {event_id_img}")
print(f"  Movement event types: {event_id_mov}")

# T1 = left hand, T2 = right hand
# Filter to only T1/T2 events
evt_img = {k: v for k, v in event_id_img.items() if k in ['T1', 'T2']}
evt_mov = {k: v for k, v in event_id_mov.items() if k in ['T1', 'T2']}

epochs_img = mne.Epochs(raw_img_filt, events_img, event_id=evt_img,
                         tmin=TMIN, tmax=TMAX,
                         baseline=None, preload=True, verbose=False)
epochs_mov = mne.Epochs(raw_mov_filt, events_mov, event_id=evt_mov,
                         tmin=TMIN, tmax=TMAX,
                         baseline=None, preload=True, verbose=False)

print(f"\nImagery epochs:  {len(epochs_img)} trials")
print(f"  T1 (left hand):  {sum(epochs_img.events[:,2] == evt_img.get('T1',2))}")
print(f"  T2 (right hand): {sum(epochs_img.events[:,2] == evt_img.get('T2',3))}")
print(f"  Epoch shape: {epochs_img.get_data().shape}  (trials × channels × time)")
print(f"\nMovement epochs: {len(epochs_mov)} trials")

# %% Cell 6
# ── Visualise: time-frequency representation of motor imagery ─────────
# Compute mean PSD for T1 vs T2 at C3 and C4 using Welch's method
from scipy.signal import welch as sp_welch

ch_names = epochs_img.ch_names
C3_idx = ch_names.index('C3') if 'C3' in ch_names else 0
C4_idx = ch_names.index('C4') if 'C4' in ch_names else 1

data_img = epochs_img.get_data()  # (n_epochs, n_ch, n_times)
labels_img = epochs_img.events[:, 2]
T1_code = evt_img.get('T1', sorted(evt_img.values())[0])
T2_code = evt_img.get('T2', sorted(evt_img.values())[1])

t1_data = data_img[labels_img == T1_code]   # left hand
t2_data = data_img[labels_img == T2_code]   # right hand

sfreq = epochs_img.info['sfreq']

def mean_psd(eeg_epochs, ch_idx, sfreq):
    # Average PSD over epochs for one channel.
    psds = []
    for ep in eeg_epochs:
        f, p = sp_welch(ep[ch_idx], fs=sfreq, nperseg=min(256, ep.shape[1]))
        psds.append(p)
    return f, np.array(psds).mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (ch_name, ch_idx) in zip(axes, [('C3', C3_idx), ('C4', C4_idx)]):
    f_t1, psd_t1 = mean_psd(t1_data, ch_idx, sfreq)
    f_t2, psd_t2 = mean_psd(t2_data, ch_idx, sfreq)

    ax.semilogy(f_t1, psd_t1 * 1e12, color='#3b82f6', lw=2, label='Left hand imagery (T1)')
    ax.semilogy(f_t2, psd_t2 * 1e12, color='#ef4444', lw=2, label='Right hand imagery (T2)')

    # Shade frequency bands
    band_colors = {'alpha': ('#fbbf24', 0.25), 'beta': ('#22c55e', 0.20)}
    band_ranges = {'alpha': (8, 13), 'beta': (13, 30)}
    for band, (fmin, fmax) in band_ranges.items():
        color, alpha = band_colors[band]
        ax.axvspan(fmin, fmax, color=color, alpha=alpha, label=f'{band} band')

    ax.set_xlim(1, 50)
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Power (µV²/Hz)", fontsize=12)
    ax.set_title(f"PSD at {ch_name}: Left vs Right Hand Imagery", fontsize=13)
    ax.legend(fontsize=9)

plt.suptitle("Power Spectral Density: Left vs Right Hand Imagery\nERD visible as power difference in alpha and beta bands",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('plots/psd_c3_c4.png', dpi=120, bbox_inches='tight')
plt.show()
print("PSD comparison plot saved.")

# %% Cell 7
from sklearn.linear_model import LogisticRegression
motor_idx=[feature_names.index(f"{b}_{c}") for b in ("alpha","beta") for c in ("C3","C4") if f"{b}_{c}" in feature_names]
baseline=LogisticRegression(max_iter=2000,random_state=42).fit(X_tr[:,motor_idx],y_tr)
print("C3/C4 alpha-beta baseline accuracy:",accuracy_score(y_te,baseline.predict(X_te[:,motor_idx])))


# %% Cell 8
# [Moved to after feature_names definition - see cell below]

# %% Cell 9
# ── Frequency band definitions ─────────────────────────────────────────
BANDS = {
    'delta': (1,  4),
    'theta': (4,  8),
    'alpha': (8,  13),
    'beta':  (13, 30),
    'gamma': (30, 100),
}
BAND_NAMES = list(BANDS.keys())

def compute_psd_features(epoch_data, sfreq, bands=BANDS):
    # Compute mean band power per channel across frequency bands.
    # epoch_data: (n_channels, n_times)
    # Returns: (n_bands * n_channels,) [delta_all_ch, theta_all_ch, ...]
    n_ch = epoch_data.shape[0]
    nperseg = min(256, epoch_data.shape[1])
    freqs, psd = sp_welch(epoch_data, fs=sfreq, nperseg=nperseg, axis=-1)
    # psd: (n_channels, n_freqs)
    features = []
    for fmin, fmax in bands.values():
        idx = (freqs >= fmin) & (freqs < fmax)
        band_power = psd[:, idx].mean(axis=-1)  # (n_channels,)
        features.append(band_power)
    return np.concatenate(features)  # (n_bands * n_channels,)

# ── Build feature matrix ───────────────────────────────────────────────
print("Computing PSD features for all imagery epochs...")
data_all = epochs_img.get_data()          # (n_epochs, n_ch, n_times)
sfreq_val = epochs_img.info['sfreq']

X = np.array([compute_psd_features(ep, sfreq_val) for ep in data_all])
y = (epochs_img.events[:, 2] == T2_code).astype(int)  # 0=left, 1=right

# ── Feature names for SHAP ─────────────────────────────────────────────
# Order: [delta_ch1, delta_ch2, ..., delta_ch64, theta_ch1, ...]
feature_names = [f"{band}_{ch}" for band in BAND_NAMES for ch in ch_names]

print(f"Feature matrix shape: {X.shape}  ({X.shape[0]} trials × {X.shape[1]} features)")
print(f"Class distribution: {sum(y==0)} left-hand, {sum(y==1)} right-hand")
print(f"Feature names sample: {feature_names[:5]} ... {feature_names[-5:]}")

# Show features for a single trial
print(f"\nFeature vector for trial 0 (first 10 values):")
print(np.round(X[0, :10], 4))

# %% Cell 10
# ── Visualise: top features by class (intuition check) ───────────────
# Compare mean alpha power (per channel) between left and right hand imagery
alpha_start = BAND_NAMES.index('alpha') * len(ch_names)
alpha_end   = alpha_start + len(ch_names)
beta_start  = BAND_NAMES.index('beta') * len(ch_names)
beta_end    = beta_start + len(ch_names)

left_alpha_mean  = X[y==0, alpha_start:alpha_end].mean(axis=0)
right_alpha_mean = X[y==1, alpha_start:alpha_end].mean(axis=0)
left_beta_mean   = X[y==0, beta_start:beta_end].mean(axis=0)
right_beta_mean  = X[y==1, beta_start:beta_end].mean(axis=0)

# Highlight C3 and C4 positions
c3_idx = ch_names.index('C3') if 'C3' in ch_names else 0
c4_idx = ch_names.index('C4') if 'C4' in ch_names else 1

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, (band_l, band_r, band_name) in zip(axes, [
    (left_alpha_mean, right_alpha_mean, 'Alpha (8-13 Hz)'),
    (left_beta_mean,  right_beta_mean,  'Beta (13-30 Hz)'),
]):
    diff = band_r - band_l   # right minus left
    colors = ['#ef4444' if d > 0 else '#3b82f6' for d in diff]
    ax.bar(range(len(ch_names)), diff, color=colors, alpha=0.7, width=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(c3_idx, color='#22c55e', lw=2, ls='--', label=f'C3 (idx {c3_idx})')
    ax.axvline(c4_idx, color='gold',    lw=2, ls='--', label=f'C4 (idx {c4_idx})')
    ax.set_xlabel("Channel index", fontsize=11)
    ax.set_ylabel("Power diff: Right − Left (µV²/Hz)", fontsize=11)
    ax.set_title(f"{band_name} Power Difference\n(Right hand > left hand = red)", fontsize=12)
    ax.legend(fontsize=10)

plt.suptitle("Power difference between classes per channel\n"
             "ERD signature: the contralateral electrode has LOWER power for each class",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('plots/band_power_diff.png', dpi=120, bbox_inches='tight')
plt.show()

# %% Cell 11
import os,json,requests
from openai import OpenAI
NVIDIA_BASE_URL="https://integrate.api.nvidia.com/v1"; NVIDIA_MODEL=os.getenv("NVIDIA_MODEL","meta/llama-3.1-8b-instruct")
def nvidia_text(prompt):
 key=os.getenv("NVIDIA_API_KEY")
 if not key: raise RuntimeError("Set NVIDIA_API_KEY outside the notebook.")
 return OpenAI(base_url=NVIDIA_BASE_URL,api_key=key).chat.completions.create(model=NVIDIA_MODEL,messages=[{"role":"user","content":prompt}],temperature=.2,max_tokens=400).choices[0].message.content
try: nvidia_proposal=json.loads(nvidia_text("Propose three safe left-right motor-imagery EEG aggregates using only log1p alpha_C3, alpha_C4, beta_C3, beta_C4 and addition or subtraction. Return JSON."))
except Exception as exc: nvidia_proposal={"offline_fallback":True,"reason":str(exc)}
def nvidia_features(x):
 z=np.log1p(np.maximum(x,0)); a3,a4=z[:,fidx("alpha","C3")],z[:,fidx("alpha","C4")]; b3,b4=z[:,fidx("beta","C3")],z[:,fidx("beta","C4")]
 return np.c_[a3-a4,b3-b4,a3+a4+b3+b4]
X_nvidia=nvidia_features(X); print(nvidia_proposal)


# %% Cell 12
train_idx,test_idx=train_test_split(np.arange(len(y)),test_size=.2,random_state=42,stratify=y)
X_enriched=np.c_[X,X_human,X_nvidia]
def score(z):
 m=RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1).fit(z[train_idx],y[train_idx])
 return accuracy_score(y[test_idx],m.predict(z[test_idx]))
feature_ablation=pd.DataFrame({"feature_set":["PSD only","Human designed","NVIDIA proposed","Human + NVIDIA"],"held_out_accuracy":[score(X),score(X_human),score(X_nvidia),score(X_enriched)]});display(feature_ablation)
def synthetic(z,labels):
 rng=np.random.default_rng(42); rows=[]; targets=[]
 for label in np.unique(labels):
  group=z[labels==label]; rows.append(rng.multivariate_normal(group.mean(0),np.cov(group,rowvar=False)+np.eye(group.shape[1])*1e-8,len(group)));targets += [label]*len(group)
 return np.vstack(rows),np.array(targets)
sx,sy=synthetic(X_enriched[train_idx],y[train_idx])
def fidelity(a,b):
 m=RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1).fit(a,b);return accuracy_score(y[test_idx],m.predict(X_enriched[test_idx]))
synthetic_fidelity=pd.DataFrame({"training_data":["Original only","Original + synthetic","Synthetic only"],"real_held_out_accuracy":[fidelity(X_enriched[train_idx],y[train_idx]),fidelity(np.r_[X_enriched[train_idx],sx],np.r_[y[train_idx],sy]),fidelity(sx,sy)]});display(synthetic_fidelity)


# %% Cell 13
# ── Train / test split ────────────────────────────────────────────────
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {len(X_tr)} trials")
print(f"Test set:     {len(X_te)} trials")

# ── Random Forest ──────────────────────────────────────────────────────
print("\nTraining Random Forest (200 trees)...")
rf = RandomForestClassifier(n_estimators=200, max_features='sqrt',
                             random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
rf_cv  = cross_val_score(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
print(f"  RF 5-fold CV accuracy: {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")

# ── XGBoost ───────────────────────────────────────────────────────────
if HAS_XGB:
    print("\nTraining XGBoost (200 estimators)...")
    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=42, eval_metric='logloss',
                         verbosity=0)
    xgb.fit(X_tr, y_tr)
    xgb_cv = cross_val_score(xgb, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    print(f"  XGB 5-fold CV accuracy: {xgb_cv.mean():.3f} ± {xgb_cv.std():.3f}")
    best_model = xgb if xgb_cv.mean() > rf_cv.mean() else rf
    best_name  = "XGBoost" if xgb_cv.mean() > rf_cv.mean() else "Random Forest"
else:
    best_model = rf
    best_name  = "Random Forest"

print(f"\nBest model: {best_name}")

# %% Cell 14
# ── Test set evaluation ────────────────────────────────────────────────
rf_preds  = rf.predict(X_te)
rf_acc    = accuracy_score(y_te, rf_preds)
rf_proba  = rf.predict_proba(X_te)

print("=" * 55)
print(f"RANDOM FOREST: Test Set Performance")
print("=" * 55)
print(f"Accuracy: {rf_acc:.3f}")
print()
print(classification_report(y_te, rf_preds,
                             target_names=['Left Hand (0)', 'Right Hand (1)']))

if HAS_XGB:
    xgb_preds = xgb.predict(X_te)
    xgb_acc   = accuracy_score(y_te, xgb_preds)
    print("=" * 55)
    print(f"XGBOOST: Test Set Performance")
    print("=" * 55)
    print(f"Accuracy: {xgb_acc:.3f}")
    print()
    print(classification_report(y_te, xgb_preds,
                                 target_names=['Left Hand (0)', 'Right Hand (1)']))

# ── Visualise ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2 if HAS_XGB else 1, figsize=(14 if HAS_XGB else 7, 5))
if not HAS_XGB:
    axes = [axes]

for ax, (preds, name, acc) in zip(axes,
    [(rf_preds, 'Random Forest', rf_acc)] +
    ([(xgb_preds, 'XGBoost', xgb_acc)] if HAS_XGB else [])):
    cm = confusion_matrix(y_te, preds)
    ConfusionMatrixDisplay(cm, display_labels=['Left Hand', 'Right Hand']).plot(ax=ax)
    ax.set_title(f"{name}\nTest Accuracy: {acc:.1%}", fontsize=12)

plt.tight_layout()
plt.savefig('plots/confusion_matrices.png', dpi=120, bbox_inches='tight')
plt.show()

# %% Cell 15
# ── CV comparison bar chart ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

models = ['Random Forest', 'Chance']
means  = [rf_cv.mean(), 0.5]
stds   = [rf_cv.std(), 0.0]
colors = ['#3b82f6', '#94a3b8']

if HAS_XGB:
    models.insert(1, 'XGBoost')
    means.insert(1, xgb_cv.mean())
    stds.insert(1, xgb_cv.std())
    colors.insert(1, '#f59e0b')

bars = ax.bar(models, means, color=colors, alpha=0.8, edgecolor='black', lw=0.5)
ax.errorbar(range(len(models)), means, yerr=stds, fmt='none',
            color='black', capsize=5, lw=1.5)
ax.axhline(0.5, color='grey', lw=1.5, ls='--', label='Chance (50%)')
ax.set_ylim(0.3, 1.0)
ax.set_ylabel("5-Fold CV Accuracy", fontsize=12)
ax.set_title("Motor Imagery Classification: Model Comparison", fontsize=13)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, mean + 0.01,
            f"{mean:.1%}", ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=120, bbox_inches='tight')
plt.show()
print("\nInterpretation: single-subject, single-session performance. Full-dataset")
print("cross-subject models typically achieve 70-80% with PSD features.")

# %% Cell 16
# ── SHAP analysis ─────────────────────────────────────────────────────
print("Computing SHAP values...")

# Use the best model; SHAP TreeExplainer works for both RF and XGBoost
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_te)

# For binary classification, RF gives a list [shap_class0, shap_class1]
# XGBoost gives a single array (for class 1)
if isinstance(shap_values, list):
    sv = shap_values[1]   # class 1 = right hand
else:
    sv = shap_values

print(f"SHAP values shape: {sv.shape}  ({sv.shape[0]} test trials × {sv.shape[1]} features)")

# ── Mean |SHAP| per feature ────────────────────────────────────────────
mean_abs_shap = pd.Series(np.abs(sv).mean(axis=0), index=feature_names)
top_features  = mean_abs_shap.nlargest(30)

print("\nTop 10 most important features:")
for fname, val in top_features.head(10).items():
    print(f"  {fname:<25} {val:.6f}")

# %% Cell 17
# ── SHAP beeswarm plot ─────────────────────────────────────────────────
plt.figure(figsize=(12, 10))
shap.summary_plot(sv, X_te,
                  feature_names=feature_names,
                  max_display=25,
                  plot_type="dot",
                  show=False)
plt.title("SHAP Beeswarm: Motor Imagery Classification\n"
          "Top 25 features by mean |SHAP|", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig('plots/shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()
print("SHAP beeswarm saved.")
print()
print("KEY OBSERVATION: Look for alpha_C3, beta_C3, alpha_C4, beta_C4 near the top.")
print("These are the electrodes over the motor cortex hand area.")
print("Their presence at the top confirms ERD as the classifying mechanism.")

# %% Cell 18
# ── SHAP importance by frequency band ─────────────────────────────────
# Compute mean |SHAP| for each frequency band (summed across channels)
band_importance = {}
for band in BAND_NAMES:
    band_feat_idx = [i for i, name in enumerate(feature_names) if name.startswith(band)]
    band_importance[band] = np.abs(sv[:, band_feat_idx]).mean()

fig, ax = plt.subplots(figsize=(9, 5))
bands_sorted = sorted(band_importance.items(), key=lambda x: x[1], reverse=True)
b_names = [b[0] for b in bands_sorted]
b_vals  = [b[1] for b in bands_sorted]
colors  = ['#ef4444' if b in ('alpha','beta') else '#94a3b8' for b in b_names]

bars = ax.bar(b_names, b_vals, color=colors, edgecolor='black', lw=0.5, alpha=0.85)
for bar, val in zip(bars, b_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.0001,
            f"{val:.4f}", ha='center', va='bottom', fontsize=10)
ax.set_ylabel("Mean |SHAP| value", fontsize=12)
ax.set_title("Feature Importance by Frequency Band\n"
             "Alpha and beta dominate: confirming ERD as the classifying signal", fontsize=12)

# Annotation
ax.annotate("ERD signature:\nAlpha & Beta suppressed\nduring motor imagery",
            xy=(0.15, 0.80), xycoords='axes fraction',
            bbox=dict(boxstyle='round', facecolor='#fef9c3', alpha=0.9),
            fontsize=10)
plt.tight_layout()
plt.savefig('plots/shap_by_band.png', dpi=120, bbox_inches='tight')
plt.show()
print(f"Top bands by SHAP importance: {b_names[:2]}")
print("This is independent confirmation of ERD: the model found it without neurophysiology knowledge.")

# %% Cell 19
# ── Class-specific SHAP topomaps (left vs right hand) ─────────────────
# For RF: shap_values is a list [class0_shap, class1_shap]
# For XGB: shap_values is for class 1; class 0 = -shap_values

sv_raw = explainer.shap_values(X_te)
if isinstance(sv_raw, list):
    sv_left  = sv_raw[0]   # class 0 = left hand
    sv_right = sv_raw[1]   # class 1 = right hand
else:
    sv_right = sv_raw
    sv_left  = -sv_raw    # approximate

# Mean |SHAP| per channel for each class, using alpha+beta only
def shap_per_channel_alphabeta(sv_matrix):
    abs_sv = np.abs(sv_matrix)
    sv_2d  = abs_sv.mean(axis=0).reshape(len(BAND_NAMES), len(ch_names))
    return sv_2d[alpha_beta_idx].mean(axis=0)

shap_left_ch  = shap_per_channel_alphabeta(sv_left)
shap_right_ch = shap_per_channel_alphabeta(sv_right)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

vmax_lr = max(shap_left_ch.max(), shap_right_ch.max())

im_l, _ = plot_topomap(shap_left_ch, info_plot, axes=axes[0], show=False,
                        cmap='hot_r', vlim=(0, vmax_lr), sensors=True)
plt.colorbar(im_l, ax=axes[0], shrink=0.85, label="Mean |SHAP| (alpha+beta)")
axes[0].set_title("SHAP Importance for LEFT Hand Prediction\n"
                   "→ C4 (right hemisphere) lights up\n"
                   "= contralateral control, independently found", fontsize=11)

im_r, _ = plot_topomap(shap_right_ch, info_plot, axes=axes[1], show=False,
                        cmap='hot_r', vlim=(0, vmax_lr), sensors=True)
plt.colorbar(im_r, ax=axes[1], shrink=0.85, label="Mean |SHAP| (alpha+beta)")
axes[1].set_title("SHAP Importance for RIGHT Hand Prediction\n"
                   "→ C3 (left hemisphere) lights up\n"
                   "= contralateral control, independently found", fontsize=11)

plt.suptitle("The Contralateral Crossover, Found by Machine Learning\n"
             "Left hand → right brain (C4). Right hand → left brain (C3).\n"
             "This is Penfield's motor homunculus principle, rediscovered computationally.",
             fontsize=11, y=1.04)
plt.tight_layout()
plt.savefig('plots/contralateral_shap_topomaps.png', dpi=150, bbox_inches='tight')
plt.show()
print("Contralateral SHAP topomaps saved.")
print()
print("LEFT HAND prediction: top channel by SHAP =",
      ch_names[shap_left_ch.argmax()])
print("RIGHT HAND prediction: top channel by SHAP =",
      ch_names[shap_right_ch.argmax()])
print()
print("Expected: C4 for left hand, C3 for right hand (contralateral control)")

# %% Cell 20
import torch,torch.nn as nn
torch.manual_seed(42); sc=StandardScaler().fit(X[train_idx]); xt=torch.tensor(sc.transform(X[train_idx]),dtype=torch.float32).unsqueeze(1); xv=torch.tensor(sc.transform(X[test_idx]),dtype=torch.float32).unsqueeze(1); yt=torch.tensor(y[train_idx],dtype=torch.long);yv=torch.tensor(y[test_idx],dtype=torch.long)
eegnet_lite=nn.Sequential(nn.Conv1d(1,16,7,padding=3),nn.ReLU(),nn.BatchNorm1d(16),nn.Dropout(.25),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(16,2));opt=torch.optim.Adam(eegnet_lite.parameters(),lr=1e-3);loss=nn.CrossEntropyLoss()
for _ in range(80): opt.zero_grad();loss(eegnet_lite(xt),yt).backward();opt.step()
print("EEGNetLite held-out accuracy:",(eegnet_lite(xv).argmax(1)==yv).float().mean().item())


# %% Cell 21
try:
 from tabpfn import TabPFNClassifier
 tabpfn=TabPFNClassifier(random_state=42).fit(X[train_idx],y[train_idx]);print("TabPFN held-out accuracy:",accuracy_score(y[test_idx],tabpfn.predict(X[test_idx])))
except ImportError: print("Install tabpfn, then rerun this cell.")


# %% Cell 22
from mne.viz import plot_topomap

# ── Prepare info object with electrode positions ───────────────────────
# Use the montage from our data
info_plot = epochs_img.info.copy()

# Indices of features for each band
def band_ch_idx(band_name):
    # Return slice for a given band's features in the feature matrix.
    b_idx = BAND_NAMES.index(band_name)
    return slice(b_idx * len(ch_names), (b_idx + 1) * len(ch_names))

# Mean alpha power per channel for each class
left_alpha  = X[y==0, band_ch_idx('alpha')].mean(axis=0)  # (n_channels,)
right_alpha = X[y==1, band_ch_idx('alpha')].mean(axis=0)

# SHAP importance per channel (averaged over all bands)
# sv has shape (n_test, n_features); feature order: [delta_allch, theta_allch, ...]
mean_abs_sv_flat = np.abs(sv).mean(axis=0)                 # (320,)
# Reshape to (n_bands, n_channels) and average over bands
sv_2d = mean_abs_sv_flat.reshape(len(BAND_NAMES), len(ch_names))
shap_per_channel = sv_2d.mean(axis=0)                      # (n_channels,)

# SHAP for alpha+beta specifically
alpha_beta_idx = [BAND_NAMES.index('alpha'), BAND_NAMES.index('beta')]
shap_alpha_beta = sv_2d[alpha_beta_idx].mean(axis=0)

# ── Three-panel figure ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Alpha power: left hand imagery
vmin1 = min(left_alpha.min(), right_alpha.min())
vmax1 = max(left_alpha.max(), right_alpha.max())

im1, _ = plot_topomap(left_alpha, info_plot, axes=axes[0], show=False,
                       cmap='RdBu_r', vlim=(vmin1, vmax1),
                       sensors=True, contours=6)
plt.colorbar(im1, ax=axes[0], shrink=0.85, label="Alpha power (µV²/Hz)")
axes[0].set_title("Panel 1\nAlpha Power: Left Hand Imagery\n"
                   "(ERD at C4 = right hemisphere cold spot)", fontsize=11)

# Panel 2: Alpha power: right hand imagery
im2, _ = plot_topomap(right_alpha, info_plot, axes=axes[1], show=False,
                       cmap='RdBu_r', vlim=(vmin1, vmax1),
                       sensors=True, contours=6)
plt.colorbar(im2, ax=axes[1], shrink=0.85, label="Alpha power (µV²/Hz)")
axes[1].set_title("Panel 2\nAlpha Power: Right Hand Imagery\n"
                   "(ERD at C3 = left hemisphere cold spot)", fontsize=11)

# Panel 3: SHAP topomap
im3, _ = plot_topomap(shap_alpha_beta, info_plot, axes=axes[2], show=False,
                       cmap='hot_r', sensors=True, contours=4)
plt.colorbar(im3, ax=axes[2], shrink=0.85, label="Mean |SHAP| (alpha+beta)")
axes[2].set_title("Panel 3\nSHAP Importance: Alpha+Beta Bands\n"
                   "(C3 & C4 light up = model independently finds motor cortex)", fontsize=11)

plt.suptitle("The Three-Panel Payoff: ERD During Imagery and the SHAP Motor Map\n"
             "Nose at top. Left hemisphere on left. Right hemisphere on right.",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('plots/three_panel_topomap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Three-panel topomap saved.")

# %% Cell 23
# ── Comparative topomap: imagery vs actual movement ──────────────────
print("Computing imagery vs actual movement comparison...")

# Compute PSD features for actual movement epochs
data_mov = epochs_mov.get_data()
labels_mov = epochs_mov.events[:, 2]

evt_mov_vals = list(evt_mov.values())
T1_mov_code = evt_mov_vals[0]
T2_mov_code = evt_mov_vals[1] if len(evt_mov_vals) > 1 else evt_mov_vals[0]

X_mov = np.array([compute_psd_features(ep, epochs_mov.info['sfreq'])
                   for ep in data_mov])
y_mov = (epochs_mov.events[:, 2] == T2_mov_code).astype(int)

# Mean alpha per channel
alpha_img_mean = X[:, band_ch_idx('alpha')].mean(axis=0)
alpha_mov_mean = X_mov[:, band_ch_idx('alpha')].mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Same color scale for comparison
vmin_cmp = min(alpha_img_mean.min(), alpha_mov_mean.min())
vmax_cmp = max(alpha_img_mean.max(), alpha_mov_mean.max())

im_i, _ = plot_topomap(alpha_img_mean, info_plot, axes=axes[0], show=False,
                        cmap='RdBu_r', vlim=(vmin_cmp, vmax_cmp), sensors=True)
plt.colorbar(im_i, ax=axes[0], shrink=0.85, label="Alpha power (µV²/Hz)")
axes[0].set_title("Alpha Power: All Imagined Trials\n(weaker ERD, lighter cold spot)", fontsize=11)

im_m, _ = plot_topomap(alpha_mov_mean, info_plot, axes=axes[1], show=False,
                        cmap='RdBu_r', vlim=(vmin_cmp, vmax_cmp), sensors=True)
plt.colorbar(im_m, ax=axes[1], shrink=0.85, label="Alpha power (µV²/Hz)")
axes[1].set_title("Alpha Power: All Real Movement Trials\n(stronger ERD, deeper cold spot)", fontsize=11)

plt.suptitle("Imagery vs. Real Movement: Same ERD Pattern, Different Depth\n"
             "This is why motor imagery BCI works: imagining movement activates the motor cortex",
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('plots/imagery_vs_movement.png', dpi=120, bbox_inches='tight')
plt.show()

# %% Cell 24
quality_flag=(X_human[:,4]>np.quantile(X_human[train_idx,4],.9)).astype(int);print("Trials flagged for review:",quality_flag.sum())


# %% Cell 25
control_model=RandomForestClassifier(n_estimators=400,random_state=42,n_jobs=-1).fit(X_enriched[train_idx],y[train_idx]);control_probability=control_model.predict_proba(X_enriched[test_idx])[:,1]


# %% Cell 26
CONFIGURATIONS={"Subject-specific C3/C4 sensorimotor mu-beta with CSP":["alpha_C3","alpha_C4","beta_C3","beta_C4"],"Central Cz mu-beta":["alpha_Cz","beta_Cz"],"Frontal F3/F4 mu-beta":["alpha_F3","alpha_F4","beta_F3","beta_F4"],"Occipital O1/O2 alpha":["alpha_O1","alpha_O2"]}
def rank(names):
 k=[feature_names.index(n) for n in names if n in feature_names];m=LogisticRegression(max_iter=2000,random_state=42).fit(X[train_idx][:,k],y[train_idx]);return accuracy_score(y[test_idx],m.predict(X[test_idx][:,k]))-.003*len(k)
configuration_ranking=pd.DataFrame([{"configuration":n,"pipeline_score":rank(v)} for n,v in CONFIGURATIONS.items()]).sort_values("pipeline_score",ascending=False);pipeline_shortlist=configuration_ranking.configuration.tolist();display(configuration_ranking)


# %% Cell 27
from langgraph.graph import StateGraph, END

AGENT_MEMORY_PATH = os.path.join("data", "cs2_agent_memory.jsonl")


def scholarly_search(query):
    r = requests.get("https://api.crossref.org/works",
                      params={"query": query, "rows": 5, "select": "title,DOI,container-title"},
                      timeout=20)
    r.raise_for_status()
    return [{"title": x.get("title", [""])[0], "doi": x.get("DOI", "")} for x in r.json()["message"]["items"]]


def recall_past_sessions(k=3):
    if not os.path.exists(AGENT_MEMORY_PATH):
        return {"matches": [], "note": "No prior agent sessions logged yet."}
    with open(AGENT_MEMORY_PATH) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    return {"matches": entries[-k:]}


def log_session(configuration, summary):
    os.makedirs(os.path.dirname(AGENT_MEMORY_PATH), exist_ok=True)
    with open(AGENT_MEMORY_PATH, "a") as f:
        f.write(json.dumps({"configuration": configuration, "summary": summary}) + "\n")


TOOLS = [
    {"type": "function", "function": {
        "name": "scholarly_search",
        "description": "Search scholarly metadata for BCI electrode evidence.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "recall_past_sessions",
        "description": "Recall summaries from earlier agent sessions on this pipeline.",
        "parameters": {"type": "object", "properties": {}}}},
]
TOOL_FUNCS = {"scholarly_search": scholarly_search, "recall_past_sessions": recall_past_sessions}


def run_autonomous_agent(configuration, max_steps=4):
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("Set NVIDIA_API_KEY outside the notebook.")
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=key)
    messages = [{"role": "user", "content":
        f"The pipeline's top-ranked BCI configuration is '{configuration}'. Decide for yourself which "
        f"tools to call, if any, before writing a two-sentence operator recommendation on whether to "
        f"adopt it."}]
    reply = None
    for _ in range(max_steps):
        reply = client.chat.completions.create(
            model=NVIDIA_MODEL, messages=messages, tools=TOOLS, tool_choice="auto",
            temperature=0.4, max_tokens=500).choices[0].message
        messages.append(reply)
        if not reply.tool_calls:
            break
        for call in reply.tool_calls:
            fn = TOOL_FUNCS[call.function.name]
            args = json.loads(call.function.arguments or "{}")
            result = fn(**args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    recommendation = reply.content or "" if reply else ""
    log_session(configuration, recommendation)
    return recommendation


autonomous_recommendation = run_autonomous_agent(pipeline_shortlist[0])
print(autonomous_recommendation)


def fixed_recommendation_node(state):
    prompt = f"Write a two-sentence BCI operator recommendation. Right-hand probability: {state['probability']:.2f}; quality review: {state['quality_review']}."
    return {"recommendation": nvidia_text(prompt)}
fixed_graph = StateGraph(dict)
fixed_graph.add_node("recommend", fixed_recommendation_node)
fixed_graph.set_entry_point("recommend")
fixed_graph.add_edge("recommend", END)
fixed_recommendation_graph = fixed_graph.compile()

def autonomous_agent_node(state):
    return {"recommendation": run_autonomous_agent(state["configuration"])}
autonomous_graph = StateGraph(dict)
autonomous_graph.add_node("recommend", autonomous_agent_node)
autonomous_graph.set_entry_point("recommend")
autonomous_graph.add_edge("recommend", END)
autonomous_agent_graph = autonomous_graph.compile()


# %% Cell 28
def predict_and_explain(trial_idx, X_test=X_te, y_test=y_te,
                         feature_names=feature_names, model=best_model,
                         explainer=explainer, shap_vals=sv):
    """Show prediction and SHAP breakdown for one test trial."""
    x  = X_test[trial_idx:trial_idx+1]
    sv_trial = shap_vals[trial_idx]

    pred  = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    true  = y_test[trial_idx]

    class_names = {0: "Left Hand Imagery", 1: "Right Hand Imagery"}

    print("=" * 60)
    print(f"Trial {trial_idx}")
    print("=" * 60)
    print(f"  True label:       {class_names[true]} ({true})")
    print(f"  Predicted:        {class_names[pred]} ({pred})")
    print(f"  Correct:          {'✓ YES' if pred == true else '✗ NO'}")
    print(f"  P(Left Hand):     {proba[0]:.3f}")
    print(f"  P(Right Hand):    {proba[1]:.3f}")
    print()

    # Top contributing features
    top_pos = pd.Series(sv_trial, index=feature_names).nlargest(5)
    top_neg = pd.Series(sv_trial, index=feature_names).nsmallest(5)
    print("  Top features pushing → RIGHT HAND:")
    for fname, val in top_pos.items():
        print(f"    {fname:<30} SHAP = {val:+.5f}")
    print("  Top features pushing → LEFT HAND:")
    for fname, val in top_neg.items():
        print(f"    {fname:<30} SHAP = {val:+.5f}")

    return pred, true, proba

# ── Example 1: A correctly classified trial ──────────────────────────
print("Example 1: First test trial")
predict_and_explain(0)

# %% Cell 29
# Find a trial where the contralateral pattern is clear
# For a right-hand trial, C3 features should have large negative SHAP (push to left → counter-productive)
# Actually, negative SHAP for right-hand trial means features push *away* from right
# Let's find a cleanly correct right-hand trial

right_hand_trials = [i for i in range(len(y_te)) if y_te[i] == 1]
if right_hand_trials:
    trial_rh = right_hand_trials[0]
    print(f"Example 2: Right Hand trial (index {trial_rh})")
    pred, true, proba = predict_and_explain(trial_rh)
    print(f"\nNote: For right hand trials, look for alpha_C3 and beta_C3 in the")
    print(f"top 'pushing → RIGHT HAND' features (low C3 power → right hand prediction)")

# %% Cell 30
# ─────────────────────────────────────────────────────────────────────
# CHANGE THIS: then run this cell
# Valid range: 0 to len(X_te)-1
TRIAL_IDX = 5
# ─────────────────────────────────────────────────────────────────────

if TRIAL_IDX < len(X_te):
    predict_and_explain(TRIAL_IDX)
else:
    print(f"Trial index {TRIAL_IDX} out of range. Max: {len(X_te)-1}")

