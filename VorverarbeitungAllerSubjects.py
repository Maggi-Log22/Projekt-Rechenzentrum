# TEIL A -- Alle Subjects laden (1-20)
import wfdb
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = Path(".")
subjects = list(range(1, 21))

# Checkliste: ML-Vorbereitung
# ✅ Zeitfenster definieren (30 Sekunden)
# ✅ Feature-Extraktion (mean, std, min, max)
# ✅ Acc-Magnitude (mean, std)
# ✅ Feature-Matrix erstellen (DataFrame: Fenster × Features)
# ✅ Bereinigung (NaNs/Inf entfernen)
# ✅ Ausreißer prüfen (z. B. Perzentil-Clipping)
# ✅ Standardisierung (z-Score / StandardScaler)

# TEIL B -- Feature-Extraktion (30s Fenster)
window_seconds = 30
overlap = 0.0
if not (0.0 <= overlap < 1.0):
    raise ValueError("overlap must be in [0.0, 1.0).")
step_seconds = window_seconds * (1.0 - overlap)

all_features = []
acc_mag_series = []
temp_series = []
eda_series = []
hr_series = []
spo2_series = []
fs_acc_list = []
fs_hr_list = []
dur_acc_list = []
dur_hr_list = []

for subject_id in subjects:
    subject_dir = base_dir / f"Subject_{subject_id}"
    acc_base = subject_dir / f"Subject{subject_id}_AccTempEDA"
    spo2_base = subject_dir / f"Subject{subject_id}_SpO2HR"
    acc_hea = acc_base.with_suffix(".hea")
    spo2_hea = spo2_base.with_suffix(".hea")

    if not acc_hea.exists() or not spo2_hea.exists():
        print(f"Skipping Subject_{subject_id}: missing files")
        continue

    acc_record = wfdb.rdrecord(acc_base)
    spo2_record = wfdb.rdrecord(spo2_base)

    acc = acc_record.p_signal[:, 0:3]
    temp = acc_record.p_signal[:, 3]
    eda = acc_record.p_signal[:, 4]
    spo2 = spo2_record.p_signal[:, 0]
    hr = spo2_record.p_signal[:, 1]

    fs_acc = acc_record.fs
    fs_hr = spo2_record.fs

    total_seconds = min(acc.shape[0] / fs_acc, hr.shape[0] / fs_hr)
    dur_acc = acc.shape[0] / fs_acc
    dur_hr = hr.shape[0] / fs_hr
    if total_seconds < window_seconds:
        print(f"Skipping Subject_{subject_id}: too short")
        continue

    num_windows = int((total_seconds - window_seconds) // step_seconds) + 1

    for i in range(num_windows):
        start_t = i * step_seconds
        end_t = start_t + window_seconds

        start_acc = int(start_t * fs_acc)
        end_acc = int(end_t * fs_acc)
        start_hr = int(start_t * fs_hr)
        end_hr = int(end_t * fs_hr)

        acc_w = acc[start_acc:end_acc]
        temp_w = temp[start_acc:end_acc]
        eda_w = eda[start_acc:end_acc]
        hr_w = hr[start_hr:end_hr]
        spo2_w = spo2[start_hr:end_hr]

        acc_mag = np.sqrt(acc_w[:, 0]**2 + acc_w[:, 1]**2 + acc_w[:, 2]**2)

        feat = {
            "subject_id": subject_id,
            "window_idx": i,
            "start_s": start_t,
            "end_s": end_t,

            "acc_x_mean": np.mean(acc_w[:, 0]),
            "acc_x_std": np.std(acc_w[:, 0]),
            "acc_x_min": np.min(acc_w[:, 0]),
            "acc_x_max": np.max(acc_w[:, 0]),

            "acc_y_mean": np.mean(acc_w[:, 1]),
            "acc_y_std": np.std(acc_w[:, 1]),
            "acc_y_min": np.min(acc_w[:, 1]),
            "acc_y_max": np.max(acc_w[:, 1]),

            "acc_z_mean": np.mean(acc_w[:, 2]),
            "acc_z_std": np.std(acc_w[:, 2]),
            "acc_z_min": np.min(acc_w[:, 2]),
            "acc_z_max": np.max(acc_w[:, 2]),

            "acc_mag_mean": np.mean(acc_mag),
            "acc_mag_std": np.std(acc_mag),

            "temp_mean": np.mean(temp_w),
            "temp_std": np.std(temp_w),
            "temp_min": np.min(temp_w),
            "temp_max": np.max(temp_w),

            "eda_mean": np.mean(eda_w),
            "eda_std": np.std(eda_w),
            "eda_min": np.min(eda_w),
            "eda_max": np.max(eda_w),

            "hr_mean": np.mean(hr_w),
            "hr_std": np.std(hr_w),
            "hr_min": np.min(hr_w),
            "hr_max": np.max(hr_w),

            "spo2_mean": np.mean(spo2_w),
            "spo2_std": np.std(spo2_w),
            "spo2_min": np.min(spo2_w),
            "spo2_max": np.max(spo2_w),
        }

        all_features.append(feat)

    # Für Durchschnitts-Plot sammeln
    acc_mag = np.sqrt(acc[:, 0] ** 2 + acc[:, 1] ** 2 + acc[:, 2] ** 2)
    acc_mag_series.append(acc_mag)
    temp_series.append(temp)
    eda_series.append(eda)
    hr_series.append(hr)
    spo2_series.append(spo2)
    fs_acc_list.append(fs_acc)
    fs_hr_list.append(fs_hr)
    dur_acc_list.append(dur_acc)
    dur_hr_list.append(dur_hr)

# Feature-Matrix
_df = pd.DataFrame(all_features)

# TEIL C -- ML-Ready: Cleaning + Clipping + Standardisierung (global)
feature_cols = [c for c in _df.columns if c not in ("subject_id", "window_idx", "start_s", "end_s")]

clean_df = _df.copy()
clean_df = clean_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

lower = clean_df[feature_cols].quantile(0.01)
upper = clean_df[feature_cols].quantile(0.99)
clipped = clean_df[feature_cols].clip(lower=lower, upper=upper, axis=1)

means = clipped.mean(axis=0)
stds = clipped.std(axis=0).replace(0, 1.0)
scaled = (clipped - means) / stds

X = scaled.to_numpy()
feature_names = list(scaled.columns)
meta = clean_df[["subject_id", "window_idx", "start_s", "end_s"]].reset_index(drop=True)

# Ausgabe in separatem Fenster
df_head = _df.head(10).round(3)
feature_names_preview = ", ".join(feature_names[:10])
if len(feature_names) > 10:
    feature_names_preview += f", ... (+{len(feature_names) - 10} more)"

fig, axs = plt.subplots(2, 1, figsize=(16, 9))
axs[0].axis("off")
summary_text = (
    f"Feature-Matrix Shape: {_df.shape}\n"
    f"X Shape (scaled): {X.shape}\n"
    f"Meta Shape: {meta.shape}\n"
    f"Features (preview): {feature_names_preview}"
)
axs[0].text(0.01, 0.95, summary_text, va="top", ha="left", fontsize=10)

axs[1].axis("off")
table = axs[1].table(
    cellText=df_head.values,
    colLabels=df_head.columns,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(6)
table.scale(1, 1.1)
table.auto_set_column_width(col=list(range(len(df_head.columns))))

plt.suptitle("analysis3.py – Output Summary")
plt.tight_layout()
plt.show()

# Durchschnitts-Plot über alle Subjects (zeitlich ausgerichtet)
if acc_mag_series and hr_series:
    target_fs_acc = min(fs_acc_list)
    target_fs_hr = min(fs_hr_list)
    common_dur_acc = min(dur_acc_list)
    common_dur_hr = min(dur_hr_list)

    t_acc = np.arange(0, common_dur_acc, 1.0 / target_fs_acc)
    t_hr = np.arange(0, common_dur_hr, 1.0 / target_fs_hr)

    def resample(signal, fs, t_target):
        t_src = np.arange(signal.shape[0]) / fs
        return np.interp(t_target, t_src, signal)

    acc_mag_stack = np.vstack([resample(s, fs, t_acc) for s, fs in zip(acc_mag_series, fs_acc_list)])
    temp_stack = np.vstack([resample(s, fs, t_acc) for s, fs in zip(temp_series, fs_acc_list)])
    eda_stack = np.vstack([resample(s, fs, t_acc) for s, fs in zip(eda_series, fs_acc_list)])

    hr_stack = np.vstack([resample(s, fs, t_hr) for s, fs in zip(hr_series, fs_hr_list)])
    spo2_stack = np.vstack([resample(s, fs, t_hr) for s, fs in zip(spo2_series, fs_hr_list)])

    mean_acc_mag = np.mean(acc_mag_stack, axis=0)
    mean_temp = np.mean(temp_stack, axis=0)
    mean_eda = np.mean(eda_stack, axis=0)
    mean_hr = np.mean(hr_stack, axis=0)
    mean_spo2 = np.mean(spo2_stack, axis=0)

    fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=False)
    axs[0].plot(t_acc, mean_acc_mag, color="black")
    axs[0].set_ylabel("Acc Mag (mean)")

    axs[1].plot(t_acc, mean_temp, color="orange")
    axs[1].set_ylabel("Temp (mean)")

    axs[2].plot(t_acc, mean_eda, color="green")
    axs[2].set_ylabel("EDA (mean)")

    axs[3].plot(t_hr, mean_hr, color="red")
    axs[3].set_ylabel("HR (mean)")

    axs[4].plot(t_hr, mean_spo2, color="blue")
    axs[4].set_ylabel("SpO2 (mean)")
    axs[4].set_xlabel("Time (s)")

    plt.suptitle("Durchschnittlicher Verlauf über alle Subjects")
    plt.tight_layout()
    plt.show()
