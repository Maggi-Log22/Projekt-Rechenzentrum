# TEIL A -- Subject_1 laden --
import wfdb
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

subject_dir = Path("Subject_1")
acc_record = wfdb.rdrecord(subject_dir / "Subject1_AccTempEDA")
spo2_record = wfdb.rdrecord(subject_dir / "Subject1_SpO2HR")

# Signale
acc = acc_record.p_signal[:, 0:3]   # Acc X/Y/Z
temp = acc_record.p_signal[:, 3]    # Temp
eda = acc_record.p_signal[:, 4]     # EDA
spo2 = spo2_record.p_signal[:, 0]   # SpO2
hr = spo2_record.p_signal[:, 1]     # HR

# Sampling Rates (in Hz)
fs_acc = acc_record.fs
fs_hr = spo2_record.fs

# TEIL C -- Feature-Extraktion (30s Fenster)
window_seconds = 30
overlap = 0.0
if not (0.0 <= overlap < 1.0):
    raise ValueError("overlap must be in [0.0, 1.0).")
step_seconds = window_seconds * (1.0 - overlap)
total_seconds = min(acc.shape[0] / fs_acc, hr.shape[0] / fs_hr)
num_windows = int((total_seconds - window_seconds) // step_seconds) + 1

features = []

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

    features.append(feat)

df = pd.DataFrame(features)
print(df.head())
print(df.shape)

# TEIL C2 -- ML-Ready: Cleaning + Clipping + Standardisierung
feature_cols = [c for c in df.columns if c not in ("window_idx", "start_s", "end_s")]

# 1) Entferne Zeilen mit NaN/Inf
clean_df = df.copy()
clean_df = clean_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

# 2) Einfaches Outlier-Clipping (1%-99% Perzentil je Feature)
lower = clean_df[feature_cols].quantile(0.01)
upper = clean_df[feature_cols].quantile(0.99)
clipped = clean_df[feature_cols].clip(lower=lower, upper=upper, axis=1)

# 3) Z-Score Standardisierung
means = clipped.mean(axis=0)
stds = clipped.std(axis=0).replace(0, 1.0)
scaled = (clipped - means) / stds

X = scaled.to_numpy()
feature_names = list(scaled.columns)

print("X shape:", X.shape)
print("Features:", feature_names)

# TEIL D -- Visualisierung
# A3: Signale mit vertikalen 30s-Fenstergrenzen
fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

acc_time_full = np.arange(acc.shape[0]) / fs_acc
temp_time_full = np.arange(temp.shape[0]) / fs_acc
eda_time_full = np.arange(eda.shape[0]) / fs_acc
hr_time_full = np.arange(hr.shape[0]) / fs_hr
spo2_time_full = np.arange(spo2.shape[0]) / fs_hr

axs[0].plot(acc_time_full, acc[:, 0], label="Acc X")
axs[0].plot(acc_time_full, acc[:, 1], label="Acc Y")
axs[0].plot(acc_time_full, acc[:, 2], label="Acc Z")
axs[0].set_ylabel("Acceleration")
axs[0].legend()

axs[1].plot(temp_time_full, temp, color="orange")
axs[1].set_ylabel("Temp (°C)")

axs[2].plot(eda_time_full, eda, color="green")
axs[2].set_ylabel("EDA")

axs[3].plot(hr_time_full, hr, color="red")
axs[3].set_ylabel("HR (bpm)")

axs[4].plot(spo2_time_full, spo2, color="blue")
axs[4].set_ylabel("SpO2 (%)")
axs[4].set_xlabel("Time (s)")

for i in range(num_windows + 1):
    t = i * window_seconds
    for ax in axs:
        ax.axvline(t, color="gray", linewidth=0.6, alpha=0.4)

plt.suptitle("Subject_1 – 30s Fenstergrenzen (vertikal)")
plt.tight_layout()
plt.show()

# B1: Feature-Verlauf pro Signal in separaten Subplots
window_centers = df["start_s"] + (window_seconds / 2)

fig, faxs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

faxs[0].plot(window_centers, df["acc_mag_mean"], label="acc_mag_mean")
faxs[0].set_ylabel("Acc Mag")
faxs[0].legend()

faxs[1].plot(window_centers, df["temp_mean"], color="orange", label="temp_mean")
faxs[1].set_ylabel("Temp")
faxs[1].legend()

faxs[2].plot(window_centers, df["eda_mean"], color="green", label="eda_mean")
faxs[2].set_ylabel("EDA")
faxs[2].legend()

faxs[3].plot(window_centers, df["hr_mean"], color="red", label="hr_mean")
faxs[3].set_ylabel("HR")
faxs[3].legend()

faxs[4].plot(window_centers, df["spo2_mean"], color="blue", label="spo2_mean")
faxs[4].set_ylabel("SpO2")
faxs[4].set_xlabel("Time (s)")
faxs[4].legend()

plt.suptitle("Feature-Verlauf pro 30s Fenster (je Signal)")
plt.tight_layout()
plt.show()

# Feature-Matrix als eigenes Fenster (Tabelle mit absoluten Werten)
feature_cols = [c for c in df.columns if c not in ("window_idx", "start_s", "end_s")]
display_df = df[feature_cols].round(3)

fig = plt.figure(figsize=(12, 6))
plt.title("Feature-Matrix (Fenster x Features)")
plt.axis("off")
table = plt.table(
    cellText=display_df.values,
    colLabels=display_df.columns,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.2)
plt.tight_layout()
plt.show()
