# Gaussian Mixture Model auf allen Subjects (1-20)
import wfdb
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

base_dir = Path(".")
subjects = list(range(1, 21))

# --- Feature-Extraktion (30s Fenster) ---
window_seconds = 30
overlap = 0.0
if not (0.0 <= overlap < 1.0):
    raise ValueError("overlap must be in [0.0, 1.0).")
step_seconds = window_seconds * (1.0 - overlap)

all_features = []

for subject_id in subjects:
    subject_dir = base_dir / f"Subject_{subject_id}"
    acc_base = subject_dir / f"Subject{subject_id}_AccTempEDA"
    spo2_base = subject_dir / f"Subject{subject_id}_SpO2HR"

    if not acc_base.with_suffix(".hea").exists() or not spo2_base.with_suffix(".hea").exists():
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

        acc_mag = np.sqrt(acc_w[:, 0] ** 2 + acc_w[:, 1] ** 2 + acc_w[:, 2] ** 2)

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

# Feature-Matrix
_df = pd.DataFrame(all_features)

# --- Cleaning + Clipping + Standardisierung ---
feature_cols = [c for c in _df.columns if c not in ("subject_id", "window_idx", "start_s", "end_s")]
clean_df = _df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

lower = clean_df[feature_cols].quantile(0.01)
upper = clean_df[feature_cols].quantile(0.99)
clipped = clean_df[feature_cols].clip(lower=lower, upper=upper, axis=1)

means = clipped.mean(axis=0)
stds = clipped.std(axis=0).replace(0, 1.0)
X = ((clipped - means) / stds).to_numpy()

# --- Modellwahl: BIC / AIC ---
comp_range = range(2, 9)
bics = []
aics = []

for k in comp_range:
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
    gmm.fit(X)
    bics.append(gmm.bic(X))
    aics.append(gmm.aic(X))

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(list(comp_range), bics, marker="o")
axs[0].set_title("BIC")
axs[0].set_xlabel("k")
axs[0].set_ylabel("BIC")

axs[1].plot(list(comp_range), aics, marker="o", color="purple")
axs[1].set_title("AIC")
axs[1].set_xlabel("k")
axs[1].set_ylabel("AIC")

plt.tight_layout()
plt.show()

# --- k manuell festlegen ---
k_manual = 2
gmm = GaussianMixture(n_components=k_manual, covariance_type="full", random_state=42)
labels = gmm.fit_predict(X)

# --- PCA Visualisierung ---
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=20)
plt.title(f"GMM (k={k_manual}) – PCA 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

# --- Komponenten-Mittelwerte in Originalskala ---
comp_means = pd.DataFrame(clipped, columns=feature_cols)
comp_means["component"] = labels
comp_means = comp_means.groupby("component").mean().round(3)

fig = plt.figure(figsize=(14, 6))
plt.title("GMM Komponenten-Mittelwerte (Originalskala, gerundet)")
plt.axis("off")
table = plt.table(
    cellText=comp_means.values,
    colLabels=comp_means.columns,
    rowLabels=comp_means.index,
    loc="center",
    cellLoc="center",
)

try:
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.2)
    table.auto_set_column_width(col=list(range(len(comp_means.columns))))
except Exception:
    pass

plt.tight_layout()
plt.show()

# --- Einfache Interpretation (Heuristik) ---
interp = comp_means[["acc_mag_mean", "hr_mean", "eda_mean"]].copy()
interp["acc_rank"] = interp["acc_mag_mean"].rank(ascending=False)
interp["hr_rank"] = interp["hr_mean"].rank(ascending=False)
interp["eda_rank"] = interp["eda_mean"].rank(ascending=False)

def label_row(row):
    if row["acc_rank"] <= 1 and row["hr_rank"] <= 1:
        return "hohe Aktivität (Bewegung + HR)"
    if row["eda_rank"] <= 1 and row["hr_rank"] <= 2:
        return "Stress-ähnlich (EDA/HR hoch)"
    if row["acc_rank"] >= len(interp) and row["hr_rank"] >= len(interp):
        return "Ruhe/geringe Aktivität"
    return "gemischt / Übergang"

interp["interpretation"] = interp.apply(label_row, axis=1)

print("\nInterpretation (Heuristik) pro Komponente:")
print(interp[["acc_mag_mean", "hr_mean", "eda_mean", "interpretation"]])
