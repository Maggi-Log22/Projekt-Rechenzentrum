# Erweiterter Modellvergleich: K-Means vs. GMM
import wfdb
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

base_dir = Path(".")
subjects = list(range(1, 21))
window_seconds = 30
overlap = 0.0
k_manual = 2

if not (0.0 <= overlap < 1.0):
    raise ValueError("overlap must be in [0.0, 1.0).")

step_seconds = window_seconds * (1.0 - overlap)


def extract_features():
    rows = []

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

        fs_acc = float(acc_record.fs)
        fs_hr = float(spo2_record.fs)

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

            rows.append(
                {
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
            )

    return pd.DataFrame(rows)


def preprocess(df):
    meta_cols = ["subject_id", "window_idx", "start_s", "end_s"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    lower = clean_df[feature_cols].quantile(0.01)
    upper = clean_df[feature_cols].quantile(0.99)
    clipped = clean_df[feature_cols].clip(lower=lower, upper=upper, axis=1)

    means = clipped.mean(axis=0)
    stds = clipped.std(axis=0).replace(0, 1.0)
    X = ((clipped - means) / stds).to_numpy()

    meta = clean_df[meta_cols].reset_index(drop=True)
    return X, meta


def evaluate_models(X, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X)

    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
    gmm_labels = gmm.fit_predict(X)

    metrics = {
        "km_silhouette": silhouette_score(X, km_labels),
        "km_inertia": km.inertia_,
        "gmm_silhouette": silhouette_score(X, gmm_labels),
        "gmm_bic": gmm.bic(X),
        "gmm_aic": gmm.aic(X),
        "ari_km_vs_gmm": adjusted_rand_score(km_labels, gmm_labels),
        "nmi_km_vs_gmm": normalized_mutual_info_score(km_labels, gmm_labels),
    }

    return km_labels, gmm_labels, metrics


def robustness_over_seeds(X, k, seeds=range(10)):
    km_sil, gmm_sil = [], []
    km_labels_ref, gmm_labels_ref = None, None
    km_ari_to_ref, gmm_ari_to_ref = [], []

    for seed in seeds:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km_labels = km.fit_predict(X)
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
        gmm_labels = gmm.fit_predict(X)

        km_sil.append(silhouette_score(X, km_labels))
        gmm_sil.append(silhouette_score(X, gmm_labels))

        if km_labels_ref is None:
            km_labels_ref = km_labels
            gmm_labels_ref = gmm_labels
        else:
            km_ari_to_ref.append(adjusted_rand_score(km_labels_ref, km_labels))
            gmm_ari_to_ref.append(adjusted_rand_score(gmm_labels_ref, gmm_labels))

    return {
        "km_sil_mean": float(np.mean(km_sil)),
        "km_sil_std": float(np.std(km_sil)),
        "gmm_sil_mean": float(np.mean(gmm_sil)),
        "gmm_sil_std": float(np.std(gmm_sil)),
        "km_stability_ari_mean": float(np.mean(km_ari_to_ref)) if km_ari_to_ref else np.nan,
        "gmm_stability_ari_mean": float(np.mean(gmm_ari_to_ref)) if gmm_ari_to_ref else np.nan,
    }


# Pipeline
raw_df = extract_features()
if raw_df.empty:
    raise RuntimeError("Keine Features extrahiert. Bitte Dateistruktur pruefen.")

X, meta = preprocess(raw_df)
km_labels, gmm_labels, scores = evaluate_models(X, k_manual)
robust = robustness_over_seeds(X, k_manual, seeds=range(10))

# Konsolen-Ausgabe
print("\n=== Modellvergleich K-Means vs. GMM ===")
print(f"k = {k_manual}, Samples = {X.shape[0]}, Features = {X.shape[1]}")
print(f"K-Means: Silhouette={scores['km_silhouette']:.4f}, Inertia={scores['km_inertia']:.2f}")
print(f"GMM:     Silhouette={scores['gmm_silhouette']:.4f}, BIC={scores['gmm_bic']:.2f}, AIC={scores['gmm_aic']:.2f}")
print(f"Agreement: ARI={scores['ari_km_vs_gmm']:.4f}, NMI={scores['nmi_km_vs_gmm']:.4f}")
print(
    "Robustheit (Seeds 0-9): "
    f"KM Sil={robust['km_sil_mean']:.4f}±{robust['km_sil_std']:.4f}, "
    f"GMM Sil={robust['gmm_sil_mean']:.4f}±{robust['gmm_sil_std']:.4f}, "
    f"KM Stab-ARI={robust['km_stability_ari_mean']:.4f}, "
    f"GMM Stab-ARI={robust['gmm_stability_ari_mean']:.4f}"
)

# Plot 1: Metrik-Vergleich
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].bar(["K-Means", "GMM"], [scores["km_silhouette"], scores["gmm_silhouette"]], color=["steelblue", "seagreen"])
axs[0].set_title("Silhouette")
axs[0].set_ylabel("Score")

axs[1].bar(["ARI", "NMI"], [scores["ari_km_vs_gmm"], scores["nmi_km_vs_gmm"]], color=["slateblue", "darkorange"])
axs[1].set_title("Label-Uebereinstimmung")
axs[1].set_ylabel("Score")

plt.tight_layout()
plt.show()

# Plot 2: Subject-Verteilung je Cluster
km_dist = pd.crosstab(meta["subject_id"], km_labels, normalize="index")
gmm_dist = pd.crosstab(meta["subject_id"], gmm_labels, normalize="index")

fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
km_dist.plot(kind="bar", stacked=True, ax=axs[0], colormap="Blues")
axs[0].set_title("K-Means: Clusteranteile pro Subject")
axs[0].set_ylabel("Anteil")
axs[0].legend(title="Cluster", loc="upper right")

gmm_dist.plot(kind="bar", stacked=True, ax=axs[1], colormap="Greens")
axs[1].set_title("GMM: Komponentenanteile pro Subject")
axs[1].set_ylabel("Anteil")
axs[1].set_xlabel("Subject ID")
axs[1].legend(title="Komponente", loc="upper right")

plt.tight_layout()
plt.show()

# Plot 3: Zuordnungs-Matrix K-Means vs GMM
cont = pd.crosstab(km_labels, gmm_labels)
plt.figure(figsize=(5, 4))
plt.imshow(cont.values, cmap="Blues", aspect="auto")
plt.colorbar(label="Anzahl Fenster")
plt.xticks(range(cont.shape[1]), cont.columns)
plt.yticks(range(cont.shape[0]), cont.index)
plt.xlabel("GMM-Komponente")
plt.ylabel("K-Means-Cluster")
plt.title("Kreuztabelle K-Means vs. GMM")

for i in range(cont.shape[0]):
    for j in range(cont.shape[1]):
        txt_color = "black" if cont.values[i, j] < cont.values.max() * 0.6 else "white"
        plt.text(j, i, str(cont.values[i, j]), ha="center", va="center", color=txt_color)

plt.tight_layout()
plt.show()
