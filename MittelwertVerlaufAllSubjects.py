import numpy as np
import matplotlib.pyplot as plt
import wfdb
from pathlib import Path

# Konfiguration
base_dir = Path(".")
subject_ids = range(1, 21)

# Container
acc_x_list, acc_y_list, acc_z_list = [], [], []
temp_list, eda_list = [], []
hr_list, spo2_list = [], []

fs_acc_list, fs_hr_list = [], []
dur_acc_list, dur_hr_list = [], []


def resample_to_grid(signal, fs_src, t_target):
    t_src = np.arange(signal.shape[0]) / fs_src
    return np.interp(t_target, t_src, signal)


# Daten einlesen
for sid in subject_ids:
    subject_dir = base_dir / f"Subject_{sid}"
    acc_base = subject_dir / f"Subject{sid}_AccTempEDA"
    hr_base = subject_dir / f"Subject{sid}_SpO2HR"

    if not acc_base.with_suffix(".hea").exists() or not hr_base.with_suffix(".hea").exists():
        print(f"Skipping Subject_{sid}: Dateien fehlen")
        continue

    acc_record = wfdb.rdrecord(acc_base)
    hr_record = wfdb.rdrecord(hr_base)

    acc = acc_record.p_signal[:, 0:3]
    temp = acc_record.p_signal[:, 3]
    eda = acc_record.p_signal[:, 4]

    spo2 = hr_record.p_signal[:, 0]
    hr = hr_record.p_signal[:, 1]

    fs_acc = float(acc_record.fs)
    fs_hr = float(hr_record.fs)

    acc_x_list.append(acc[:, 0])
    acc_y_list.append(acc[:, 1])
    acc_z_list.append(acc[:, 2])
    temp_list.append(temp)
    eda_list.append(eda)
    hr_list.append(hr)
    spo2_list.append(spo2)

    fs_acc_list.append(fs_acc)
    fs_hr_list.append(fs_hr)
    dur_acc_list.append(acc.shape[0] / fs_acc)
    dur_hr_list.append(hr.shape[0] / fs_hr)


if not acc_x_list or not hr_list:
    raise RuntimeError("Keine verwertbaren Subject-Daten gefunden.")

# Gemeinsame Zeitachsen (auf minimale Dauer begrenzen)
target_fs_acc = min(fs_acc_list)
target_fs_hr = min(fs_hr_list)
common_dur_acc = min(dur_acc_list)
common_dur_hr = min(dur_hr_list)

t_acc = np.arange(0, common_dur_acc, 1.0 / target_fs_acc)
t_hr = np.arange(0, common_dur_hr, 1.0 / target_fs_hr)

# Resampling auf gemeinsame Gitter
acc_x_stack = np.vstack([resample_to_grid(s, fs, t_acc) for s, fs in zip(acc_x_list, fs_acc_list)])
acc_y_stack = np.vstack([resample_to_grid(s, fs, t_acc) for s, fs in zip(acc_y_list, fs_acc_list)])
acc_z_stack = np.vstack([resample_to_grid(s, fs, t_acc) for s, fs in zip(acc_z_list, fs_acc_list)])
temp_stack = np.vstack([resample_to_grid(s, fs, t_acc) for s, fs in zip(temp_list, fs_acc_list)])
eda_stack = np.vstack([resample_to_grid(s, fs, t_acc) for s, fs in zip(eda_list, fs_acc_list)])

hr_stack = np.vstack([resample_to_grid(s, fs, t_hr) for s, fs in zip(hr_list, fs_hr_list)])
spo2_stack = np.vstack([resample_to_grid(s, fs, t_hr) for s, fs in zip(spo2_list, fs_hr_list)])


def mean_std(stack):
    return np.mean(stack, axis=0), np.std(stack, axis=0)


acc_x_mean, acc_x_std = mean_std(acc_x_stack)
acc_y_mean, acc_y_std = mean_std(acc_y_stack)
acc_z_mean, acc_z_std = mean_std(acc_z_stack)
temp_mean, temp_std = mean_std(temp_stack)
eda_mean, eda_std = mean_std(eda_stack)
hr_mean, hr_std = mean_std(hr_stack)
spo2_mean, spo2_std = mean_std(spo2_stack)

# Plot: Mittelwert ± Standardabweichung
fig, axs = plt.subplots(7, 1, figsize=(14, 16), sharex=False)

series_acc = [
    (acc_x_mean, acc_x_std, "Acc X", "tab:blue"),
    (acc_y_mean, acc_y_std, "Acc Y", "tab:orange"),
    (acc_z_mean, acc_z_std, "Acc Z", "tab:green"),
    (temp_mean, temp_std, "Temp", "tab:red"),
    (eda_mean, eda_std, "EDA", "tab:purple"),
]

for i, (m, s, label, color) in enumerate(series_acc):
    axs[i].plot(t_acc, m, color=color, label=f"{label} Mittelwert")
    axs[i].fill_between(t_acc, m - s, m + s, color=color, alpha=0.2, label="±1 SD")
    axs[i].set_ylabel(label)
    axs[i].legend(loc="upper right")

axs[5].plot(t_hr, hr_mean, color="tab:brown", label="HR Mittelwert")
axs[5].fill_between(t_hr, hr_mean - hr_std, hr_mean + hr_std, color="tab:brown", alpha=0.2, label="±1 SD")
axs[5].set_ylabel("HR")
axs[5].legend(loc="upper right")

axs[6].plot(t_hr, spo2_mean, color="tab:cyan", label="SpO2 Mittelwert")
axs[6].fill_between(t_hr, spo2_mean - spo2_std, spo2_mean + spo2_std, color="tab:cyan", alpha=0.2, label="±1 SD")
axs[6].set_ylabel("SpO2")
axs[6].set_xlabel("Zeit (s)")
axs[6].legend(loc="upper right")

plt.suptitle("Aggregierte Verläufe über alle Subjects (Mittelwert ± SD)")
plt.tight_layout()
plt.show()
