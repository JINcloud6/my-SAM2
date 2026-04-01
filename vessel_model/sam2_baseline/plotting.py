import matplotlib.pyplot as plt
import numpy as np


def plot_time_value_heatmap(
    values_per_frame,
    title,
    savepath,
    n_time_bins=40,
    n_val_bins=40,
    val_range=None,
    log_scale=True,
):
    """
    values_per_frame: list[np.ndarray|None], samples per frame
    Build heatmap: x=time(frame), y=value bins, color=freq
    """
    T = len(values_per_frame)
    if T == 0:
        return

    xs = []
    ys = []
    for t, arr in enumerate(values_per_frame):
        if arr is None:
            continue
        arr = np.asarray(arr).reshape(-1)
        if arr.size == 0:
            continue
        xs.append(np.full(arr.shape, t, dtype=np.float32))
        ys.append(arr.astype(np.float32))

    if len(xs) == 0:
        return

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    x_bins = np.linspace(0, T, n_time_bins + 1)
    if val_range is None:
        y_min, y_max = float(np.nanpercentile(y, 1)), float(np.nanpercentile(y, 99))
        if y_max <= y_min + 1e-6:
            y_min, y_max = float(np.min(y)), float(np.max(y) + 1e-6)
        y_range = (y_min, y_max)
    else:
        y_range = val_range
    y_bins = np.linspace(y_range[0], y_range[1], n_val_bins + 1)

    H, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    H = H.T

    H_show = np.log1p(H) if log_scale else H

    plt.figure(figsize=(7, 4))
    plt.imshow(
        H_show,
        origin="lower",
        aspect="auto",
        extent=[0, T, y_bins[0], y_bins[-1]],
    )
    plt.title(title)
    plt.xlabel("frame idx")
    plt.ylabel("value")
    plt.colorbar(label="log count" if log_scale else "count")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_spatial_heatmap(heatmap, savepath, base_image=None, alpha=0.45, cmap="magma"):
    if heatmap is None:
        return
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2:
        return

    hm_min, hm_max = float(np.min(hm)), float(np.max(hm))
    if hm_max <= hm_min + 1e-6:
        hm = np.zeros_like(hm, dtype=np.float32)
    else:
        hm = (hm - hm_min) / (hm_max - hm_min + 1e-6)

    plt.figure(figsize=(6, 5))
    if base_image is not None:
        plt.imshow(base_image, cmap="gray")
        plt.imshow(hm, cmap=cmap, alpha=alpha)
    else:
        plt.imshow(hm, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()
