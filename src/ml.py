"""特徴量行列から譜面パターンを発見するモジュール。

教師なし学習を 2 段階で適用する:

1. 標準化 + PCA で次元削減 (2 次元へ射影)
2. KMeans でクラスタリング (シルエットスコアでクラスタ数を自動決定)

結果は CSV/PNG として書き出す。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from features import FeatureVector


@dataclass
class PatternResult:
    labels: np.ndarray             # 各譜面のクラスタ ID
    centroids: np.ndarray          # 標準化空間でのクラスタ重心 (特徴量次元)
    centroids_2d: np.ndarray       # PCA 2D 空間での重心
    coords_2d: np.ndarray          # PCA 2D 空間での各譜面座標
    n_clusters: int
    silhouette: float
    feature_importance: pd.DataFrame  # クラスタ毎の特徴量平均


def _choose_k(x: np.ndarray, k_min: int = 2, k_max: int = 10) -> tuple[int, float]:
    """シルエットスコア最大のクラスタ数を選ぶ。"""
    n_samples = x.shape[0]
    if n_samples < 4:
        return max(min(n_samples, 2), 1), 0.0
    best_k = k_min
    best_score = -1.0
    k_max = min(k_max, n_samples - 1)
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(x)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(x, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, float(best_score)


def find_patterns(
    matrix: np.ndarray,
    items: list[FeatureVector],
    n_clusters: int | None = None,
) -> PatternResult:
    if matrix.shape[0] == 0:
        raise ValueError("特徴量行列が空です。スクレイピング結果を確認してください。")

    scaler = StandardScaler()
    x = scaler.fit_transform(matrix)

    if n_clusters is None:
        n_clusters, silhouette = _choose_k(x)
    else:
        km_tmp = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels_tmp = km_tmp.fit_predict(x)
        silhouette = (
            silhouette_score(x, labels_tmp) if len(np.unique(labels_tmp)) > 1 else 0.0
        )

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(x)

    # PCA で 2 次元に圧縮 (可視化用)
    pca = PCA(n_components=min(2, x.shape[1]))
    coords_2d = pca.fit_transform(x)
    centroids_2d = pca.transform(km.cluster_centers_)

    # クラスタ別の特徴量平均 (標準化前の値)
    feature_names = items[0].feature_names
    df = pd.DataFrame(matrix, columns=feature_names)
    df["cluster"] = labels
    cluster_means = df.groupby("cluster").mean()

    return PatternResult(
        labels=labels,
        centroids=km.cluster_centers_,
        centroids_2d=centroids_2d,
        coords_2d=coords_2d,
        n_clusters=n_clusters,
        silhouette=silhouette,
        feature_importance=cluster_means,
    )


def save_report(
    result: PatternResult,
    items: list[FeatureVector],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV: 譜面毎のクラスタ割当
    rows = []
    for item, label, (x2, y2) in zip(items, result.labels, result.coords_2d):
        row = {
            "song_id": item.song_id,
            "title": item.title,
            "level": item.level,
            "difficulty": item.difficulty,
            "cluster": int(label),
            "pca_x": float(x2),
            "pca_y": float(y2),
        }
        for name, value in zip(item.feature_names, item.values):
            row[name] = float(value)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "assignments.csv", index=False)

    # クラスタ別特徴量平均
    result.feature_importance.to_csv(output_dir / "cluster_profiles.csv")

    # サマリ
    summary = {
        "n_charts": len(items),
        "n_clusters": result.n_clusters,
        "silhouette_score": result.silhouette,
    }
    pd.Series(summary).to_csv(output_dir / "summary.csv", header=False)

    # 散布図
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(
            result.coords_2d[:, 0],
            result.coords_2d[:, 1] if result.coords_2d.shape[1] > 1 else np.zeros(result.coords_2d.shape[0]),
            c=result.labels,
            cmap="tab10",
            s=30,
            alpha=0.75,
        )
        ax.scatter(
            result.centroids_2d[:, 0],
            result.centroids_2d[:, 1] if result.centroids_2d.shape[1] > 1 else np.zeros(result.centroids_2d.shape[0]),
            marker="X",
            s=180,
            c="black",
            edgecolors="white",
            linewidths=1.5,
            label="centroid",
        )
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
        ax.set_title(
            f"IIDX chart pattern clusters (k={result.n_clusters}, "
            f"silhouette={result.silhouette:.3f})"
        )
        ax.legend(*scatter.legend_elements(), title="cluster", loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / "clusters.png", dpi=120)
        plt.close(fig)
    except ImportError:
        print("[ML] matplotlib が無いため散布図はスキップしました")
