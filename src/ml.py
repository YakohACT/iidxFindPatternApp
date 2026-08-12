"""特徴量行列から譜面パターンを発見・保存・予測するモジュール。

処理の流れ:

1. StandardScaler で特徴量 (17 次元) を標準化する
2. 標準化した全次元に対して KMeans でクラスタリングする
   (クラスタ数はシルエット + Calinski-Harabasz の合成スコアで自動決定)
3. PCA は **結果の 2 次元可視化 (散布図・座標出力) にのみ** 使用する。
   クラスタリング自体は次元削減せず 17 次元のまま行う。

学習後のモデル (Scaler / KMeans / PCA / クラスタ統計 / 学習母集団の統計) は
``data/model/model.pkl`` に永続化される。次回起動時に ``load_model`` で呼び出し、
URL に対応する譜面の傾向を予測 (``predict``) できる。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from features import FeatureVector


MODEL_FILE_NAME = "model.pkl"


@dataclass
class PatternResult:
    labels: np.ndarray
    centroids: np.ndarray          # 標準化空間でのクラスタ重心
    centroids_2d: np.ndarray
    coords_2d: np.ndarray
    n_clusters: int
    silhouette: float
    ch_score: float                # Calinski-Harabasz 指標
    k_scores: pd.DataFrame         # k ごとの全スコア (空の場合あり)
    feature_importance: pd.DataFrame  # クラスタ毎の特徴量平均 (元スケール)
    scaler: StandardScaler
    kmeans: KMeans
    pca: PCA
    cluster_names: dict[int, str]  # 各クラスタの自然言語ラベル
    feature_mean: pd.Series        # 学習母集団の特徴量平均 (元スケール)
    feature_std: pd.Series         # 学習母集団の特徴量標準偏差 (ddof=0)


def _choose_k(
    x: np.ndarray,
    k_min: int = 2,
    k_max: int = 20,
    method: str = "combined",
) -> tuple[int, float, float, pd.DataFrame]:
    """最適なクラスタ数を複数指標で評価して選ぶ。

    KMeans + シルエット単独だと k=2 を選びやすい既知の偏りがあるため、
    Calinski-Harabasz (CH) 指標との組合せで評価できるようにする。

    method:
        - "silhouette": シルエット単独 (古来の挙動)
        - "ch":         Calinski-Harabasz 単独 (k を大きく選ぶ傾向)
        - "combined":   両方の min-max 正規化済みスコアを平均 (既定)

    シルエットが極端に小さい k=2 が "勝つ" のを避けたい場合は ``k_min`` を
    上げて (例: 5) 強制的に細かく分けることもできる。

    Returns:
        best_k, silhouette_at_best_k, ch_at_best_k, dataframe_of_all_scores
    """
    n_samples = x.shape[0]
    if n_samples < 4:
        k = max(min(n_samples, 2), 1)
        return k, 0.0, 0.0, pd.DataFrame()

    k_max = min(k_max, n_samples - 1)
    if k_min > k_max:
        k_min = k_max

    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(x)
        if len(np.unique(labels)) < 2:
            continue
        s = float(silhouette_score(x, labels))
        ch = float(calinski_harabasz_score(x, labels))
        rows.append({"k": k, "silhouette": s, "ch": ch})

    if not rows:
        return k_min, 0.0, 0.0, pd.DataFrame()

    scores = pd.DataFrame(rows).set_index("k")

    # min-max 正規化して合成スコアを計算
    def _norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        if rng < 1e-12:
            return pd.Series(0.5, index=s.index)
        return (s - s.min()) / rng

    scores["silhouette_norm"] = _norm(scores["silhouette"])
    scores["ch_norm"] = _norm(scores["ch"])
    scores["combined"] = (scores["silhouette_norm"] + scores["ch_norm"]) / 2.0

    if method == "silhouette":
        best_k = int(scores["silhouette"].idxmax())
    elif method == "ch":
        best_k = int(scores["ch"].idxmax())
    else:
        best_k = int(scores["combined"].idxmax())

    return (
        best_k,
        float(scores.loc[best_k, "silhouette"]),
        float(scores.loc[best_k, "ch"]),
        scores,
    )


# クラスタ命名: 全体平均からの z-score が大きい特徴量でラベルを付ける
# (特徴量名, ラベル, 方向)。方向 +1 は「平均より高いとき」、-1 は「低いとき」。
_TAG_DEFS: list[tuple[str, str, int]] = [
    ("stair_mono_ratio", "単階段", +1),
    ("stair_turn_ratio", "折返し階段", +1),
    ("big_stair_rate", "大階段", +1),
    ("double_stair_ratio", "二重階段", +1),
    ("garbage_stair_ratio", "ゴミ付き階段", +1),
    ("denim_ratio", "デニム", +1),
    ("chord_trill_ratio", "二重トリル", +1),
    ("trill_ratio", "トリル", +1),
    ("jack_ratio", "縦連", +1),
    ("wall_ratio", "壁", +1),
    ("axis_ratio", "軸", +1),
    ("scratch_stream_ratio", "連皿", +1),
    ("scratch_key_mix_ratio", "皿複合", +1),
    ("scratch_left_chord_ratio", "無理皿", +1),
    ("scratch_ratio", "皿多め", +1),
    ("mss_rate", "MSS", +1),
    ("stream16_ratio", "乱打", +1),
    ("peak_density", "発狂あり", +1),
    ("peak_position", "終盤発狂", +1),
    ("density", "高密度", +1),
    ("density", "低密度", -1),
    ("max_chord", "同時押し多", +1),
    ("cn_ratio", "CN多め", +1),
    ("offgrid_ratio", "ズレ/裏拍", +1),
    ("hand_bias", "片手偏重", +1),
    ("bpm_range_log2", "ソフラン(幅)", +1),
    ("bpm_change_rate", "ソフラン(頻)", +1),
    ("random_advantage", "乱推奨", +1),
    ("random_advantage", "正規向き", -1),
    ("random_gamble", "乱ガチャ", +1),
    ("mirror_advantage", "鏡推奨", +1),
]

_TAG_Z_THRESHOLD = 0.6  # このクラスタ平均が全体平均から何σ 離れていれば特徴とみなすか
_TAG_MAX = 4            # ラベルに採用する最大タグ数


def _describe_cluster(
    profile_row: pd.Series,
    global_mean: pd.Series,
    global_std: pd.Series,
) -> str:
    """クラスタの特徴量平均を全体分布と比較し、傾向ラベルを生成する。"""
    std = global_std.replace(0, 1.0)
    z = (profile_row - global_mean) / std
    candidates: list[tuple[float, str]] = []
    for name, label, direction in _TAG_DEFS:
        if name not in z:
            continue
        score = float(z[name]) * direction
        if score > _TAG_Z_THRESHOLD:
            candidates.append((score, label))
    candidates.sort(reverse=True)
    tags = [label for _score, label in candidates[:_TAG_MAX]]
    if not tags:
        tags.append("標準")
    return "・".join(tags)


def find_patterns(
    matrix: np.ndarray,
    items: list[FeatureVector],
    n_clusters: int | None = None,
    k_min: int = 2,
    k_max: int = 20,
    score_method: str = "combined",
) -> PatternResult:
    if matrix.shape[0] == 0:
        raise ValueError("特徴量行列が空です。スクレイピング結果を確認してください。")

    scaler = StandardScaler()
    x = scaler.fit_transform(matrix)

    k_scores = pd.DataFrame()
    if n_clusters is None:
        n_clusters, silhouette, ch_score, k_scores = _choose_k(
            x, k_min=k_min, k_max=k_max, method=score_method
        )
    else:
        km_tmp = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels_tmp = km_tmp.fit_predict(x)
        if len(np.unique(labels_tmp)) > 1:
            silhouette = float(silhouette_score(x, labels_tmp))
            ch_score = float(calinski_harabasz_score(x, labels_tmp))
        else:
            silhouette, ch_score = 0.0, 0.0

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(x)

    pca = PCA(n_components=min(2, x.shape[1]))
    coords_2d = pca.fit_transform(x)
    centroids_2d = pca.transform(kmeans.cluster_centers_)

    feature_names = items[0].feature_names
    df = pd.DataFrame(matrix, columns=feature_names)
    df["cluster"] = labels
    cluster_means = df.groupby("cluster").mean()

    features_only = df.drop(columns=["cluster"])
    global_mean = features_only.mean()
    global_std = features_only.std(ddof=0).replace(0, 1.0)
    cluster_names = {
        int(c): _describe_cluster(cluster_means.loc[c], global_mean, global_std)
        for c in cluster_means.index
    }

    return PatternResult(
        labels=labels,
        centroids=kmeans.cluster_centers_,
        centroids_2d=centroids_2d,
        coords_2d=coords_2d,
        n_clusters=n_clusters,
        silhouette=silhouette,
        ch_score=ch_score,
        k_scores=k_scores,
        feature_importance=cluster_means,
        scaler=scaler,
        kmeans=kmeans,
        pca=pca,
        cluster_names=cluster_names,
        feature_mean=global_mean,
        feature_std=global_std,
    )


def save_model(result: PatternResult, model_dir: Path) -> Path:
    """学習済みモデル一式を 1 ファイルに保存する。"""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / MODEL_FILE_NAME
    joblib.dump(
        {
            "scaler": result.scaler,
            "kmeans": result.kmeans,
            "pca": result.pca,
            "cluster_profiles": result.feature_importance,
            "cluster_names": result.cluster_names,
            "n_clusters": result.n_clusters,
            "silhouette": result.silhouette,
            "ch_score": result.ch_score,
            "feature_mean": result.feature_mean,
            "feature_std": result.feature_std,
        },
        path,
    )
    return path


@dataclass
class TrainedModel:
    scaler: StandardScaler
    kmeans: KMeans
    pca: PCA
    cluster_profiles: pd.DataFrame
    cluster_names: dict[int, str]
    n_clusters: int
    silhouette: float
    ch_score: float = 0.0
    feature_mean: pd.Series | None = None  # 学習母集団の平均 (旧モデルでは None)
    feature_std: pd.Series | None = None   # 学習母集団の標準偏差 (同上)


def load_model(model_dir: Path) -> TrainedModel | None:
    path = model_dir / MODEL_FILE_NAME
    if not path.exists():
        return None
    data = joblib.load(path)
    return TrainedModel(
        scaler=data["scaler"],
        kmeans=data["kmeans"],
        pca=data["pca"],
        cluster_profiles=data["cluster_profiles"],
        cluster_names=data["cluster_names"],
        n_clusters=data["n_clusters"],
        silhouette=data["silhouette"],
        ch_score=float(data.get("ch_score", 0.0)),
        feature_mean=data.get("feature_mean"),
        feature_std=data.get("feature_std"),
    )


@dataclass
class Prediction:
    cluster: int
    cluster_name: str
    distance: float           # 標準化空間での重心からの距離
    distance_rank: int        # 全クラスタ重心の中で何番目に近いか (1=最近接)
    feature_values: dict[str, float]
    top_features: list[tuple[str, float]]  # (特徴量名, 全体平均との差の z-score)
    pca_xy: tuple[float, float]


def predict(model: TrainedModel, fv: FeatureVector) -> Prediction:
    """1 譜面の特徴量から、最も近いクラスタとその傾向を返す。"""
    feature_names = list(model.cluster_profiles.columns)
    # FeatureVector は標準名順だが、保存時の列順に合わせ直す
    name_to_value = dict(zip(fv.feature_names, fv.values))
    x_raw = np.array(
        [[name_to_value.get(n, 0.0) for n in feature_names]], dtype=float
    )
    x = model.scaler.transform(x_raw)

    centers = model.kmeans.cluster_centers_
    dists = np.linalg.norm(centers - x, axis=1)
    cluster = int(np.argmin(dists))
    rank_order = np.argsort(dists)
    distance_rank = int(np.where(rank_order == cluster)[0][0]) + 1

    coords = model.pca.transform(x)[0]
    pca_xy = (float(coords[0]), float(coords[1]) if coords.size > 1 else 0.0)

    # この譜面が「全体平均と比べてどの特徴で外れているか」を z-score で算出
    # (学習時に保存した母集団統計を使う。旧モデルはクラスタ重心の統計で代用)
    if model.feature_mean is not None and model.feature_std is not None:
        global_mean = model.feature_mean.reindex(feature_names)
        global_std = model.feature_std.reindex(feature_names).replace(0, 1.0)
    else:
        global_mean = model.cluster_profiles.mean()
        global_std = model.cluster_profiles.std(ddof=0).replace(0, 1.0)
    z = (pd.Series(x_raw[0], index=feature_names) - global_mean) / global_std
    top_feats = sorted(
        ((n, float(v)) for n, v in z.items()),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )[:5]

    return Prediction(
        cluster=cluster,
        cluster_name=model.cluster_names.get(cluster, ""),
        distance=float(dists[cluster]),
        distance_rank=distance_rank,
        feature_values={n: float(v) for n, v in zip(feature_names, x_raw[0])},
        top_features=top_feats,
        pca_xy=pca_xy,
    )


def save_report(
    result: PatternResult,
    items: list[FeatureVector],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item, label, (x2, y2) in zip(items, result.labels, result.coords_2d):
        row = {
            "song_id": item.song_id,
            "title": item.title,
            "level": item.level,
            "difficulty": item.difficulty,
            "cluster": int(label),
            "cluster_name": result.cluster_names.get(int(label), ""),
            "pca_x": float(x2),
            "pca_y": float(y2),
        }
        for name, value in zip(item.feature_names, item.values):
            row[name] = float(value)
        rows.append(row)
    df_all = pd.DataFrame(rows)
    df_all.to_csv(output_dir / "assignments.csv", index=False)

    # クラスタ別の CSV を別ディレクトリにも書き出す
    clusters_dir = output_dir / "clusters"
    # 既存のクラスタ CSV を一旦掃除 (前回学習との混在を防ぐ)
    if clusters_dir.exists():
        for old in clusters_dir.glob("cluster_*.csv"):
            old.unlink()
    clusters_dir.mkdir(parents=True, exist_ok=True)
    for cluster_id, group in df_all.groupby("cluster"):
        name = result.cluster_names.get(int(cluster_id), "")
        # ファイル名に傾向ラベルも含める (Windows で安全な文字に置換)
        safe_name = re.sub(r"[^\w]+", "_", name).strip("_")
        fname = f"cluster_{int(cluster_id)}{('_' + safe_name) if safe_name else ''}.csv"
        group.sort_values(["level", "title"], na_position="last").to_csv(
            clusters_dir / fname, index=False
        )

    # クラスタ別特徴量平均 + 命名
    profiles = result.feature_importance.copy()
    profiles.insert(
        0, "cluster_name",
        [result.cluster_names.get(int(c), "") for c in profiles.index],
    )
    profiles.to_csv(output_dir / "cluster_profiles.csv")

    summary = {
        "n_charts": len(items),
        "n_clusters": result.n_clusters,
        "silhouette_score": result.silhouette,
        "calinski_harabasz_score": result.ch_score,
    }
    pd.Series(summary).to_csv(output_dir / "summary.csv", header=False)

    # k 候補ごとのスコアもファイルに残す
    if not result.k_scores.empty:
        result.k_scores.to_csv(output_dir / "k_scores.csv")

    # 散布図
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(
            result.coords_2d[:, 0],
            result.coords_2d[:, 1]
            if result.coords_2d.shape[1] > 1
            else np.zeros(result.coords_2d.shape[0]),
            c=result.labels,
            cmap="tab10",
            s=30,
            alpha=0.75,
        )
        ax.scatter(
            result.centroids_2d[:, 0],
            result.centroids_2d[:, 1]
            if result.centroids_2d.shape[1] > 1
            else np.zeros(result.centroids_2d.shape[0]),
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
