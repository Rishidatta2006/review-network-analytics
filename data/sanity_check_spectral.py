#!/usr/bin/env python3
"""
sanity_check_spectral.py  (OPTIONAL - requires scikit-learn)
Quick feasibility test of the planned pipeline on data/: it only checks that the synthetic groups are
neither trivially separable nor indistinguishable. It is NOT the project's final implementation.
Ground-truth columns are used ONLY for scoring, never as features.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

rv = pd.read_csv("data/reviews.csv", parse_dates=["timestamp"])   # campaign_id is NOT used below
gt = pd.read_csv("data/ground_truth.csv").set_index("reviewer_id")
ids = sorted(rv["reviewer_id"].unique())
pids = sorted(rv["product_id"].unique())

# --- behavioural features per reviewer (no label columns) ---
g = rv.groupby("reviewer_id")
stat = pd.DataFrame({
    "n_reviews": g.size(), "mean_rating": g["rating"].mean(), "std_rating": g["rating"].std().fillna(0),
    "frac_high": g["rating"].apply(lambda s: (s >= 4).mean()), "frac_low": g["rating"].apply(lambda s: (s <= 2).mean()),
    "frac_verified": g["verified_purchase"].mean(),
    "mean_words": g["review_text"].apply(lambda s: s.str.split().str.len().mean()),
}).loc[ids]
# product x rating-deviation matrix (rating minus the product's mean rating)
dev = rv["rating"] - rv.groupby("product_id")["rating"].transform("mean")
pm = rv.assign(dev=dev).pivot_table(index="reviewer_id", columns="product_id", values="dev", fill_value=0.0).reindex(index=ids, columns=pids, fill_value=0.0)
# weekly activity profile
wk = ((rv["timestamp"] - rv["timestamp"].min()).dt.days // 7)
tm = pd.crosstab(rv["reviewer_id"], wk).reindex(ids, fill_value=0)
tm = (tm > 0).astype(float)

X = np.hstack([StandardScaler().fit_transform(stat), pm.to_numpy(), tm.to_numpy()])
S = np.clip(cosine_similarity(X), 0, None)
np.fill_diagonal(S, 0)
# kNN sparsification of the weighted graph
k = 15
A = np.zeros_like(S)
for i in range(len(S)):
    nn = np.argsort(S[i])[-k:]
    A[i, nn] = S[i, nn]
A = np.maximum(A, A.T)

y_cat = gt.loc[ids, "ground_truth_category"].to_numpy()
y_bin = np.where(np.isin(y_cat, ["positive_manipulation", "negative_manipulation"]), "coordinated", "other")
for k_clusters in (3, 4, 5, 6):
    labels = SpectralClustering(n_clusters=k_clusters, affinity="precomputed", random_state=42).fit_predict(A)
    print(f"k={k_clusters}: ARI(4 categories)={adjusted_rand_score(y_cat, labels):.3f} "
          f"NMI={normalized_mutual_info_score(y_cat, labels):.3f} "
          f"ARI(coordinated vs rest)={adjusted_rand_score(y_bin, labels):.3f} "
          f"silhouette={silhouette_score(X, labels):.3f}")
    if k_clusters == 4:
        print(pd.crosstab(labels, y_cat).to_string(), "\n")
