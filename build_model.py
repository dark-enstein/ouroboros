"""
build_model.py — Train and save the recommendation model artifacts.

Builds:
  1. TF-IDF index   (vocab, IDF weights, post vectors)
  2. Scoring weights (learned from engagement history via logistic regression)
  3. Post metadata   (id-to-index mapping for fast lookup)

Saves everything to model/ as pickle + JSON for portability.

Usage:
    python build_model.py              # build with defaults
    python build_model.py --out model  # custom output directory
"""

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.db import setup_db, get_user
from core.type import Post, UserData


# ─── TF-IDF ────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_vocab(documents: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    idx = 0
    for doc in documents:
        for token in tokenize(doc):
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def compute_idf(tokenized_docs: list[list[str]]) -> dict[str, float]:
    n = len(tokenized_docs)
    doc_freq: dict[str, int] = {}
    for tokens in tokenized_docs:
        for t in set(tokens):
            doc_freq[t] = doc_freq.get(t, 0) + 1
    return {t: math.log((n + 1) / (df + 1)) + 1 for t, df in doc_freq.items()}


def tfidf_vector(tokens: list[str], idf_map: dict[str, float], vocab: dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(vocab))
    counts = Counter(tokens)
    total = len(tokens) or 1
    for token, count in counts.items():
        if token in vocab:
            tf = count / total
            vec[vocab[token]] = tf * idf_map.get(token, 1.0)
    return vec


def post_to_text(post: Post) -> str:
    return f"{post.title} {post.content} {' '.join(post.tags)}"


# ─── Feature extraction ────────────────────────────────────────────────────

def extract_features(post: Post, user: UserData, post_vec: np.ndarray, user_vec: np.ndarray) -> np.ndarray:
    """Extract the 6 scoring features for a single post."""
    # 1. cosine relevance
    dot = float(np.dot(post_vec, user_vec))
    norm_p = float(np.linalg.norm(post_vec))
    norm_u = float(np.linalg.norm(user_vec))
    relevance = dot / (norm_p * norm_u) if (norm_p > 0 and norm_u > 0) else 0.0

    # 2. engagement (normalized to [0, 1] range via sigmoid-like)
    raw_eng = post.analytics.engagement_score
    engagement = raw_eng / (1.0 + abs(raw_eng))

    # 3. recency (time decay, half-life 48h)
    now = datetime.now(timezone.utc)
    created = post.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age_hours = max((now - created).total_seconds() / 3600.0, 0.0)
    recency = 2.0 ** (-age_hours / 48.0)

    # 4. tag overlap (jaccard)
    user_tags = set(user.profile.interest_tags)
    post_tags = set(post.tags)
    union = user_tags | post_tags
    tag_overlap = len(user_tags & post_tags) / len(union) if union else 0.0

    # 5. network weight
    network = 1.0 if post.network == "in" else 0.0

    # 6. owner in social graph
    following = set(user.socialgraph.following.chain)
    friends = set(user.socialgraph.friends)
    social = 1.0 if post.owner in (following | friends) else 0.0

    return np.array([relevance, engagement, recency, tag_overlap, network, social])


# ─── Scoring weight learning ───────────────────────────────────────────────

def learn_weights(
    features: np.ndarray,
    labels: np.ndarray,
    lr: float = 0.1,
    epochs: int = 500,
) -> np.ndarray:
    """Learn scoring weights via logistic regression (gradient descent).
    features: (n_samples, n_features), labels: (n_samples,) in {0, 1}."""
    n_features = features.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(epochs):
        z = features @ weights + bias
        preds = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))  # sigmoid
        error = preds - labels
        grad_w = (features.T @ error) / len(labels)
        grad_b = error.mean()
        weights -= lr * grad_w
        bias -= lr * grad_b

    return np.append(weights, bias)


def build_training_data(
    posts: list[Post],
    user: UserData,
    post_vectors: list[np.ndarray],
    user_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build training matrix from engagement history.
    Positive: posts the user liked/clicked/shared.
    Negative: posts the user disliked or never engaged with."""
    positive_ids = {
        r.post_id for r in user.anti_filters.engagement_history
        if r.action in ("like", "click", "share")
    }
    negative_ids = {
        r.post_id for r in user.anti_filters.engagement_history
        if r.action in ("dislike", "hide")
    }

    post_id_to_idx = {p.id: i for i, p in enumerate(posts)}
    features_list = []
    labels_list = []

    for post in posts:
        idx = post_id_to_idx[post.id]
        feat = extract_features(post, user, post_vectors[idx], user_vec)

        if post.id in positive_ids:
            features_list.append(feat)
            labels_list.append(1.0)
        elif post.id in negative_ids:
            features_list.append(feat)
            labels_list.append(0.0)
        else:
            # implicit negative (not engaged) — downweight
            features_list.append(feat)
            labels_list.append(0.2)

    return np.array(features_list), np.array(labels_list)


# ─── Build & Save ──────────────────────────────────────────────────────────

def build(output_dir: str = "model"):
    print("=" * 60)
    print("  Ouroboros — Model Build")
    print("=" * 60)

    # ── 1. Load data ──
    print("\n[1/6] Loading posts and user data...")
    db = setup_db()
    in_posts, out_posts = db.retrieve_by_network()
    all_posts = in_posts + out_posts
    user = get_user()
    print(f"  Posts: {len(in_posts)} in-network + {len(out_posts)} out-network = {len(all_posts)} total")
    print(f"  User: {user.profile.name} ({user.id})")

    # ── 2. Build TF-IDF index ──
    print("\n[2/6] Building TF-IDF index...")
    doc_texts = [post_to_text(p) for p in all_posts]
    user_text = f"{user.profile.preferences} {' '.join(user.profile.interest_tags)}"
    all_texts = doc_texts + [user_text]

    vocab = build_vocab(all_texts)
    tokenized_docs = [tokenize(t) for t in all_texts]
    idf_map = compute_idf(tokenized_docs)

    post_vectors = [tfidf_vector(tokenize(t), idf_map, vocab) for t in doc_texts]
    user_vec = tfidf_vector(tokenize(user_text), idf_map, vocab)

    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Vector dimensions: {len(vocab)}")
    print(f"  Post vectors: {len(post_vectors)}")

    # ── 3. Extract features ──
    print("\n[3/6] Extracting scoring features...")
    feature_names = ["relevance", "engagement", "recency", "tag_overlap", "network", "social"]
    X, y = build_training_data(all_posts, user, post_vectors, user_vec)
    print(f"  Training samples: {len(y)}")
    print(f"  Positive (engaged): {int((y > 0.5).sum())}")
    print(f"  Negative/implicit: {int((y <= 0.5).sum())}")

    # ── 4. Learn scoring weights ──
    print("\n[4/6] Learning scoring weights...")
    weights = learn_weights(X, y, lr=0.1, epochs=1000)
    feature_weights = dict(zip(feature_names, weights[:-1]))
    bias = weights[-1]
    print(f"  Learned weights:")
    for name, w in feature_weights.items():
        print(f"    {name:<15} {w:+.4f}")
    print(f"    {'bias':<15} {bias:+.4f}")

    # ── 5. Build post index ──
    print("\n[5/6] Building post index...")
    post_index = {}
    for i, post in enumerate(all_posts):
        post_index[post.id] = {
            "idx": i,
            "title": post.title,
            "owner": post.owner,
            "network": post.network,
            "tags": post.tags,
            "created_at": post.created_at.isoformat(),
        }
    print(f"  Indexed {len(post_index)} posts")

    # ── 6. Save artifacts ──
    print(f"\n[6/6] Saving model to {output_dir}/...")

    # remove old empty 'model' file if it exists and is not a directory
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        os.remove(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # TF-IDF artifacts (pickle — internal use)
    tfidf_path = os.path.join(output_dir, "tfidf.pkl")
    with open(tfidf_path, "wb") as f:
        pickle.dump({
            "vocab": vocab,
            "idf_map": idf_map,
            "post_vectors": [v.tolist() for v in post_vectors],
            "user_vector": user_vec.tolist(),
        }, f)
    print(f"  {tfidf_path} ({os.path.getsize(tfidf_path):,} bytes)")

    # scoring weights (JSON — human readable)
    weights_path = os.path.join(output_dir, "weights.json")
    with open(weights_path, "w") as f:
        json.dump({
            "feature_names": feature_names,
            "weights": {name: float(w) for name, w in feature_weights.items()},
            "bias": float(bias),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_samples": len(y),
            "epochs": 1000,
        }, f, indent=2)
    print(f"  {weights_path} ({os.path.getsize(weights_path):,} bytes)")

    # post index (JSON)
    index_path = os.path.join(output_dir, "post_index.json")
    with open(index_path, "w") as f:
        json.dump(post_index, f, indent=2)
    print(f"  {index_path} ({os.path.getsize(index_path):,} bytes)")

    # metadata
    meta_path = os.path.join(output_dir, "meta.json")
    meta = {
        "version": "1.0.0",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "vocab_size": len(vocab),
        "num_posts": len(all_posts),
        "num_in_network": len(in_posts),
        "num_out_network": len(out_posts),
        "user_id": user.id,
        "feature_names": feature_names,
        "artifacts": ["tfidf.pkl", "weights.json", "post_index.json", "meta.json"],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {meta_path} ({os.path.getsize(meta_path):,} bytes)")

    print("\n" + "=" * 60)
    print("  Build complete.")
    print("=" * 60)
    return output_dir


# ─── Load (used by Engine at inference time) ────────────────────────────────

def load_model(model_dir: str = "model") -> dict:
    """Load saved model artifacts from disk."""
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No model found at {model_dir}/meta.json")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    with open(os.path.join(model_dir, "tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)

    with open(os.path.join(model_dir, "weights.json"), "r") as f:
        weights = json.load(f)

    with open(os.path.join(model_dir, "post_index.json"), "r") as f:
        post_index = json.load(f)

    return {
        "meta": meta,
        "vocab": tfidf["vocab"],
        "idf_map": tfidf["idf_map"],
        "post_vectors": [np.array(v) for v in tfidf["post_vectors"]],
        "user_vector": np.array(tfidf["user_vector"]),
        "weights": {k: float(v) for k, v in weights["weights"].items()},
        "bias": float(weights["bias"]),
        "feature_names": weights["feature_names"],
        "post_index": post_index,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and save the recommendation model")
    parser.add_argument("--out", default="model", help="Output directory (default: model/)")
    args = parser.parse_args()
    build(args.out)
