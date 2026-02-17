from __future__ import annotations
import math
import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from core.type import Post, ScoredPost, UserData


# ─── Text utilities ─────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _build_vocab(documents: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    idx = 0
    for doc in documents:
        for token in _tokenize(doc):
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


def _tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens)
    if total == 0:
        return {}
    return {t: c / total for t, c in counts.items()}


def _idf(documents: list[list[str]]) -> dict[str, float]:
    n = len(documents)
    doc_freq: dict[str, int] = {}
    for doc_tokens in documents:
        seen = set(doc_tokens)
        for t in seen:
            doc_freq[t] = doc_freq.get(t, 0) + 1
    return {t: math.log((n + 1) / (df + 1)) + 1 for t, df in doc_freq.items()}


def _tfidf_vector(tokens: list[str], idf_map: dict[str, float], vocab: dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(vocab))
    tf_map = _tf(tokens)
    for token, tf_val in tf_map.items():
        if token in vocab:
            vec[vocab[token]] = tf_val * idf_map.get(token, 1.0)
    return vec


def _cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _time_decay(post: Post, half_life_hours: float = 48.0) -> float:
    now = datetime.now(timezone.utc)
    created = post.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age_hours = max((now - created).total_seconds() / 3600.0, 0.0)
    return 2.0 ** (-age_hours / half_life_hours)


def _extract_features(
    post: Post,
    user: UserData,
    post_vec: np.ndarray,
    user_vec: np.ndarray,
) -> np.ndarray:
    """Extract the 6 scoring features for a single post."""
    relevance = _cosine_similarity(post_vec, user_vec)

    raw_eng = post.analytics.engagement_score
    engagement = raw_eng / (1.0 + abs(raw_eng))

    recency = _time_decay(post)

    user_tags = set(user.profile.interest_tags)
    post_tags = set(post.tags)
    union = user_tags | post_tags
    tag_overlap = len(user_tags & post_tags) / len(union) if union else 0.0

    network = 1.0 if post.network == "in" else 0.0

    following = set(user.socialgraph.following.chain)
    friends = set(user.socialgraph.friends)
    social = 1.0 if post.owner in (following | friends) else 0.0

    return np.array([relevance, engagement, recency, tag_overlap, network, social])


# ─── Saved model loading ────────────────────────────────────────────────────

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
_cached_model: dict | None = None


def _load_saved_model() -> dict | None:
    """Try loading pre-built model from disk. Returns None if not found."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    try:
        from build_model import load_model
        _cached_model = load_model(_MODEL_DIR)
        return _cached_model
    except (FileNotFoundError, ImportError):
        return None


# ─── Engine ──────────────────────────────────────────────────────────────────

class Engine:

    @staticmethod
    def compute_embeddings(posts: list[Post], user: UserData) -> tuple[dict[str, int], dict[str, float], list[np.ndarray]]:
        """Compute TF-IDF embeddings for all posts. Returns (vocab, idf_map, vectors)."""
        documents = []
        for p in posts:
            text = f"{p.title} {p.content} {' '.join(p.tags)}"
            documents.append(_tokenize(text))

        user_text = f"{user.profile.preferences} {' '.join(user.profile.interest_tags)}"
        documents.append(_tokenize(user_text))

        all_docs_for_vocab = [f"{p.title} {p.content} {' '.join(p.tags)}" for p in posts] + [user_text]
        vocab = _build_vocab(all_docs_for_vocab)
        idf_map = _idf(documents)

        vectors = []
        for doc_tokens in documents[:-1]:
            vectors.append(_tfidf_vector(doc_tokens, idf_map, vocab))

        return vocab, idf_map, vectors

    @staticmethod
    def compute_user_vector(user: UserData, vocab: dict[str, int], idf_map: dict[str, float]) -> np.ndarray:
        user_text = f"{user.profile.preferences} {' '.join(user.profile.interest_tags)}"
        return _tfidf_vector(_tokenize(user_text), idf_map, vocab)

    @staticmethod
    def score_relevance(post_vec, user_vec) -> float:
        return _cosine_similarity(post_vec, user_vec)

    @staticmethod
    def _score_with_saved_model(
        model: dict,
        posts: list[Post],
        user: UserData,
    ) -> list[ScoredPost]:
        """Score posts using saved model weights."""
        vocab = model["vocab"]
        idf_map = model["idf_map"]
        weights = model["weights"]
        bias = model["bias"]

        user_vec = _tfidf_vector(
            _tokenize(f"{user.profile.preferences} {' '.join(user.profile.interest_tags)}"),
            idf_map, vocab,
        )

        scored = []
        for post in posts:
            post_text = f"{post.title} {post.content} {' '.join(post.tags)}"
            post_vec = _tfidf_vector(_tokenize(post_text), idf_map, vocab)
            features = _extract_features(post, user, post_vec, user_vec)

            # dot product with learned weights
            feature_names = ["relevance", "engagement", "recency", "tag_overlap", "network", "social"]
            z = sum(features[i] * weights.get(feature_names[i], 0.0) for i in range(len(feature_names))) + bias
            score = 1.0 / (1.0 + math.exp(-max(min(z, 500), -500)))  # sigmoid

            scored.append(ScoredPost(
                post=post,
                score=score,
                source="in_network" if post.network == "in" else "out_network",
                relevance=float(features[0]),
                engagement=post.analytics.engagement_score,
                recency=float(features[2]),
            ))

        return scored

    @staticmethod
    def classify_posts(
        in_posts: list[Post],
        out_posts: list[Post],
        user: UserData,
    ) -> tuple[list[ScoredPost], list[ScoredPost]]:
        """Score and classify posts by network, relevance, engagement, and recency.
        Uses saved model weights if available, otherwise computes from scratch."""
        all_posts = in_posts + out_posts
        if not all_posts:
            return [], []

        # try saved model first
        model = _load_saved_model()
        if model is not None:
            scored = Engine._score_with_saved_model(model, all_posts, user)
            scored_in = [sp for sp in scored if sp.source == "in_network"]
            scored_out = [sp for sp in scored if sp.source == "out_network"]
            scored_in.sort(key=lambda s: s.score, reverse=True)
            scored_out.sort(key=lambda s: s.score, reverse=True)
            return scored_in, scored_out

        # fallback: compute live
        vocab, idf_map, vectors = Engine.compute_embeddings(all_posts, user)
        user_vec = Engine.compute_user_vector(user, vocab, idf_map)

        positive_post_ids = {
            r.post_id for r in user.anti_filters.engagement_history
            if r.action in ("like", "click", "share")
        }

        scored_in: list[ScoredPost] = []
        scored_out: list[ScoredPost] = []

        for i, post in enumerate(all_posts):
            relevance = Engine.score_relevance(vectors[i], user_vec)
            engagement = post.analytics.engagement_score
            recency = _time_decay(post)

            history_boost = 0.1 if post.id in positive_post_ids else 0.0

            is_in = post.network == "in"
            network_weight = 0.6 if is_in else 0.4

            user_tags = set(user.profile.interest_tags)
            post_tags = set(post.tags)
            tag_overlap = len(user_tags & post_tags) / max(len(user_tags | post_tags), 1)

            score = (
                relevance * 0.30
                + (engagement / max(engagement, 1)) * 0.25
                + recency * 0.20
                + tag_overlap * 0.15
                + history_boost * 0.05
            ) * network_weight

            sp = ScoredPost(
                post=post,
                score=score,
                source="in_network" if is_in else "out_network",
                relevance=relevance,
                engagement=engagement,
                recency=recency,
            )

            if is_in:
                scored_in.append(sp)
            else:
                scored_out.append(sp)

        scored_in.sort(key=lambda s: s.score, reverse=True)
        scored_out.sort(key=lambda s: s.score, reverse=True)
        return scored_in, scored_out

    @staticmethod
    def rank_with_diversity(scored_posts: list[ScoredPost], diversity_weight: float = 0.15) -> list[ScoredPost]:
        """Re-rank to inject diversity — penalize consecutive posts from same owner or same tags."""
        if len(scored_posts) <= 1:
            return scored_posts

        result: list[ScoredPost] = []
        remaining = list(scored_posts)
        seen_owners: dict[str, int] = {}
        seen_tags: set[str] = set()

        while remaining:
            best_idx = 0
            best_adjusted = -float("inf")

            for i, sp in enumerate(remaining):
                owner_penalty = seen_owners.get(sp.post.owner, 0) * 0.1
                tag_penalty = len(set(sp.post.tags) & seen_tags) * 0.05
                diversity_bonus = -((owner_penalty + tag_penalty) * diversity_weight)
                adjusted = sp.score + diversity_bonus
                if adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_idx = i

            chosen = remaining.pop(best_idx)
            chosen.diversity_bonus = best_adjusted - chosen.score
            result.append(chosen)
            seen_owners[chosen.post.owner] = seen_owners.get(chosen.post.owner, 0) + 1
            seen_tags.update(chosen.post.tags)

        return result

    @staticmethod
    def cold_start_fallback(posts: list[Post], k: int = 10) -> list[ScoredPost]:
        """For new users with no history: rank by raw engagement + recency."""
        scored = []
        for p in posts:
            engagement = p.analytics.engagement_score
            recency = _time_decay(p)
            score = engagement * 0.5 + recency * 0.5
            scored.append(ScoredPost(
                post=p,
                score=score,
                source="in_network" if p.network == "in" else "out_network",
                engagement=engagement,
                recency=recency,
            ))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    @staticmethod
    def reset_model_cache():
        """Clear the cached model so next call reloads from disk."""
        global _cached_model
        _cached_model = None
