from __future__ import annotations
import math
import random
from collections import Counter
from datetime import datetime, timezone
from core.type import Post, ScoredPost, UserData


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


def _tfidf_vector(tokens: list[str], idf_map: dict[str, float], vocab: dict[str, int]) -> list[float]:
    vec = [0.0] * len(vocab)
    tf_map = _tf(tokens)
    for token, tf_val in tf_map.items():
        if token in vocab:
            vec[vocab[token]] = tf_val * idf_map.get(token, 1.0)
    return vec


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _time_decay(post: Post, half_life_hours: float = 48.0) -> float:
    now = datetime.now(timezone.utc)
    created = post.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age_hours = (now - created).total_seconds() / 3600.0
    return 2.0 ** (-age_hours / half_life_hours)


class Engine:

    @staticmethod
    def compute_embeddings(posts: list[Post], user: UserData) -> tuple[dict[str, int], dict[str, float], list[list[float]]]:
        """Compute TF-IDF embeddings for all posts. Returns (vocab, idf_map, vectors)."""
        documents = []
        for p in posts:
            text = f"{p.title} {p.content} {' '.join(p.tags)}"
            documents.append(_tokenize(text))

        # include user interest text in vocab building
        user_text = f"{user.profile.preferences} {' '.join(user.profile.interest_tags)}"
        documents.append(_tokenize(user_text))

        all_docs_for_vocab = [f"{p.title} {p.content} {' '.join(p.tags)}" for p in posts] + [user_text]
        vocab = _build_vocab(all_docs_for_vocab)
        idf_map = _idf(documents)

        vectors = []
        for doc_tokens in documents[:-1]:  # exclude user doc
            vectors.append(_tfidf_vector(doc_tokens, idf_map, vocab))

        return vocab, idf_map, vectors

    @staticmethod
    def compute_user_vector(user: UserData, vocab: dict[str, int], idf_map: dict[str, float]) -> list[float]:
        user_text = f"{user.profile.preferences} {' '.join(user.profile.interest_tags)}"
        return _tfidf_vector(_tokenize(user_text), idf_map, vocab)

    @staticmethod
    def score_relevance(post_vec: list[float], user_vec: list[float]) -> float:
        return _cosine_similarity(post_vec, user_vec)

    @staticmethod
    def classify_posts(
        in_posts: list[Post],
        out_posts: list[Post],
        user: UserData,
    ) -> tuple[list[ScoredPost], list[ScoredPost]]:
        """Score and classify posts by network, relevance, engagement, and recency."""
        all_posts = in_posts + out_posts
        if not all_posts:
            return [], []

        vocab, idf_map, vectors = Engine.compute_embeddings(all_posts, user)
        user_vec = Engine.compute_user_vector(user, vocab, idf_map)

        # engagement from history: posts the user liked/clicked get a boost
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

            # personal history boost
            history_boost = 0.1 if post.id in positive_post_ids else 0.0

            # network weight
            is_in = post.network == "in"
            network_weight = 0.6 if is_in else 0.4

            # tag overlap boost
            user_tags = set(user.profile.interest_tags)
            post_tags = set(post.tags)
            tag_overlap = len(user_tags & post_tags) / max(len(user_tags | post_tags), 1)

            # composite score
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
