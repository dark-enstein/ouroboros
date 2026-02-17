from __future__ import annotations
from core.type import ScoredPost, UserData


BLOCKED_KEYWORDS = ["spam", "scam", "nsfw", "phishing"]


def rank(scored_posts: list[ScoredPost]) -> list[ScoredPost]:
    """Final engagement-based ranking. Posts are already scored; this applies
    a final sort and normalizes scores to [0, 1]."""
    if not scored_posts:
        return []

    sorted_posts = sorted(scored_posts, key=lambda s: s.score, reverse=True)

    max_score = sorted_posts[0].score if sorted_posts[0].score > 0 else 1.0
    for sp in sorted_posts:
        sp.score = sp.score / max_score

    return sorted_posts


def filter(scored_posts: list[ScoredPost], user: UserData | None = None) -> list[ScoredPost]:
    """Post-filtering: social guardrails, PG restrictions, duplicate removal."""
    result = []
    seen_ids: set[str] = set()

    blocked_owners: set[str] = set()
    blocked_tags: set[str] = set()
    disliked_ids: set[str] = set()

    if user:
        blocked_owners = set(user.anti_filters.blocked_owners)
        blocked_tags = set(user.anti_filters.blocked_tags)
        disliked_ids = set(user.anti_filters.disliked_posts)

    for sp in scored_posts:
        post = sp.post

        # deduplicate
        if post.id in seen_ids:
            continue
        seen_ids.add(post.id)

        # blocked owners
        if post.owner in blocked_owners:
            continue

        # disliked posts
        if post.id in disliked_ids:
            continue

        # blocked tags
        if blocked_tags & set(post.tags):
            continue

        # content-level keyword filter (PG / social norms)
        text = f"{post.title} {post.content}".lower()
        if any(kw in text for kw in BLOCKED_KEYWORDS):
            continue

        # high-dislike ratio filter: if >40% reactions are dislikes, suppress
        total_reactions = post.analytics.reactions.likes + post.analytics.reactions.dislikes
        if total_reactions > 5 and post.analytics.reactions.dislikes / total_reactions > 0.4:
            continue

        result.append(sp)

    return result
