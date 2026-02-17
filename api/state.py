"""
Application state — singleton that holds the post database, user store,
and loaded model in memory. Initialized at FastAPI startup.
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone

from core.type import (
    Post, UserData, Profile, SocialGraph, Follows,
    Filters, EngagementRecord, Analytics, Reactions, ScoredPost,
)
from core.db import setup_db, get_user
from core.models import Engine
from core.utils import rank, filter as post_filter
from build_model import load_model, build

MODEL_DIR = "model"


class AppState:
    def __init__(self):
        self.posts: dict[str, Post] = {}
        self.users: dict[str, UserData] = {}
        self.model: dict | None = None
        self.model_meta: dict = {}

    def initialize(self):
        """Load posts from disk, seed the default user, load saved model."""
        db = setup_db()
        in_posts, out_posts = db.retrieve_by_network()
        for p in in_posts + out_posts:
            self.posts[p.id] = p

        default_user = get_user()
        self.users[default_user.id] = default_user

        try:
            self.model = load_model(MODEL_DIR)
            self.model_meta = self.model.get("meta", {})
        except FileNotFoundError:
            self.model = None
            self.model_meta = {}

    # ─── Posts ───────────────────────────────────────────────────────────────

    def add_post(self, title: str, owner: str, content: str,
                 tags: list[str], network: str, attachments: list[dict]) -> Post:
        post_id = f"post_{uuid.uuid4().hex[:8]}"
        post = Post(
            id=post_id,
            title=title,
            owner=owner,
            content=content,
            tags=tags,
            network=network,
            created_at=datetime.now(timezone.utc),
            attachments=[],
        )
        self.posts[post_id] = post
        return post

    def get_post(self, post_id: str) -> Post | None:
        return self.posts.get(post_id)

    def delete_post(self, post_id: str) -> bool:
        if post_id in self.posts:
            del self.posts[post_id]
            return True
        return False

    def list_posts(self, network: str | None = None) -> list[Post]:
        posts = list(self.posts.values())
        if network:
            posts = [p for p in posts if p.network == network]
        return sorted(posts, key=lambda p: p.created_at, reverse=True)

    # ─── Users ───────────────────────────────────────────────────────────────

    def create_user(self, user_id: str, name: str, country: str,
                    preferences: str, interest_tags: list[str],
                    verified: bool, following: list[str],
                    friends: list[str]) -> UserData:
        user = UserData(
            id=user_id,
            profile=Profile(
                name=name,
                country=country,
                preferences=preferences,
                interest_tags=interest_tags,
                verified=verified,
            ),
            socialgraph=SocialGraph(
                followers=Follows(len=0, chain=[]),
                following=Follows(len=len(following), chain=following),
                friends=friends,
            ),
            anti_filters=Filters(),
        )
        self.users[user_id] = user
        return user

    def get_user(self, user_id: str) -> UserData | None:
        return self.users.get(user_id)

    # ─── Recommendations ─────────────────────────────────────────────────────

    def get_feed(self, user_id: str, limit: int = 10,
                 diversity: float = 0.15,
                 network_filter: str | None = None) -> tuple[list[ScoredPost], int, int]:
        """Run the full recommendation pipeline. Returns (feed, total_candidates, filtered_count)."""
        user = self.users.get(user_id)
        if user is None:
            return [], 0, 0

        all_posts = list(self.posts.values())
        if network_filter:
            all_posts = [p for p in all_posts if p.network == network_filter]

        total_candidates = len(all_posts)
        if total_candidates == 0:
            return [], 0, 0

        in_posts = [p for p in all_posts if p.network == "in"]
        out_posts = [p for p in all_posts if p.network == "out"]

        # cold-start check
        is_cold = len(user.anti_filters.engagement_history) == 0
        if is_cold:
            scored = Engine.cold_start_fallback(all_posts, k=limit)
        else:
            scored_in, scored_out = Engine.classify_posts(in_posts, out_posts, user)
            scored = scored_in + scored_out

        # pre-filter (user guardrails)
        blocked_owners = set(user.anti_filters.blocked_owners)
        blocked_tags = set(user.anti_filters.blocked_tags)
        disliked = set(user.anti_filters.disliked_posts)
        prefiltered = [
            sp for sp in scored
            if sp.post.owner not in blocked_owners
            and sp.post.id not in disliked
            and not (blocked_tags & set(sp.post.tags))
        ]

        # rank + diversity
        ranked = rank(prefiltered)
        diverse = Engine.rank_with_diversity(ranked, diversity_weight=diversity)

        # post-filter (social guardrails)
        final = post_filter(diverse, user)

        filtered_count = total_candidates - len(final)
        return final[:limit], total_candidates, filtered_count

    # ─── Engagement ──────────────────────────────────────────────────────────

    def record_engagement(self, user_id: str, events: list[dict],
                          block_owners: list[str],
                          block_tags: list[str]) -> int:
        user = self.users.get(user_id)
        if user is None:
            return 0

        count = 0
        for event in events:
            record = EngagementRecord(
                post_id=event["post_id"],
                action=event["action"],
                timestamp=datetime.now(timezone.utc),
                dwell_seconds=event.get("dwell_seconds", 0.0),
            )
            user.anti_filters.engagement_history.append(record)
            count += 1

            if event["action"] == "dislike":
                if event["post_id"] not in user.anti_filters.disliked_posts:
                    user.anti_filters.disliked_posts.append(event["post_id"])

            # update post analytics for likes/dislikes
            post = self.posts.get(event["post_id"])
            if post:
                if event["action"] == "like":
                    post.analytics.reactions.data["like"] = post.analytics.reactions.likes + 1
                elif event["action"] == "dislike":
                    post.analytics.reactions.data["dislike"] = post.analytics.reactions.dislikes + 1

        for owner in block_owners:
            if owner not in user.anti_filters.blocked_owners:
                user.anti_filters.blocked_owners.append(owner)

        for tag in block_tags:
            if tag not in user.anti_filters.blocked_tags:
                user.anti_filters.blocked_tags.append(tag)

        return count

    # ─── Model ───────────────────────────────────────────────────────────────

    def rebuild_model(self) -> dict:
        """Rebuild the model from current post + user data and reload."""
        output_dir = build(MODEL_DIR)
        Engine.reset_model_cache()
        self.model = load_model(output_dir)
        self.model_meta = self.model.get("meta", {})
        return self.model_meta


# singleton
app_state = AppState()
