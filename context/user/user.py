from __future__ import annotations
from datetime import datetime, timezone
from core.db import get_user
from core.type import UserData, Post, ScoredPost, EngagementRecord
from core.models import Engine


class User:
    def __init__(self):
        self.tree: UserData = get_user()

    @property
    def is_cold_start(self) -> bool:
        """True if user has no engagement history — trigger cold-start fallback."""
        return len(self.tree.anti_filters.engagement_history) == 0

    def embedding(self) -> UserData:
        """Return the full user data tree (profile + graph + filters)."""
        return self.tree

    def get_profile_embedding(self):
        return self.tree.profile

    def get_real_graph(self):
        return self.tree.socialgraph

    def get_anti_signals(self):
        return self.tree.anti_filters

    def get_following_owners(self) -> set[str]:
        return set(self.tree.socialgraph.following.chain)

    def get_friends(self) -> set[str]:
        return set(self.tree.socialgraph.friends)

    def filter(self, scored_posts: list[ScoredPost]) -> list[ScoredPost]:
        """Pre-filter: remove posts from blocked owners, with blocked tags,
        or that the user has already disliked."""
        blocked_owners = set(self.tree.anti_filters.blocked_owners)
        blocked_tags = set(self.tree.anti_filters.blocked_tags)
        disliked = set(self.tree.anti_filters.disliked_posts)

        result = []
        for sp in scored_posts:
            post = sp.post
            if post.owner in blocked_owners:
                continue
            if post.id in disliked:
                continue
            if blocked_tags & set(post.tags):
                continue
            result.append(sp)
        return result

    def record_engagement(self, post_id: str, action: str, dwell_seconds: float = 0.0):
        """Record a new engagement event."""
        record = EngagementRecord(
            post_id=post_id,
            action=action,
            timestamp=datetime.now(timezone.utc),
            dwell_seconds=dwell_seconds,
        )
        self.tree.anti_filters.engagement_history.append(record)

        if action == "dislike":
            if post_id not in self.tree.anti_filters.disliked_posts:
                self.tree.anti_filters.disliked_posts.append(post_id)

    def update_filters(self, engagement_params: dict | None):
        """Update user filters from engagement feedback in real-time."""
        if not engagement_params:
            return

        for event in engagement_params.get("events", []):
            self.record_engagement(
                post_id=event.get("post_id", ""),
                action=event.get("action", "click"),
                dwell_seconds=event.get("dwell_seconds", 0.0),
            )

        # update blocked owners if user explicitly hides
        for owner in engagement_params.get("block_owners", []):
            if owner not in self.tree.anti_filters.blocked_owners:
                self.tree.anti_filters.blocked_owners.append(owner)

        # update blocked tags
        for tag in engagement_params.get("block_tags", []):
            if tag not in self.tree.anti_filters.blocked_tags:
                self.tree.anti_filters.blocked_tags.append(tag)
