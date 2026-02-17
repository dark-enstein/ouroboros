from __future__ import annotations
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class Attachment(BaseModel):
    id: str
    signed_link: str


class Reactions(BaseModel):
    data: dict = Field(default_factory=lambda: {"like": 0, "dislike": 0})

    @property
    def likes(self) -> int:
        return self.data.get("like", 0)

    @property
    def dislikes(self) -> int:
        return self.data.get("dislike", 0)

    @property
    def net_score(self) -> int:
        return self.likes - self.dislikes


class Comment(BaseModel):
    id: str
    owner: str
    content: str
    reactions: Reactions = Field(default_factory=Reactions)


class Analytics(BaseModel):
    reactions: Reactions = Field(default_factory=Reactions)
    comments: list[Comment] = []

    @property
    def engagement_score(self) -> float:
        likes = self.reactions.likes
        dislikes = self.reactions.dislikes
        comment_count = len(self.comments)
        comment_reactions = sum(c.reactions.net_score for c in self.comments)
        return likes - (dislikes * 1.5) + (comment_count * 2.0) + (comment_reactions * 0.5)


class Post(BaseModel):
    id: str
    title: str
    owner: str
    content: str
    attachments: list[Attachment] = []
    analytics: Analytics = Field(default_factory=Analytics)
    created_at: datetime = Field(default_factory=datetime.now)
    tags: list[str] = []
    network: str = "out"  # "in" or "out"
    _embedding: Optional[list[float]] = None
    _score: float = 0.0

    def hydrate(self) -> Post:
        """Return self with full data loaded (already hydrated from JSON)."""
        return self


class Profile(BaseModel):
    name: str
    country: str
    preferences: str
    photo: list[Attachment] = []
    verified: bool = False
    interest_tags: list[str] = []


class Follows(BaseModel):
    len: int = 0
    chain: list[str] = []  # list of user IDs


class SocialGraph(BaseModel):
    followers: Follows = Field(default_factory=Follows)
    following: Follows = Field(default_factory=Follows)
    friends: list[str] = []  # list of user IDs


class EngagementRecord(BaseModel):
    post_id: str
    action: str  # "like", "dislike", "click", "dwell", "share", "hide"
    timestamp: datetime = Field(default_factory=datetime.now)
    dwell_seconds: float = 0.0


class Filters(BaseModel):
    disliked_posts: list[str] = []  # post IDs
    blocked_owners: list[str] = []
    blocked_tags: list[str] = []
    engagement_history: list[EngagementRecord] = []


class UserData(BaseModel):
    id: str
    profile: Profile
    socialgraph: SocialGraph
    anti_filters: Filters = Field(default_factory=Filters)


class ScoredPost(BaseModel):
    post: Post
    score: float = 0.0
    source: str = "out_network"  # "in_network" or "out_network"
    relevance: float = 0.0
    engagement: float = 0.0
    recency: float = 0.0
    diversity_bonus: float = 0.0


def get_current_user(raw: dict) -> UserData:
    return UserData.model_validate(raw)


def get_current_posts(raw: list | dict) -> list[Post]:
    if isinstance(raw, dict):
        for key in raw:
            if isinstance(raw[key], list):
                return [Post.model_validate(p) for p in raw[key]]
    if isinstance(raw, list):
        return [Post.model_validate(p) for p in raw]
    return []
