from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field


# ─── Posts ───────────────────────────────────────────────────────────────────

class PostCreate(BaseModel):
    title: str
    owner: str
    content: str
    tags: list[str] = []
    network: str = "out"
    attachments: list[dict] = []


class PostResponse(BaseModel):
    id: str
    title: str
    owner: str
    content: str
    tags: list[str]
    network: str
    created_at: datetime
    likes: int
    dislikes: int
    comment_count: int
    engagement_score: float


# ─── Feed / Recommendations ─────────────────────────────────────────────────

class FeedRequest(BaseModel):
    user_id: str
    limit: int = Field(default=10, ge=1, le=100)
    diversity: float = Field(default=0.15, ge=0.0, le=1.0)
    network_filter: str | None = None  # "in", "out", or None for both


class RecommendedPost(BaseModel):
    rank: int
    post_id: str
    title: str
    owner: str
    content: str
    tags: list[str]
    source: str
    score: float
    relevance: float
    engagement: float
    recency: float
    diversity_bonus: float


class FeedResponse(BaseModel):
    user_id: str
    feed: list[RecommendedPost]
    total_candidates: int
    filtered_count: int
    model_version: str


# ─── Engagement ──────────────────────────────────────────────────────────────

class EngagementEvent(BaseModel):
    post_id: str
    action: str  # like, dislike, click, share, hide
    dwell_seconds: float = 0.0


class EngagementRequest(BaseModel):
    user_id: str
    events: list[EngagementEvent]
    block_owners: list[str] = []
    block_tags: list[str] = []


class EngagementResponse(BaseModel):
    user_id: str
    events_recorded: int
    disliked_posts: list[str]
    blocked_owners: list[str]


# ─── Model ───────────────────────────────────────────────────────────────────

class ModelInfoResponse(BaseModel):
    version: str
    built_at: str
    vocab_size: int
    num_posts: int
    feature_names: list[str]
    weights: dict[str, float]
    bias: float


class RebuildResponse(BaseModel):
    status: str
    model_dir: str
    num_posts: int
    vocab_size: int
    built_at: str


# ─── User ────────────────────────────────────────────────────────────────────

class UserProfileRequest(BaseModel):
    name: str
    country: str
    preferences: str
    interest_tags: list[str] = []
    verified: bool = False


class UserCreateRequest(BaseModel):
    user_id: str
    profile: UserProfileRequest
    following: list[str] = []
    friends: list[str] = []


class UserResponse(BaseModel):
    id: str
    name: str
    country: str
    preferences: str
    interest_tags: list[str]
    following_count: int
    friends: list[str]
    is_cold_start: bool
    engagement_count: int
    disliked_posts: list[str]
