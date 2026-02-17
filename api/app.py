"""
Ouroboros Recommendation API

Usage:
    uvicorn api.app:app --reload --port 8000
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager

from api.state import app_state
from api.schemas import (
    PostCreate, PostResponse,
    FeedRequest, FeedResponse, RecommendedPost,
    EngagementRequest, EngagementResponse,
    ModelInfoResponse, RebuildResponse,
    UserCreateRequest, UserResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state.initialize()
    yield


app = FastAPI(
    title="Ouroboros",
    description="Recommendation engine API — personalized feeds with TF-IDF relevance, engagement scoring, time decay, and diversity injection.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "posts": len(app_state.posts),
        "users": len(app_state.users),
        "model_loaded": app_state.model is not None,
    }


# ─── Posts ───────────────────────────────────────────────────────────────────

def _post_to_response(p) -> PostResponse:
    return PostResponse(
        id=p.id,
        title=p.title,
        owner=p.owner,
        content=p.content,
        tags=p.tags,
        network=p.network,
        created_at=p.created_at,
        likes=p.analytics.reactions.likes,
        dislikes=p.analytics.reactions.dislikes,
        comment_count=len(p.analytics.comments),
        engagement_score=p.analytics.engagement_score,
    )


@app.get("/posts", response_model=list[PostResponse])
def list_posts(network: str | None = Query(None, description="Filter by 'in' or 'out'")):
    posts = app_state.list_posts(network=network)
    return [_post_to_response(p) for p in posts]


@app.get("/posts/{post_id}", response_model=PostResponse)
def get_post(post_id: str):
    post = app_state.get_post(post_id)
    if post is None:
        raise HTTPException(status_code=404, detail=f"Post {post_id} not found")
    return _post_to_response(post)


@app.post("/posts", response_model=PostResponse, status_code=201)
def create_post(body: PostCreate):
    post = app_state.add_post(
        title=body.title,
        owner=body.owner,
        content=body.content,
        tags=body.tags,
        network=body.network,
        attachments=body.attachments,
    )
    return _post_to_response(post)


@app.delete("/posts/{post_id}")
def delete_post(post_id: str):
    if not app_state.delete_post(post_id):
        raise HTTPException(status_code=404, detail=f"Post {post_id} not found")
    return {"status": "deleted", "post_id": post_id}


# ─── Users ───────────────────────────────────────────────────────────────────

def _user_to_response(u) -> UserResponse:
    return UserResponse(
        id=u.id,
        name=u.profile.name,
        country=u.profile.country,
        preferences=u.profile.preferences,
        interest_tags=u.profile.interest_tags,
        following_count=u.socialgraph.following.len,
        friends=u.socialgraph.friends,
        is_cold_start=len(u.anti_filters.engagement_history) == 0,
        engagement_count=len(u.anti_filters.engagement_history),
        disliked_posts=u.anti_filters.disliked_posts,
    )


@app.post("/users", response_model=UserResponse, status_code=201)
def create_user(body: UserCreateRequest):
    if app_state.get_user(body.user_id):
        raise HTTPException(status_code=409, detail=f"User {body.user_id} already exists")
    user = app_state.create_user(
        user_id=body.user_id,
        name=body.profile.name,
        country=body.profile.country,
        preferences=body.profile.preferences,
        interest_tags=body.profile.interest_tags,
        verified=body.profile.verified,
        following=body.following,
        friends=body.friends,
    )
    return _user_to_response(user)


@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: str):
    user = app_state.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return _user_to_response(user)


# ─── Feed / Recommendations ─────────────────────────────────────────────────

@app.post("/feed", response_model=FeedResponse)
def get_feed(body: FeedRequest):
    user = app_state.get_user(body.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User {body.user_id} not found")

    feed, total, filtered = app_state.get_feed(
        user_id=body.user_id,
        limit=body.limit,
        diversity=body.diversity,
        network_filter=body.network_filter,
    )

    model_version = app_state.model_meta.get("version", "live") if app_state.model else "live"

    recommended = []
    for i, sp in enumerate(feed, 1):
        recommended.append(RecommendedPost(
            rank=i,
            post_id=sp.post.id,
            title=sp.post.title,
            owner=sp.post.owner,
            content=sp.post.content,
            tags=sp.post.tags,
            source=sp.source,
            score=round(sp.score, 4),
            relevance=round(sp.relevance, 4),
            engagement=round(sp.engagement, 2),
            recency=round(sp.recency, 4),
            diversity_bonus=round(sp.diversity_bonus, 4),
        ))

    return FeedResponse(
        user_id=body.user_id,
        feed=recommended,
        total_candidates=total,
        filtered_count=filtered,
        model_version=model_version,
    )


# ─── Engagement ──────────────────────────────────────────────────────────────

@app.post("/engagement", response_model=EngagementResponse)
def record_engagement(body: EngagementRequest):
    user = app_state.get_user(body.user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User {body.user_id} not found")

    events = [{"post_id": e.post_id, "action": e.action, "dwell_seconds": e.dwell_seconds}
              for e in body.events]
    count = app_state.record_engagement(
        user_id=body.user_id,
        events=events,
        block_owners=body.block_owners,
        block_tags=body.block_tags,
    )

    return EngagementResponse(
        user_id=body.user_id,
        events_recorded=count,
        disliked_posts=user.anti_filters.disliked_posts,
        blocked_owners=user.anti_filters.blocked_owners,
    )


# ─── Model ───────────────────────────────────────────────────────────────────

@app.get("/model", response_model=ModelInfoResponse)
def model_info():
    if app_state.model is None:
        raise HTTPException(status_code=404, detail="No model built yet. POST /model/rebuild to build one.")
    return ModelInfoResponse(
        version=app_state.model_meta.get("version", "unknown"),
        built_at=app_state.model_meta.get("built_at", "unknown"),
        vocab_size=app_state.model_meta.get("vocab_size", 0),
        num_posts=app_state.model_meta.get("num_posts", 0),
        feature_names=app_state.model.get("feature_names", []),
        weights=app_state.model.get("weights", {}),
        bias=app_state.model.get("bias", 0.0),
    )


@app.post("/model/rebuild", response_model=RebuildResponse)
def rebuild_model():
    meta = app_state.rebuild_model()
    return RebuildResponse(
        status="rebuilt",
        model_dir="model",
        num_posts=meta.get("num_posts", 0),
        vocab_size=meta.get("vocab_size", 0),
        built_at=meta.get("built_at", "unknown"),
    )
