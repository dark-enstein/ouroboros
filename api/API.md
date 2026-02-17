# Ouroboros Recommendation API

**Base URL:** `http://localhost:8000`

```bash
uvicorn api.app:app --reload --port 8000
```

---

## Health

### `GET /health`

Check API status and loaded resource counts.

**Response:**

```json
{
  "status": "ok",
  "posts": 15,
  "users": 1,
  "model_loaded": true
}
```

---

## Posts

### `GET /posts`

List all posts, optionally filtered by network.

| Query Param | Type   | Description                     |
|-------------|--------|---------------------------------|
| `network`   | string | Filter by `"in"` or `"out"` (optional) |

```bash
curl http://localhost:8000/posts
curl http://localhost:8000/posts?network=in
```

**Response:** `PostResponse[]`

### `GET /posts/{post_id}`

Get a single post by ID.

```bash
curl http://localhost:8000/posts/in_001
```

**Response:** `PostResponse`

### `POST /posts`

Create a new post.

**Request body — `PostCreate`:**

| Field         | Type       | Default | Description              |
|---------------|------------|---------|--------------------------|
| `title`       | string     | —       | Post title (required)    |
| `owner`       | string     | —       | Author ID (required)     |
| `content`     | string     | —       | Post body (required)     |
| `tags`        | string[]   | `[]`    | Topic tags               |
| `network`     | string     | `"out"` | `"in"` or `"out"`        |
| `attachments` | object[]   | `[]`    | File attachments         |

```bash
curl -X POST http://localhost:8000/posts \
  -H "Content-Type: application/json" \
  -d '{
    "title": "New ML Paper",
    "owner": "user_002",
    "content": "Interesting findings on transformer architectures...",
    "tags": ["machine-learning", "transformers"],
    "network": "out"
  }'
```

**Response (201):** `PostResponse`

### `DELETE /posts/{post_id}`

Delete a post by ID.

```bash
curl -X DELETE http://localhost:8000/posts/post_abc123
```

**Response:**

```json
{ "status": "deleted", "post_id": "post_abc123" }
```

---

### PostResponse schema

| Field              | Type     | Description                          |
|--------------------|----------|--------------------------------------|
| `id`               | string   | Post ID                              |
| `title`            | string   | Post title                           |
| `owner`            | string   | Author ID                            |
| `content`          | string   | Post body                            |
| `tags`             | string[] | Topic tags                           |
| `network`          | string   | `"in"` or `"out"`                    |
| `created_at`       | datetime | Creation timestamp (ISO 8601)        |
| `likes`            | int      | Like count                           |
| `dislikes`         | int      | Dislike count                        |
| `comment_count`    | int      | Number of comments                   |
| `engagement_score` | float    | Computed engagement score            |

---

## Users

### `POST /users`

Create a new user.

**Request body — `UserCreateRequest`:**

| Field       | Type                | Description                |
|-------------|---------------------|----------------------------|
| `user_id`   | string              | Unique user ID (required)  |
| `profile`   | `UserProfileRequest`| Profile data (required)    |
| `following` | string[]            | IDs of followed users      |
| `friends`   | string[]            | IDs of friends             |

**`UserProfileRequest`:**

| Field          | Type     | Default | Description            |
|----------------|----------|---------|------------------------|
| `name`         | string   | —       | Display name (required)|
| `country`      | string   | —       | Country (required)     |
| `preferences`  | string   | —       | Free-text prefs        |
| `interest_tags`| string[] | `[]`    | Interest topics        |
| `verified`     | bool     | `false` | Verification flag      |

```bash
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_042",
    "profile": {
      "name": "Ada Lovelace",
      "country": "UK",
      "preferences": "algorithms, mathematics",
      "interest_tags": ["math", "computing"],
      "verified": true
    },
    "following": ["user_001"],
    "friends": ["user_003"]
  }'
```

**Response (201):** `UserResponse`

### `GET /users/{user_id}`

Get a user by ID.

```bash
curl http://localhost:8000/users/user_001
```

**Response:** `UserResponse`

---

### UserResponse schema

| Field              | Type     | Description                          |
|--------------------|----------|--------------------------------------|
| `id`               | string   | User ID                              |
| `name`             | string   | Display name                         |
| `country`          | string   | Country                              |
| `preferences`      | string   | Free-text preferences                |
| `interest_tags`    | string[] | Interest topics                      |
| `following_count`  | int      | Number of followed users             |
| `friends`          | string[] | Friend IDs                           |
| `is_cold_start`    | bool     | `true` if no engagement history      |
| `engagement_count` | int      | Total engagement events              |
| `disliked_posts`   | string[] | IDs of disliked posts                |

---

## Feed / Recommendations

### `POST /feed`

Get a personalized feed for a user. This is the main recommendation endpoint — it runs the full pipeline: TF-IDF scoring, engagement weighting, time decay, diversity re-ranking, and two-pass filtering.

**Request body — `FeedRequest`:**

| Field            | Type   | Default | Constraints  | Description                           |
|------------------|--------|---------|--------------|---------------------------------------|
| `user_id`        | string | —       | required     | Target user                           |
| `limit`          | int    | `10`    | `1–100`      | Max posts to return                   |
| `diversity`      | float  | `0.15`  | `0.0–1.0`    | Diversity injection weight            |
| `network_filter` | string | `null`  | `"in"/"out"` | Restrict to one network (optional)    |

```bash
curl -X POST http://localhost:8000/feed \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "limit": 5,
    "diversity": 0.2
  }'
```

**Response — `FeedResponse`:**

| Field              | Type                | Description                      |
|--------------------|---------------------|----------------------------------|
| `user_id`          | string              | Requesting user                  |
| `feed`             | `RecommendedPost[]` | Ranked post list                 |
| `total_candidates` | int                 | Posts considered before filtering |
| `filtered_count`   | int                 | Posts removed by filters         |
| `model_version`    | string              | Active model version             |

**`RecommendedPost`:**

| Field             | Type     | Description                              |
|-------------------|----------|------------------------------------------|
| `rank`            | int      | Position in feed (1-indexed)             |
| `post_id`         | string   | Post ID                                  |
| `title`           | string   | Post title                               |
| `owner`           | string   | Author ID                                |
| `content`         | string   | Post body                                |
| `tags`            | string[] | Topic tags                               |
| `source`          | string   | Scoring source (`"model"` or `"live"`)   |
| `score`           | float    | Final composite score                    |
| `relevance`       | float    | TF-IDF cosine similarity component       |
| `engagement`      | float    | Engagement score component               |
| `recency`         | float    | Time decay component                     |
| `diversity_bonus` | float    | Diversity re-ranking bonus               |

---

## Engagement

### `POST /engagement`

Record user engagement events (likes, dislikes, clicks, shares, hides) and update user filters.

**Request body — `EngagementRequest`:**

| Field          | Type                | Default | Description                     |
|----------------|---------------------|---------|---------------------------------|
| `user_id`      | string              | —       | User ID (required)              |
| `events`       | `EngagementEvent[]` | —       | Engagement events (required)    |
| `block_owners` | string[]            | `[]`    | Owner IDs to block              |
| `block_tags`   | string[]            | `[]`    | Tags to block                   |

**`EngagementEvent`:**

| Field           | Type   | Default | Description                               |
|-----------------|--------|---------|-------------------------------------------|
| `post_id`       | string | —       | Post ID (required)                        |
| `action`        | string | —       | `like`, `dislike`, `click`, `share`, `hide` |
| `dwell_seconds` | float  | `0.0`   | Time spent viewing                        |

```bash
curl -X POST http://localhost:8000/engagement \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "events": [
      { "post_id": "in_001", "action": "like", "dwell_seconds": 12.5 },
      { "post_id": "out_003", "action": "dislike" }
    ],
    "block_owners": [],
    "block_tags": ["spam"]
  }'
```

**Response — `EngagementResponse`:**

| Field             | Type     | Description                     |
|-------------------|----------|---------------------------------|
| `user_id`         | string   | User ID                         |
| `events_recorded` | int      | Number of events processed      |
| `disliked_posts`  | string[] | Updated disliked post IDs       |
| `blocked_owners`  | string[] | Updated blocked owner IDs       |

---

## Model

### `GET /model`

View the current model weights and metadata.

```bash
curl http://localhost:8000/model
```

**Response — `ModelInfoResponse`:**

| Field           | Type              | Description                   |
|-----------------|-------------------|-------------------------------|
| `version`       | string            | Model version                 |
| `built_at`      | string            | Build timestamp               |
| `vocab_size`    | int               | TF-IDF vocabulary size        |
| `num_posts`     | int               | Posts used in training        |
| `feature_names` | string[]          | Scoring feature names         |
| `weights`       | dict[string,float]| Learned feature weights       |
| `bias`          | float             | Logistic regression bias term |

### `POST /model/rebuild`

Retrain the model from current post and user data, then hot-reload it.

```bash
curl -X POST http://localhost:8000/model/rebuild
```

**Response — `RebuildResponse`:**

| Field        | Type   | Description             |
|--------------|--------|-------------------------|
| `status`     | string | `"rebuilt"`             |
| `model_dir`  | string | Output directory        |
| `num_posts`  | int    | Posts in training set   |
| `vocab_size` | int    | Vocabulary size         |
| `built_at`   | string | Build timestamp         |

---

## Error Responses

All errors follow this shape:

```json
{ "detail": "Post post_999 not found" }
```

| Status | Meaning                              |
|--------|--------------------------------------|
| 404    | Resource not found                   |
| 409    | Conflict (e.g. duplicate user ID)    |
| 422    | Validation error (bad request body)  |

---

## Interactive Docs

FastAPI auto-generates interactive documentation:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
