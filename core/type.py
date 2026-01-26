from typing import List
from pydantic import BaseModel

class Attachment(BaseModel):
    id = str
    signed_link = str

class Post(BaseModel):
    id: str
    title: str
    owner: str
    content: str
    attachments: list[Attachment] = []
    analytics: Analytics

class Reactions(BaseModel): 
    data = {
        "like": 0,
        "dislike": 0,
    }

class Comment(BaseModel):
    id = str
    owner = str
    content = str
    reactions = Reactions

class Analytics(BaseModel):
    reactions: Reactions
    comments: list[Comment] = []

class Profile(BaseModel):
    name: str
    country: str
    preferences: str
    photo: list[Attachment] = []
    verified: bool

class SocialGraph(BaseModel):
    followers: Follows
    following: Follows
    friends: list[UserData] = []

class Follows(BaseModel):
    len: int
    chain: list[UserData] = []

class Filters(BaseModel):
    disliked_posts: list[Post] = []

class UserData(BaseModel):
    id: str
    profile: Profile
    socialgraph: SocialGraph
    anti_filters: Filters

def get_current_user(raw):
    return UserData.model_validate(raw)

def get_current_posts(raw):
    return Post.model_validate(raw)
