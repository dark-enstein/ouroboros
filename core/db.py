import json
import os
from core.type import UserData, Post, get_current_user, get_current_posts

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IN_NETWORK_FILE = os.path.join(BASE_DIR, "datasets", "raw", "in_network.json")
OUT_NETWORK_FILE = os.path.join(BASE_DIR, "datasets", "raw", "out_network.json")
USER_FILE = os.path.join(BASE_DIR, "datasets", "raw", "user.json")


class Database:
    def __init__(self, conn_files: list[str]):
        self.conn_files = conn_files
        self.state: list[Post] = []

    def retrieve_all(self) -> list[Post]:
        self.state = []
        for filepath in self.conn_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                posts = get_current_posts(raw)
                self.state.extend(posts)
            except FileNotFoundError:
                print(f"[db] file not found: {filepath}")
            except json.JSONDecodeError as e:
                print(f"[db] json decode error in {filepath}: {e}")
        return self.state

    def retrieve_by_network(self) -> tuple[list[Post], list[Post]]:
        in_posts, out_posts = [], []
        for filepath in self.conn_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                posts = get_current_posts(raw)
                network = "in" if "in_network" in filepath else "out"
                for p in posts:
                    p.network = network
                if network == "in":
                    in_posts.extend(posts)
                else:
                    out_posts.extend(posts)
            except FileNotFoundError:
                print(f"[db] file not found: {filepath}")
            except json.JSONDecodeError as e:
                print(f"[db] json decode error in {filepath}: {e}")
        return in_posts, out_posts


def setup_db() -> Database:
    return Database([IN_NETWORK_FILE, OUT_NETWORK_FILE])


def get_user() -> UserData:
    with open(USER_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return get_current_user(raw)
