import json
from core.type import get_current_user, Post, get_current_posts

default_db_files = "datasets\raw\in_network.json,datasets\raw\in_network.json"
user_file = "datasets\raw\user.json"

class Database():
    conn = None
    state = list[Post] = []

    def __init__(conn_string):
        pass
    
    def retrieve_all(self) -> list[Post]:
        for i in self.conn.split(","):
            try:
                with open(i, 'r', encoding='utf-8') as file:
                    self.state.append(get_current_posts(json.load(file)))

            except FileNotFoundError:
                print(f"Error: The file '{i}' was not found.")
            except json.JSONDecodeError as e:
                print(f"Error: Failed to decode JSON. Check file format. Details: {e}")

    def create():
        pass

    def read():
        pass

    def write():
        pass

    def delete():
        pass

def setup_db():
    return Database(default_db_files)

def get_user():
    raw = Database(user_file)
    return get_current_user(raw)