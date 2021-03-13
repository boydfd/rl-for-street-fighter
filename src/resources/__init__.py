import os
from pathlib import Path

from tinydb import TinyDB
from tinydb.storages import YamlStorage


def get_path(path, file__=__file__):
    directory = os.path.dirname(file__)
    return os.path.join(directory, path)


def path_for(relative_path):
    return Path(get_path(relative_path, __file__))


models_path = path_for("./models")
db_path = path_for("./db")
records_path = path_for("./records")
tensor_board_logs_path = path_for("./tensor_board_logs")
session_path = path_for("./sessions")

db = TinyDB(db_path, storage=YamlStorage)
session_table = db.table("session")
