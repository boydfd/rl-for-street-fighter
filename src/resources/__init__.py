import os
import time
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Callable

import retro
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from tinydb import TinyDB
from tinydb.storages import YamlStorage
from tinydb.table import Document
from tqdm import tqdm


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


class Session(Document):
    def __init__(self, doc_id: Any, value=None):
        if value is None:
            value = {}
        super(Session, self).__init__(value, doc_id)

    def get_models_path(self, suffix=""):
        return self.get_path_with_doc_id_and("model" + suffix)

    def get_path_with_doc_id_and(self, model):
        return session_path.joinpath(str(self.doc_id)).joinpath(model)

    def get_records_path(self):
        return self.get_path_with_doc_id_and("records")

    def get_tensor_board_logs_path(self):
        return self.get_path_with_doc_id_and("tensorboard_logs")

    def save_model(self, model: BaseAlgorithm, model_path_suffix=""):
        model.save(self.get_models_path(model_path_suffix))

    def load_model(self, model_class):
        model_class.load(self.get_models_path())

    def train(self, model_getter: Callable[[SubprocVecEnv, PathLike], BaseAlgorithm]):
        sts = ['ryu_vs_ryu', 'car', 'ryu_vs_hoda', 'ryu_vs_guile', 'ryu_vs_blanka']
        env = SubprocVecEnv(
            [lambda: retro.make('StreetFighterIISpecialChampionEdition-Genesis', state=st, scenario='scenario')
             for st in sts]
        )
        model = model_getter(env, self.get_tensor_board_logs_path())
        for _ in tqdm(range(1, 1000), desc='Main Loop'):
            model.learn(total_timesteps=5)
            path = self.get_models_path(str(datetime.now()))
            print(path)
            model.save(path)
        env.close()


db = TinyDB(db_path, storage=YamlStorage)
session_table = db.table("session")
