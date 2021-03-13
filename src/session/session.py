from datetime import datetime
from os import PathLike
from typing import Any, Callable

import retro
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import SubprocVecEnv
from tinydb.table import Document
from tqdm import tqdm

from src.resources import session_path


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
            model.learn(total_timesteps=50000)
            path = self.get_models_path(str(datetime.now()))
            print(path)
            model.save(path)
        env.close()
