import time

import retro
from retro.scripts.playback_movie import main
from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm


def train_test(model, env, model_name):
    # model = A2C.load('/gdrive/My Drive/ROMS/Fighter_a2c_pt2.zip')
    model.set_env(env)
    model.learn(total_timesteps=1000)
    # Saves Model into
    model.save(model_name)  # "Whatever Your File Name is/" + modelname)
    env.close()


def train(model):
    sts = ['RyuVsGuile', 'RyuVsBlanka', 'RyuVsRyu', 'RyuVsKen', 'RyuVsChunLi', 'RyuVsZangief', 'RyuVsDhalsim',
           'RyuVsHonda',
           'RyuVsBalrog', 'RyuVsVega', 'RyuVsSagat', 'RyuVsBison']
    sts = ['Champion.Level1.RyuVsGuile']
    start_time = time.time()
    for st in tqdm(sts, desc='Main Loop'):
        print(st)
        env = DummyVecEnv(
            [lambda: retro.make('StreetFighterIISpecialChampionEdition-Genesis', state=st, scenario='scenario')])
        model.set_env(env)
        model.learn(total_timesteps=500000)
        model.save(modelname)
        env.close()
    end_time = time.time() - start_time
    print(f'\n The Training Took {end_time} seconds')


def test(model_path):
    env = DummyVecEnv([lambda: retro.make('StreetFighterIISpecialChampionEdition-Genesis', state='Champion.Level1.RyuVsGuile',
                                          record="./records")])
    model = A2C.load(model_path)
    model.set_env(env)
    obs = env.reset()
    done = False
    reward = 0
    while not done:
        actions, _ = model.predict(obs)
        obs, rew, done, info = env.step(actions)
        reward += rew
    print(reward)
    env.close()
    ### Convert BK2 to MP4 File
    # !python / usr / local / lib / python3
    # .6 / dist - packages / retro / scripts / playback_movie.py
    # "/gdrive/My Drive/Level16.RyuVsHonda-Easy-000000.bk2"



gamename = 'StreetFighterIISpecialChampionEdition-Genesis'
modelname = 'Fighter_a2c_pt2'  # whatever name you want to give it

env = DummyVecEnv([lambda: retro.make(gamename, state='Champion.Level1.RyuVsGuile')])
model = A2C(ActorCriticCnnPolicy, env, n_steps=128, verbose=1, tensorboard_log="./logs")
env.close()

model_path = "./models/" + modelname
# train_test(model, env, model_path)
# test(model_path)

train(model)

main(["./records/StreetFighterIISpecialChampionEdition-Genesis-Champion.Level1.RyuVsGuile-000000.bk2"])
