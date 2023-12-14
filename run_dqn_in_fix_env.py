import numpy as np
from PIL import Image
import argparse
import os
import json
import datetime
import random

import torch
import cv2
from matplotlib import pyplot as plt

from rl_core import dqn, models
import deepmind_lab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(state, img_size, th=0.4):
    state = np.array(Image.fromarray(state).resize(img_size,Image.BILINEAR))
    state = state.astype(float) / 255.
    state = state.transpose(2,0,1)
    return state

def epsilon_compute(frame_id, epsilon_max=1, epsilon_min=0.2, epsilon_decay=100000):
    return epsilon_min + (epsilon_max - epsilon_min) * np.exp(-frame_id / epsilon_decay)

def save_video(img_buffer, fname, video_path="video"):
    size = (img_buffer[0].shape[1], img_buffer[0].shape[0])
    out = cv2.VideoWriter(os.path.join(video_path, fname), cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
    for i in range(len(img_buffer)):
        out.write(cv2.cvtColor(img_buffer[i], cv2.COLOR_BGR2RGB))
    out.release()

def action_convert(input):
    action_spec = np.zeros(7, dtype=np.intc)
    if input == 0:
        action_spec = np.array([-128, 0, 0, 0, 0, 0, 0], dtype=np.intc)
    elif input == 1:
        action_spec = np.array([128, 0, 0, 0, 0, 0, 0], dtype=np.intc)
    elif input == 2:
        action_spec = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.intc)
    return action_spec

def train(agent, stack_frames, img_size, level_script, config, exp_path="experiments_rl", seed=-10, eps_steps=1000, max_steps=1000000):
    total_step = 0
    episode = 0

    save_path = os.path.join(exp_path, "save")
    video_path = os.path.join(exp_path, "video")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    train_record = []
    eval_record = []
    while True:
        episode += 1
        # Reset environment.
        objectCount = random.randint(1,8)
        config['objectCount'] = f'{objectCount}'
        config['spawnCount'] = '1'
        env = deepmind_lab.Lab(level_script, 
            ['RGB_INTERLEAVED', 'DEBUG.CAMERA.TOP_DOWN', 'DEBUG.POS.TRANS'], config)

        env.reset(seed=seed)
        obs = env.observations()
        state = obs['RGB_INTERLEAVED']
        state = preprocess(state, img_size=img_size)
        state = state.repeat(stack_frames, axis=0)
        state_dict = {"obs":state}

        # Initialize information.
        step = 0
        total_reward = 0
        loss = 0.
        # pos_seq = []
        peek = False

        # One episode.
        while True:

            # Check env
            if not env.is_running() or step >= eps_steps:
                # Train record
                train_record.append({"eps":episode, "step": total_step, "score":total_reward})
                with open(model_path+'train_record.json', 'w') as file:
                    json.dump(train_record, file)
                print()
                break

            step += 1
            total_step += 1

            # Select action.
            epsilon = epsilon_compute(total_step)
            output = agent.choose_action(state_dict, epsilon)

            # Get next stacked state.
            action = action_convert(output)
            reward = env.step(action, num_steps=4)
            obs = env.observations()
            state_next = obs['RGB_INTERLEAVED']
            state_next = preprocess(state_next, img_size=img_size)
            state_next = np.concatenate([state_next, state[3:]], 0)
            state_next_dict = {"obs":state_next}

            # Show env status.
            if peek:
                agent_view = obs['RGB_INTERLEAVED']
                top_down_view = obs['DEBUG.CAMERA.TOP_DOWN']
                top_down_view = np.transpose(top_down_view, (1, 2, 0)).astype(np.uint8).copy()
                # agent_pos = obs['DEBUG.POS.TRANS']

                # plane_pos = (int((agent_pos[0] / 1700 * 640)), int((1700 - agent_pos[1]) / 1700 * 640))
                # if len(pos_seq) == 0:
                #     pos_seq.append(plane_pos)
                # elif len(pos_seq) > 0 and pos_seq[-1] != plane_pos:
                #     pos_seq.append(plane_pos)
                # cv2.polylines(top_down_view, np.array([pos_seq], dtype=np.int32), isClosed=False, color=(255, 0, 0), thickness=3)

                merge = np.concatenate([agent_view, top_down_view], axis=1)
                
                plt.imshow(merge)
                plt.show()

                peek = False

            # Store transition and learn.
            agent.store_transition(state_dict, output, reward, state_next_dict, False)
            if total_step > 4 * agent.batch_size:
                loss = agent.learn()

            #env.render(show=True)
            #cv2.waitKey(10)

            total_reward += reward
            state_dict = state_next_dict
            
            if total_step % 100 == 0 or step > eps_steps:
                print('\rEpisode: {:3d} | Step: {:4d} / {:4d} | Reward: {:.3f} / {:.3f} | Loss: {:.3f} | Epsilon: {:.3f}'\
                    .format(episode, step, total_step, reward, total_reward, loss, epsilon), end="")
            
            if total_step % 5000 == 0:
                print("\nSave Model ...")
                agent.save_model(path=save_path, step=total_step)
                print("Generate GIF ...")
                config['objectCount'] = '8'
                env = deepmind_lab.Lab(level_script, 
                    ['RGB_INTERLEAVED', 'DEBUG.CAMERA.TOP_DOWN', 'DEBUG.POS.TRANS'], config)
                img_buffer, score = play(env, agent, stack_frames, img_size, seed=seed)
                eval_record.append({"eps":episode, "step": total_step, "score":score})
                with open(model_path+'eval_record.json', 'w') as file:
                    json.dump(eval_record, file)
                save_video(img_buffer, "train_" + str(total_step).zfill(6) + ".avi", video_path)
                print("Score:", score)
                print("Done !!")
        
        if total_step > max_steps:
            break

def play(env, agent, stack_frames, img_size, eps_steps=2000, render=False, seed=None):
    img_buffer = []
    pos_seq = []

    # Reset environment.
    if seed is None:
        env.reset()
    else:
        env.reset(seed=seed)
        
    obs = env.observations()
    agent_view = obs['RGB_INTERLEAVED']
    top_down_view = obs['DEBUG.CAMERA.TOP_DOWN']
    top_down_view = np.transpose(top_down_view, (1, 2, 0)).astype(np.int32).copy()
    agent_pos = obs['DEBUG.POS.TRANS']
    state = preprocess(agent_view, img_size=img_size)
    state = state.repeat(stack_frames, axis=0)
    state_dict = {"obs":state}

    plane_pos = (int((agent_pos[0] / 900 * 192)), int((900 - agent_pos[1]) / 900 * 192))
    pos_seq.append(plane_pos)
    cv2.polylines(top_down_view, np.array([pos_seq], dtype=np.int32), isClosed=False, color=(255, 0, 0), thickness=1)


    env_render = np.concatenate([agent_view, top_down_view], axis=1).astype(np.uint8)
    img_buffer.append(env_render)

    # Initialize information.
    step = 0
    total_reward = 0
    done = False

    # One episode.
    while True:
        # Select action.
        output = agent.choose_action(state_dict, 0.2)

        # Get next stacked state.
        action = action_convert(output)
        if not env.is_running() or step > eps_steps:
            score = total_reward
            print()
            return img_buffer, score
        
        reward = env.step(action)
        obs = env.observations()
        agent_view = obs['RGB_INTERLEAVED']
        top_down_view = obs['DEBUG.CAMERA.TOP_DOWN']
        top_down_view = np.transpose(top_down_view, (1, 2, 0)).astype(np.int32).copy()
        agent_pos = obs['DEBUG.POS.TRANS']
        state_next = preprocess(agent_view, img_size=img_size)
        state_next = np.concatenate([state_next, state[3:]], 0)
        state_next_dict = {"obs":state_next}

        plane_pos = (int((agent_pos[0] / 900 * 192)), int((900 - agent_pos[1]) / 900 * 192))
        if len(pos_seq) > 0 and pos_seq[-1] != plane_pos:
            pos_seq.append(plane_pos)
        cv2.polylines(top_down_view, np.array([pos_seq], dtype=np.int32), isClosed=False, color=(255, 0, 0), thickness=1)


        env_render_next = np.concatenate([agent_view, top_down_view], axis=1).astype(np.uint8)
        img_buffer.append(env_render_next)

        # Store transition and learn.
        total_reward += reward
        print('\rStep: {:3d} | Reward: {:.3f} / {:.3f}'.format(step, reward, total_reward), end="")
            
        state_dict = state_next_dict
        step += 1


if __name__ == "__main__":
    ############ Parser ############
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument('--exp_name', nargs='?', type=str, default="rltest" ,help='Experiment name.')
    parser.add_argument('--n_items', '-n', nargs='?', type=int, default=15 ,help='Number of items.')
    parser.add_argument('--step', nargs='?', type=int, default=250000 ,help='Checkpoint step.')
    # New parser for dmlab
    parser.add_argument('--level_script', '-l', type=str,
                      default='explore_object_locations_xs',
                      help='The level that is to be played. Levels'
                      'are Lua scripts, and a script called \"name\" means that'
                      'a file \"assets/game_scripts/name.lua is loaded.')
    parser.add_argument('--level_settings', '-s', type=str, default=['width=192', 'height=192'],
                      action='append',
                      help='Applies an opaque key-value setting. The setting is'
                      'available to the level script. This flag may be provided'
                      'multiple times. Universal settings are `width` and '
                      '`height` which give the screen size in pixels, '
                      '`fps` which gives the frames per second, and '
                      '`random_seed` which can be specified to ensure the '
                      'same content is generated on every run.')
    test = parser.parse_args().test
    exp_name = parser.parse_args().exp_name
    n_items = parser.parse_args().n_items
    step = parser.parse_args().step
    level_script = parser.parse_args().level_script
    level_settings = parser.parse_args().level_settings

    config = {
      k: v
      for k, v in [s.split('=') for s in level_settings]
    }

    ############ Create Folder ############
    if not test:
        # Experiments Path
        exp_path = "experiments_dqn/"
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        # Model Path
        now = datetime.datetime.now()
        tinfo = "%d-%d-%d"%(now.year, now.month, now.day)
        model_path = os.path.join(exp_path, tinfo + "_" + exp_name + "/")
        video_path = os.path.join(model_path, "video/")
        save_path = os.path.join(model_path, "save/")
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    ############ Create Env ############
    # env = deepmind_lab.Lab(level_script, 
    #     ['RGB_INTERLEAVED', 'DEBUG.CAMERA.TOP_DOWN', 'DEBUG.POS.TRANS'], config)

    stack_frames = 4
    img_size = (64, 64)

    ############ Create Agent ############
    agent = dqn.DQNAgent(
        n_actions = 3,
        input_shape = [(stack_frames)*3, *img_size],
        qnet = models.QNet,
        device = device,
        learning_rate = 2e-4, 
        reward_decay = 0.98,
        replace_target_iter = 1000, 
        memory_size = 10000,
        batch_size = 32,)

    if not test:
        train(agent, stack_frames, img_size, level_script, config, model_path, eps_steps=1000, max_steps=250000)
    else:
        agent.load_model(os.path.join(exp_name, "save"), step=step)
        eval_path = os.path.join(exp_name, "eval_test/")
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)
        score_rec = []

        for seed in range(100):
            objectCount = random.randint(1, 9)
            config['objectCount'] = str(objectCount)
            config['spawnCount'] = '1'
            env = deepmind_lab.Lab(level_script, 
                ['RGB_INTERLEAVED', 'DEBUG.CAMERA.TOP_DOWN', 'DEBUG.POS.TRANS'], config)
            print(f"Maze {seed}")
            img_buffer, score = play(env, agent, stack_frames, img_size, seed=-1)
            save_video(img_buffer, "test_" + str(seed).zfill(4) + ".avi", eval_path)
            score_rec.append(score)

        print(score_rec)
        print(np.array(score_rec).mean(), "+/-", np.array(score_rec).std())