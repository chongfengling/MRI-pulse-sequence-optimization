import numpy as np
import time
import matplotlib.pyplot as plt
from utilities import *
import argparse
from datetime import datetime
import os
from env import Env
from ddpg import DDPG

def parse_arguments():
    parser = argparse.ArgumentParser(description='DPPG_Two-Constant-Gradient')
    parser.add_argument(
        '--env', default='Two-Constant-Gradient-With_Slope', type=str, help='env'
    )
    parser.add_argument('--FOV_x', default=32, type=int, help='FOV_x')
    parser.add_argument(
        '--N',
        default=32,
        type=int,
        help='sampling points in x space (and k space, time space during ADC)',
    )
    parser.add_argument(
        '--T2', default=3e0, type=float, help='T2 relaxation time in ms'
    )
    parser.add_argument(
        '--max_slew_rate',
        default=2e-4,
        type=float,
        help='max slew rate of gradient in (T / (m * ms)), defined as peak amplitude of gradient divided by rise time',
    )
    parser.add_argument(
        '--slope_penalty_factor', default=0, type=float, help='slope penalty factor'
    )
    parser.add_argument('--seed', default=215, type=int, help='seed')
    parser.add_argument(
        '--state_space',
        default=32 * 2,
        type=int,
        help='state_space including density of objects',
    )
    parser.add_argument(
        '--action_space',
        default=3,
        type=int,
        help='action_space including t1, t3, d1, d2, G1symbol, G2symbol, Gvalue',
    )
    parser.add_argument(
        '--num_episode',
        default=4,
        type=int,
        help='number of episodes. Each episode initializes new random process and state',
    )
    parser.add_argument(
        '--num_steps_per_ep', default=4096, type=int, help='number of steps per episode'
    )

    parser.add_argument(
        '--a_hidden1', default=512, type=int, help='hidden layer 1 in actor network'
    )
    parser.add_argument(
        '--a_hidden2', default=128, type=int, help='hidden layer 2 in actor network'
    )
    parser.add_argument(
        '--a_hidden3', default=32, type=int, help='hidden layer 3 in actor network'
    )

    parser.add_argument(
        '--c_s_hidden1',
        default=128,
        type=int,
        help='hidden layer 1 in critic network for state input stream',
    )
    parser.add_argument(
        '--c_s_hidden2',
        default=32,
        type=int,
        help='hidden layer 2 in critic network for state input stream',
    )
    parser.add_argument(
        '--c_a_hidden1',
        default=16,
        type=int,
        help='hidden layer 1 in critic network for action input stream',
    )
    parser.add_argument(
        '--c_a_hidden2',
        default=32,
        type=int,
        help='hidden layer 2 in critic network for action input stream',
    )

    parser.add_argument(
        '--c_combined_hidden1',
        default=128,
        type=int,
        help='hidden layer 1 in critic network for combined network',
    )
    parser.add_argument(
        '--c_combined_hidden2',
        default=256,
        type=int,
        help='hidden layer 2 in critic network for combined network',
    )

    parser.add_argument(
        '--lr_a', default=0.0001, type=float, help='learning rate of actor network'
    )
    parser.add_argument(
        '--lr_c', default=0.001, type=float, help='learning rate of critic network'
    )
    parser.add_argument('--warmup', default=256, type=int, help='warmup, no training')
    parser.add_argument(
        '--memory_capacity', default=500000, type=int, help='memory capacity'
    )
    parser.add_argument(
        '--batch_size', default=128, type=int, help='minibatch size from memory'
    )

    parser.add_argument(
        '--exploration_var',
        default=0.8,
        type=float,
        help='exploration variance in random process',
    )
    parser.add_argument('--left_clip', default=0.01, type=float, help='left clip')
    parser.add_argument('--right_clip', default=0.99, type=float, help='right clip')

    parser.add_argument(
        '--gamma', default=0.9, type=float, help='reward discount factor'
    )  # 0.8
    parser.add_argument(
        '--tau', default=0.01, type=float, help='soft update factor'
    )  # 0.1

    args = parser.parse_args()

    return args

def train(
    agent, env, num_episode, num_steps_per_ep, args, path, plot=False, save=False
):
    reward_record = []
    path = f'{path}_e{num_episode}_s{num_steps_per_ep}'
    for i in range(num_episode):
        # reset the environment
        state = env.reset()
        # record time and current reward in this episode
        t1 = time.time()
        episode_reward = 0
        reward_set = []
        for j in range(num_steps_per_ep):
            # return an action based on the current state
            action = agent.select_action(state, exploration_noise=True)
            # interact with the environment
            state_, reward, done, info = env.step(action)
            # store the transition
            agent.store_transition(state, action, reward, state_, done)
            # update the network if the replay memory is full
            if agent.mpointer > args.warmup:
                agent.update_network()
            # output records
            state = state_
            episode_reward += reward
            reward_set.append(reward)
            if j == num_steps_per_ep - 1 or done:
                # if True:
                print(
                    '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                        i+1, num_episode, episode_reward, time.time() - t1
                    )
                )
                reward_record.append(episode_reward)
                break
            if not j % 128:
                show_state(env, path, state, i, j, info, reward)

        print(f'mean reward: {np.mean(reward_set)}, std: {np.std(reward_set)}')
    agent.save_model(path=path)

def test(agent, model_path, env, debug=False, save=True):
    # agent = torch.load(model_path)
    agent.load_model(model_path)
    agent.is_training = False
    agent.eval()
    
    num_episode = 10
    num_steps_per_ep = 4096
    # store the reward of each episode during testing
    test_result = []

    for i in range(num_episode):
        state = env.reset()
        done = False
        episode_reward = 0
        for j in range(num_steps_per_ep):
            action = agent.select_action(state, exploration_noise=False)
            state_, reward, done, info = env.step(action)
            state = state_
            episode_reward += reward
            if j == num_steps_per_ep - 1 or done:
                break
        if debug:
            print(f'episode {i}, episode reward: {episode_reward}')
        test_result.append(episode_reward)
    if save:
        _, ax1 = plt.subplots(1, figsize=(10, 6))
        ax1.plot(range(len(test_result)), test_result)
        ax1.set_xlabel('episode')
        ax1.set_ylabel('Average Reward')
        ax1.set_title(f'Test Result: mean_reward = {np.mean(test_result)}')
        plt.savefig(f'{model_path}_test.png', dpi=300)
        plt.close()

    if debug:
        print(f'Testing result: mean_reward = {np.mean(test_result)}')


def main():
    current_datetime = datetime.now()
    # Convert the month, day, hour, and minute to a string, excluding the year
    datetime_string = f"{current_datetime.month}-{current_datetime.day}-{current_datetime.hour}{current_datetime.minute:02}"
    if not os.path.exists(f'src/Training/{datetime_string}'):
        os.makedirs(f'src/Training/{datetime_string}')
    # path = f'src/Training/{datetime_string}/'
    path = 'src/Training/8-13-18-30/'

    args = parse_arguments()
    env = Env(args=args, plot=False)
    agent = DDPG(env=env, args=args)
    env.make(args=args)
    train(agent=agent, env=env, num_episode=args.num_episode, num_steps_per_ep=args.num_steps_per_ep, args=args, path=path, plot=False, save=True)
    path = f'{path}_e{args.num_episode}_s{args.num_steps_per_ep}'
    test(agent=agent, model_path=path, env=env, debug=True, save=True)


if __name__ == "__main__":
    main()