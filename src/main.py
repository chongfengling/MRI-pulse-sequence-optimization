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
    parser.add_argument('--FOV_x', default=64, type=int, help='FOV_x')
    parser.add_argument(
        '--N',
        default=64,
        type=int,
        help='sampling points in x space (and k space, time space during ADC)',
    )
    parser.add_argument(
        '--T2', default=3e1, type=float, help='T2 relaxation time in ms'
    )
    parser.add_argument(
        '--max_slew_rate',
        default=2e-4,
        type=float,
        help='max slew rate of gradient in (T / (m * ms)), defined as peak amplitude of gradient divided by rise time',
    )
    parser.add_argument(
        '--constraint_slew_rate',
        default=0.8,
        type=float,
        help='constraint on slew rate of gradient. limits the percentage of the duration of two constant gradients'
    )
    parser.add_argument('--seed', default=215, type=int, help='seed')
    parser.add_argument(
        '--state_space',
        default=64 * 2,
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
        default=32,
        type=int,
        help='number of episodes. Each episode initializes new random process and state',
    )
    parser.add_argument(
        '--num_steps_per_ep', default=1024, type=int, help='number of steps per episode'
    )
    parser.add_argument(
        '--num_episode_testing',
        default=16,
        type=int,
        help='number of episodes for testing. Each episode initializes new random process and state',
    )
    parser.add_argument(
        '--num_steps_per_ep_testing', default=1024, type=int, help='number of steps per episode for testing'
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
        default=64,
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
        '--memory_capacity', default=10000, type=int, help='memory capacity'
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
    parser.add_argument('--left_clip', default=0.05, type=float, help='left clip')
    parser.add_argument('--right_clip', default=0.95, type=float, help='right clip')

    parser.add_argument(
        '--gamma', default=0.9, type=float, help='reward discount factor'
    )  # 0.8
    parser.add_argument(
        '--tau', default=0.01, type=float, help='soft update factor'
    )  # 0.1

    args = parser.parse_args()

    return args

def train(
    agent, env, num_episode, num_steps_per_ep, args, path, plot=False, save=False, debug=False, test=False
):
    training_records = []
    test_records = []
    recommended_action = []
    path = f'{path}_e{num_episode}_s{num_steps_per_ep}'
    for i in range(num_episode):
        # reset the environment, the state (reconstructed density with two components: real and imaginary are initialized in the environment)
        state = env.reset()
        state_signal = state[: len(env.x_axis)] + 1j * state[len(env.x_axis) :]
        # record time and current reward in this episode
        t1 = time.time()
        episode_reward = []
        # info records the mse in the last step of current episode. initialized as 0
        info = mse_of_two_complex_nparrays(state_signal, env.density_complex)
        for j in range(num_steps_per_ep):
            # return an action based on the current state
            action = agent.select_action(state, exploration_noise=True)
            # interact with the environment
            state_, reward, done, info_ = env.step(action, info=info)
            # store the transition
            agent.store_transition(state, action, reward, state_, done)
            # update the network if the replay memory is full. if not, no update of networks
            if agent.mpointer > args.warmup:
                agent.update_network()
            # update the state: s_t = s_t+1
            state = state_
            info = info_
            # record the total reward in this episode
            episode_reward.append(reward)
            # if the episode is finished or the maximum number of steps is reached, print the episode reward and running time
            # sum of episode reward is the total reward of this episode
            if j == num_steps_per_ep - 1 or done:
                print(
                    '\rEpisode: {}/{}  | Episode Reward: {:.4f} | Running Time: {:.4f}'.format(
                        i+1, num_episode, np.sum(episode_reward), time.time() - t1
                    )
                )
                # each episode has its array-like reward
                training_records.append(episode_reward)
                break
            # plot the state and action at some steps if needed
            if not j % 128 and not i % 4 and debug:
                show_state(env, path, state, i, j, info_, reward)
        if i % 2 == 0 and i > 0 and test:
            # do test
            agent.eval()
            best_action_mse = 1e10
            best_action =  None
            state = env.reset(test=True)
            state_signal = state[: len(env.x_axis)] + 1j * state[len(env.x_axis) :]
            info = mse_of_two_complex_nparrays(state_signal, env.density_complex)
            done = False
            test_reward = []
            for k in range(num_steps_per_ep):
                action = agent.select_action(state, exploration_noise=False)
                state_, reward, done, info_ = env.step(action, info=info)
                # if the mse of the current action is smaller than the best mse, update the best mse and best action
                if info_ < best_action_mse:
                    best_action_mse = info_
                    best_action = action
                    best_info = np.concatenate((best_action, [best_action_mse]))
                state = state_
                info = info_
                test_reward.append(reward)
            recommended_action.append(best_info)
            test_records.append(np.sum(test_reward))
            if debug:
                print(f'testing reward: {np.sum(test_reward)}, minimum MSE: {best_action_mse}')
    np.savetxt(f'{path}_recommended_action.txt', recommended_action, delimiter=',')

    if plot:
        _, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        training_records = np.concatenate(training_records).reshape(-1)
        ax1.plot(range(len(training_records))[::32], training_records[::32])
        ax1.set_xlabel('step')
        ax1.set_ylabel('reward') 
        plt.savefig(f'{path}_training_records.png', dpi=300)

    agent.save_model(path=path)

# testing embedded in the training process
def test(agent, model_path, env, num_episode, num_steps_per_ep, debug=False, save=True):
    # agent = torch.load(model_path)
    agent.load_model(model_path)
    agent.is_training = False
    agent.eval()
    
    # store the reward of each episode during testing
    test_records = []
    # recommended action
    recommended_action = []
    for i in range(num_episode):
        #for  one episode, output action has a smallest mse
        best_action_mse = 1e10
        best_action =  None
        state = env.reset()
        state_signal = state[: len(env.x_axis)] + 1j * state[len(env.x_axis) :]
        info = mse_of_two_complex_nparrays(state_signal, env.density_complex)
        done = False
        episode_reward = []
        for j in range(num_steps_per_ep):
            action = agent.select_action(state, exploration_noise=False)
            state_, reward, done, info_ = env.step(action, info=info)
            # if the mse of the current action is smaller than the best mse, update the best mse and best action
            if info_ < best_action_mse:
                best_action_mse = info_
                best_action = action
                best_info = np.concatenate((best_action, [best_action_mse]))
            state = state_
            info = info_
            episode_reward.append(reward)
            if j == num_steps_per_ep - 1 or done:
                break
        # store the best action in this episode
        recommended_action.append(best_info)
        test_records.append(episode_reward)
        if debug: 
            print(f'episode {i}, episode reward: {np.sum(episode_reward)}')
    np.savetxt(f'{model_path}_recommended_action.txt', recommended_action, delimiter=',')

    if save:
        _, ax1 = plt.subplots(1, figsize=(10, 6))
        test_records = np.concatenate(test_records).reshape(-1)
        ax1.plot(range(len(test_records)), test_records)
        ax1.set_xlabel('episode')
        ax1.set_ylabel('Average Reward')
        ax1.set_title(f'Test Result: mean_reward = {np.mean(test_records)}')
        plt.savefig(f'{model_path}_testing_records.png', dpi=300)
        plt.close()

    if debug:
        print(f'Testing result: mean_reward = {np.mean(test_records)}')


def main():
    current_datetime = datetime.now()
    # Convert the month, day, hour, and minute to a string, excluding the year
    datetime_string = f"{current_datetime.month}-{current_datetime.day}-{current_datetime.hour}{current_datetime.minute:02}"
    if not os.path.exists(f'src/Training/{datetime_string}'):
        os.makedirs(f'src/Training/{datetime_string}')
    path = f'src/Training/{datetime_string}/'

    args = parse_arguments()
    env = Env(args=args)
    agent = DDPG(env=env, args=args)
    train(agent=agent, env=env, num_episode=args.num_episode, num_steps_per_ep=args.num_steps_per_ep, args=args, path=path, plot=True, save=True, debug=True, test=True)


if __name__ == "__main__":
    main()