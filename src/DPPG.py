import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

# action space: {t1:_, t3:_, d1:_, d2:_, G1:_, G2:_} 6 arrays. 

class Env():
    # Environment, generates state and reward based on action
    def __init__(self):
        pass

    def make(self, env_name, plot=False):
        # create the environment
        if env_name == "Two-Constant-Gradient":
            # x space (spatial space)
            self.FOV_x = 512 # field of view in x space
            self.N = 512 # sampling points in x space (and k space, time space during ADC)
            self.delta_x = self.FOV_x / self.N # sampling interval in x space
            self.x_axis = np.linspace(-self.FOV_x / 2, self.FOV_x / 2 - self.delta_x, self.N) # symmetric x space
            # create object
            self.density = np.zeros(len(self.x_axis))
            self.density[int(len(self.x_axis) / 4 + len(self.x_axis) / 8): int(len(self.x_axis) / 4 * 3 - len(self.x_axis) / 8)] = 1
            if plot:
                plt.figure(figsize=(10, 6))
                plt.plot(self.x_axis, self.density, '-', label='object')
                plt.legend()
                plt.xlabel('x')
                plt.ylabel('density')
                plt.grid()
                plt.show()

        elif env_name == "Two-Constant-Gradient-With-slope":
            pass

    def reset(self):
        # reset the environment and return the initial state
        pass

    def action_space_sample(self):
        # return a random action from the action space
        # 6 variables: t1:_, t3:_, d1:_, d2:_, G1:_, G2:_
        self.low = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0])
        self.high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.action_space = np.random.uniform(low=self.low, high=self.high, size=6)
        

    def step(self, action):
        # take an action and return the next state, reward, a boolean indicating if the episode is done and additional info
        pass

    # def action_space.sample(self):
    #     # return a random action
    #     pass

    def render(self):
        # display the current state of the environment
        pass

    

class ActorNetwork():
    # Actor Network, generates action based on state
    # observation is the state
    def __init__(self, state_space, action_space):
        super(ActorNetwork, self).__init__()

        # Fully-connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(state_space, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        # Output layer
        self.output_layer = nn.Linear(32, action_space)


    def forward(self, state):
        tmp = self.fc_layers(state)
        out = self.output_layer(tmp)
        return out

class CriticNetwork():
    # Critic Network, generates Q value based on (current?) state and action
    def __init__(self, state_space, action_space):
        super(CriticNetwork, self).__init__()

        # state input stream
        self.state_stream = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )

        # action input stream
        self.action_stream = nn.Sequential(
            nn.Linear(action_space, 16),
            nn.ReLU()
            nn.Linear(16, 32),
            nn.ReLU()
        )

        # combined layer
        self.combined_layer = nn.Sequential(
            nn.Linear(32 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        def forward(self, state, action):
            state_tmp = self.state_stream(state)
            action_tmp = self.action_stream(action)
            combined = torch.cat((state_tmp, action_tmp), dim=1)
            out = self.combined_layer(combined)
            return out
        

class DPPG():

    def __init__(self, state_space, action_space, env):
        self.state_space = state_space
        self.action_space = action_space

        self.env = env

        self.seed = 215

        # create Actor Network and its target network
        self.actor = ActorNetwork(state_space, action_space)
        self.actor_target = ActorNetwork(state_space, action_space)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001) #!
        # make sure the target network has the same weights as the original network
        for target_param, param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(param.data)

        # create Critic Network and its target network
        self.critic = CriticNetwork(state_space, action_space)
        self.critic_target = CriticNetwork(state_space, action_space)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001) #!
        # make sure the target network has the same weights as the original network
        for target_param, param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(param.data)

        # initialize replay buffer
        # self.memory = 
        # self.random_process = 

        # define hyper-parameters
        self.batch_size = 64
        self.tau = 0.001
        self.discount = 0.99
        self.depsilon = 1.0 / 50000

        self.epsilon = 1.0
        self.s_t = None # most recent state
        self.a_t = None # most recent action
        self.is_training = True

    def actor(self, state, exploraion_noise=True):
        # return an action based on the current state with or without exploration noise
        pass

    def store_transition(self, state, action, reward, state_, done):
        pass

    def update_network(self):
        pass

def main():

    def train(agent, env, num_episode = 1000, num_steps_per_ep = 1000):
        for i in range(num_episode): 
            # reset the environment
            state = env.reset()
            # record time and current reward in this episode
            t1 = time.time()
            episode_reward = 0

            for j in range(num_steps_per_ep):
                # return an action based on the current state
                actor = agent.actor(state)
                # interact with the environment
                state_, reward, done, info = env.step(actor)
                # store the transition
                agent.store_transition(state, actor, reward, state_, done)

                # update the network if the replay memory is full
                if agent.memory.is_full():
                    agent.update_network()

                # output records
                state = state_
                episode_reward += reward
                if j == num_steps_per_ep - 1:
                    print(
                        '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                            i, num_episode, episode_reward,
                            time.time() - t1
                        ), end=''
                    )

                


