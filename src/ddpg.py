from utilities import *
from networks import ActorNetwork, CriticNetwork
import torch.nn as nn
import random

class DDPG:
    def __init__(self, env, args):
        self.state_space = args.state_space
        self.action_space = args.action_space

        self.env = env  #! necessary?

        self.seed = args.seed
        self.lr_a, self.lr_c = args.lr_a, args.lr_c

        # create Actor Network and its target network
        self.actor = ActorNetwork(self.state_space, self.action_space, args)
        self.actor_target = ActorNetwork(self.state_space, self.action_space, args)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr_a
        )  
        # make sure the target network has the same weights as the original network
        hard_update(self.actor_target, self.actor)

        self.critic = CriticNetwork(self.state_space, self.action_space, args)
        self.critic_target = CriticNetwork(self.state_space, self.action_space, args)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr_c
        )  
        # make sure the target network has the same weights as the original network

        hard_update(self.critic_target, self.critic)
        # initialize replay buffer
        self.memory_capacity = args.memory_capacity
        self.batch_size = args.batch_size
        # store one current states, one action, one reward, one next state, one done
        self.memory = np.zeros(
            (self.memory_capacity, self.state_space * 2 + self.action_space + 2),
            dtype=np.float32,
        )
        self.mpointer = 0  # memory pointer

        # define hyper-parameters
        self.tau = args.tau
        self.gamma = args.gamma

        self.exploration_var = args.exploration_var
        self.left_clip, self.right_clip = args.left_clip, args.right_clip

        # self.epsilon = 1.0
        self.s_t = None  # most recent state
        self.a_t = None  # most recent action
        self.is_training = True

        self.criterion = nn.MSELoss()

    def select_action(self, state, exploration_noise=True):
        # return an action based on the current state with or without exploration noise
        # print(to_numpy(self.actor(to_tensor(state))).shape)
        action = to_numpy(self.actor(to_tensor(state)))

        if exploration_noise:
            # add noise controlled by hyper-parameter var
            action = np.clip(
                np.random.normal(action, self.exploration_var),
                self.left_clip,
                self.right_clip,
            )
            action[:8] = np.exp(action[:8]) / np.sum(np.exp(action[:8]), axis=0)

        self.s_t = action
        return action

    def store_transition(self, state, action, reward, state_, done):
        # store the transition in the replay buffer
        transition = np.hstack(
            (
                state.astype(np.float32),
                action.astype(np.float32),
                [reward],
                state_.astype(np.float32),
                [done],
            )
        )

        current_index = (
            self.mpointer % self.memory_capacity
        )  # replace if memory is full
        self.memory[current_index, :] = transition
        self.mpointer += 1

    def update_network(self):
        # update two networks based on the transitions stored in the replay buffer

        # sample a batch of transitions
        indices = random.sample(range(self.memory_capacity), self.batch_size)

        batch = self.memory[indices, :]
        batch_state = to_tensor(batch[:, : self.state_space])
        batch_action = to_tensor(
            batch[:, self.state_space : self.state_space + self.action_space]
        )
        # ? scale the reward to small value?
        batch_reward = batch[:, -self.state_space - 2 : -self.state_space - 1]
        batch_state_ = to_tensor(batch[:, -self.state_space - 1 : -1])
        batch_done = batch[:, -1:]

        # prepare for the target q batch
        q_target_batch = self.critic_target(
            batch_state_, self.actor_target(batch_state_)
        )

        # if game is over (done = 0.0), no value for the next state's Q-value
        # y is the target Q-value
        y = batch_reward + self.gamma * to_numpy(q_target_batch) * batch_done
        # calculate the current Q-value
        q_batch = self.critic(batch_state, batch_action)
        # calculate the value loss
        value_loss = self.criterion(q_batch, to_tensor(y))
        # update the critic network based on the value loss
        self.critic.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # update the actor network
        policy_loss = -self.critic(batch_state, self.actor(batch_state)).mean()
        self.actor.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # (soft) update two target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def save_model(self, path):
        # save the model
        torch.save(self.actor.state_dict(), path + '_actor.pth')
        torch.save(self.critic.state_dict(), path + '_critic.pth')
        torch.save(self.actor_target.state_dict(), path + '_actor_target.pth')
        torch.save(
            self.critic_target.state_dict(),
            path + '_critic_target.pth',
        )

    def load_model(self, path):
        # load the model
        self.actor.load_state_dict(torch.load(path + '_actor.pth'))
        self.critic.load_state_dict(torch.load(path + '_critic.pth'))

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()