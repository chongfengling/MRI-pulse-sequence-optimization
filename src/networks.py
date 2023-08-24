import torch
import torch.nn as nn
from utilities import *

class ActorNetwork(nn.Module):
    # Actor Network, generates action based on state
    # observation is the state
    def __init__(self, state_space, action_space, args):
        super(ActorNetwork, self).__init__()

        # Fully-connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(state_space, args.a_hidden1),
            nn.ReLU(),
            nn.Linear(args.a_hidden1, args.a_hidden2),
            nn.ReLU(),
            nn.Linear(args.a_hidden2, args.a_hidden3),
            nn.ReLU(),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(args.a_hidden3, action_space), nn.Sigmoid()
        )

    def forward(self, state):
        tmp = self.fc_layers(state)
        out = self.output_layer(tmp)

        return out


class CriticNetwork(nn.Module):
    # Critic Network, generates Q value based on (current?) state and action
    def __init__(self, state_space, action_space, args):
        super(CriticNetwork, self).__init__()

        # state input stream
        self.state_stream = nn.Sequential(
            nn.Linear(state_space, args.c_s_hidden1),
            nn.ReLU(),
            nn.Linear(args.c_s_hidden1, args.c_s_hidden2),
            nn.ReLU(),
        )

        # action input stream
        self.action_stream = nn.Sequential(
            nn.Linear(action_space, args.c_a_hidden1),
            nn.ReLU(),
            nn.Linear(args.c_a_hidden1, args.c_a_hidden2),
            nn.ReLU(),
        )

        # combined layer
        self.combined_layer = nn.Sequential(
            nn.Linear(args.c_s_hidden2 + args.c_a_hidden2, args.c_combined_hidden1),
            nn.ReLU(),
            nn.Linear(args.c_combined_hidden1, args.c_combined_hidden2),
            nn.ReLU(),
            nn.Linear(args.c_combined_hidden2, 1),
        )

    def forward(self, state, action):
        state_tmp = self.state_stream(state)
        action_tmp = self.action_stream(action)
        combined = torch.cat((state_tmp, action_tmp), dim=1)
        out = self.combined_layer(combined)
        return out