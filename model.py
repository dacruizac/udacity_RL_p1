import torch
import torch.nn as nn
import torch.nn.functional as F

class pytorch_model_1(nn.Module):
    """ Policy Model. """

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """ Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(pytorch_model_1, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """ Build a network that maps state -> action values. """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class pytorch_model_2(nn.Module):
    """ Policy Model. """

    def __init__(self, state_size, action_size, seed, hidden_units = [64,64]):
        """ Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        if len(hidden_units<2):
            hidden_units=[64,64]

        self.fc=[nn.Linear(state_size, hidden_units[0])]
        for i in range(1,len(hidden_units)):
            layer = nn.Linear(hidden_units[i-1], hidden_units[i]) 

        self.fc_final = nn.Linear(hidden_units[-1], action_size)

    def forward(self, state):
        """ Build a network that maps state -> action values. """
        x = state
        for layer in self.fc:
            x = F.relu(layer(x))
        return self.fc_final(x)