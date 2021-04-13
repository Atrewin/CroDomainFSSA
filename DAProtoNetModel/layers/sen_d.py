import sys
sys.path.append('..')
from torch import autograd, optim, nn


class Sen_Discriminator(nn.Module):

    def __init__(self, hidden_size=230, num_labels=2):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        self.sentimentClass = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/8)*2),
            nn.ReLU(),
            nn.Linear(int(hidden_size/8)*2, int(hidden_size/16)),
            nn.ReLU(),
            nn.Linear(int(hidden_size/16), 2)
        )

        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(hidden_size, 2)


    def forward(self, x):
        # logits = self.sentimentClass(x)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop(x)
        logits = self.fc2(x)
        return logits


