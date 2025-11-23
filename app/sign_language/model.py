import torch.nn as nn


class SignLanguageMLP(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=26):
        super(SignLanguageMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),  # zapobiega overfittingowi przy malym datasecie
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.network(x)
