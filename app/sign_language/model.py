import torch.nn as nn


class SignLanguageMLP(nn.Module):
    """Implementuje siec MLP do klasyfikacji liter PJM z cechami relatywnymi."""

    def __init__(self, input_size=63, hidden_size=256, num_classes=26):
        super(SignLanguageMLP, self).__init__()
        # architektura glebsza dla bogatszych cech (88D)
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        return self.network(x)
