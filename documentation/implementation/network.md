# Neural Network Model

The core of the AlphaZero-Clone project is the neural network model responsible for predicting move probabilities (policy) and evaluating board states (value). This model is based on a residual neural network architecture with configurable depth and hidden layer sizes. The network processes the game state representation and outputs policy and value predictions for the current board state.

```python
class Network(nn.Module):
    def __init__(self, num_res_blocks: int, hidden_size: int, device: torch.device) -> None:
        encoding_channels, row_count, column_count = CurrentGame.representation_shape
        action_size = CurrentGame.action_size

        self.startBlock = nn.Sequential(
            nn.Conv2d(encoding_channels, hidden_size, kernel_size=3, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList([ResBlock(hidden_size) for _ in range(num_res_blocks)])

        self.policyHead = nn.Sequential(
            nn.Conv2d(hidden_size, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (row_count - 2) * (column_count - 2), action_size),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(hidden_size, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (row_count - 2) * (column_count - 2), 1),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        return self.policyHead(x), self.valueHead(x)
        
class ResBlock(nn.Module):
    def __init__(self, num_hidden: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(num_hidden) 

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual) 
```

## Key Components Explained

- **Network Class:** Defines the neural network architecture with a configurable number of residual blocks. It processes the game state and outputs both policy and value predictions.
- **ResBlock Class:** Implements a residual block with two convolutional layers and batch normalization, facilitating deeper networks by mitigating vanishing gradient issues.
