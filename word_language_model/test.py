import torch
import torch.nn as nn

encoder = nn.Embedding(10,2)
decoder = nn.Linear(2,10)
encoder.weight.data()