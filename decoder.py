import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Deocder(nn.Module):
  def __init__(self, input_dim, hidden_dims):
    super(Deocder, self).__init__()
    self.fc_layers = nn.Sequential(
      nn.Linear(input_dim * 2, hidden_dims[0]), # 要変更(エッジが含まれていない)
      nn.ReLU(),
      nn.Linear(hidden_dims[0], hidden_dims[1]),
      nn.ReLU(),
      nn.Linear(hidden_dims[1],2) # 要変更(現状はドアと壁のみ)
    )

  def forward(self, X_encoded, X_initial=None):
    num_nodes = X_encoded.size(0)
    features = X_encoded
    if not X_initial == None:
      features = torch.cat([X_encoded, X_initial], dim=-1) # 要変更(エッジが含まれていない)

    pair_features = torch.cat([
      features.repeat(1, num_nodes).view(num_nodes * num_nodes, -1),
      features.repeat(num_nodes, 1)
    ], dim=-1).view(num_nodes, num_nodes, -1)

    edge_probs = self.fc_layers(pair_features)
    return torch.sigmoid(edge_probs)

def compute_loss(predicted_edges, true_edges):
  loss = F.binary_cross_entropy(predicted_edges, true_edges)
  return loss

num_nodes = 10
X_encoded = torch.randn(num_nodes, 32)
true_edges = torch.zeros(num_nodes, num_nodes, 2)
true_edges[0,1,0] = 1
true_edges[1,2,1] = 1

input_dim = 32
hidden_dims = [64, 32]
model = Deocder(input_dim=input_dim, hidden_dims=hidden_dims)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
  model.train()
  optimizer.zero_grad()

  predicted_edges = model(X_encoded)
  loss = compute_loss(predicted_edges, true_edges)

  loss.backward()
  optimizer.step()

  if (epoch + 1) % 10 == 0:
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
  predicted_edges = model(X_encoded)
  print("Predicted edge probabilities:", predicted_edges)
