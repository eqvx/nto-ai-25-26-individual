"""
PyTorch Hybrid Neural Recommender for Book Ratings

- Embedding layers for major categorical features (user/book/author/etc.)
- Accepts arbitrary dense/numeric features (BERT, TFIDF, aggregates)
- Simple interface for fit, predict, evaluate, GPU support
- Easily extensible for stacking, meta-learning, deeper architectures

Example usage:
    from models.neural_recommender import NeuralRecommender, train_nn
    model = NeuralRecommender(...)
    train_nn(model, X_train, y_train, X_val, y_val)
    preds = model.predict(X_test)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NeuralRecommender(nn.Module):
    def __init__(self, cat_feat_info, dense_dim, out_dim=1, emb_dim=32, hidden_dims=[128,64], dropout=0.2):
        """
        cat_feat_info: list of (feature_name, num_categories)
        dense_dim: int, dimension of concatenated dense input (aggregates, BERT, etc.)
        out_dim: usually 1 for regression
        emb_dim: output dimension for all embeddings (can be int or dict)
        hidden_dims: list of ints, sizes for MLP layers
        dropout: dropout rate for MLP
        """
        super().__init__()
        self.cat_feat_names = [name for name, _ in cat_feat_info]
        self.emb_layers = nn.ModuleDict({name: nn.Embedding(num, emb_dim if isinstance(emb_dim,int) else emb_dim[name])
                                         for name,num in cat_feat_info})
        total_emb_dim = sum([emb_dim if isinstance(emb_dim,int) else emb_dim[name] for name,_ in cat_feat_info])
        self.input_dim = total_emb_dim + dense_dim
        mlp = []
        last_dim = self.input_dim
        for hid in hidden_dims:
            mlp += [nn.Linear(last_dim, hid), nn.ReLU(), nn.Dropout(dropout)]
            last_dim = hid
        mlp += [nn.Linear(last_dim, out_dim)]
        self.mlp = nn.Sequential(*mlp)
    def forward(self, x_cat, x_dense):
        # x_cat: list of category tensors (batch_size,), one per cat_feat
        # x_dense: (batch_size, dense_dim)
        emb = [self.emb_layers[name](x_cat[i]) for i,name in enumerate(self.cat_feat_names)]
        x = torch.cat(emb + [x_dense], dim=1)
        return self.mlp(x).squeeze(-1)
    def predict(self, x_cat, x_dense):
        self.eval()
        with torch.no_grad():
            return self.forward(x_cat, x_dense).cpu().numpy()

def train_nn(model, X_cat, X_dense, y, X_cat_val, X_dense_val, y_val,
             lr=1e-3, batch_size=512, n_epochs=10, device=None, verbose=True):
    """
    model: NeuralRecommender
    X_cat: list/tuple of numpy arrays (N,)
    X_dense: numpy array (N, D)
    y: numpy array (N,)
    *_val: validation sets
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(*(torch.LongTensor(x) for x in X_cat), torch.FloatTensor(X_dense), torch.FloatTensor(y))
    valset = TensorDataset(*(torch.LongTensor(x) for x in X_cat_val), torch.FloatTensor(X_dense_val), torch.FloatTensor(y_val))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_loss = float('inf')
    best_state = None  # Initialize to None
    for epoch in range(n_epochs):
        model.train()
        for batch in loader:
            *x_cat, x_dense, yb = batch
            # Move to device
            x_cat = [x.to(device) for x in x_cat]
            x_dense = x_dense.to(device)
            yb = yb.to(device)
            pred = model(x_cat, x_dense)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # Validation after each epoch
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in valloader:
                *x_cat, x_dense, yb = batch
                x_cat = [x.to(device) for x in x_cat]
                x_dense = x_dense.to(device)
                yb = yb.to(device)
                pred = model(x_cat, x_dense)
                val_losses.append(loss_fn(pred, yb).item())
        val_loss = np.mean(val_losses)
        if verbose:
            print(f"Epoch {epoch+1}: val MSE {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
    # Load best model (or use current if no improvement)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
