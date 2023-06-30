import torch
import torch.nn as nn
from torchvision import transforms
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

# Circuit imports
import sys

import torchvision

sys.path.append(os.path.join(sys.path[0], "hmc-utils"))
sys.path.append(os.path.join(sys.path[0], "hmc-utils", "pypsdd"))
sys.path.append(os.path.join(sys.path[0], "dataset"))

from GatingFunction import DenseGatingFunction
from compute_mpe import CircuitMPE
from pysdd.sdd import SddManager, Vtree

# misc
from common import *
from dataset import load_dataloaders

# variables
dataset_name = "mnist"
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

dataloaders = load_dataloaders(img_size=28, img_depth=1, device=device_str)

train = dataloaders["train"]
test = dataloaders["test"]
A = dataloaders["train_set"].get_A()
S = 0
hidden_dim = 200
gates = 1
num_reps = 1

# Set the hyperparameters
hyperparams = {
    "num_layers": 3,
    "dropout": 0.7,
    "non_lin": "relu",
}

# MLP
class ConstrainedFFNNModel(nn.Module):
    """C-HMCNN(h) model - during training it returns the not-constrained output that is then passed to MCLoss"""

    def __init__(self, input_dim, hidden_dim, output_dim, hyperparams, R):
        super(ConstrainedFFNNModel, self).__init__()

        self.nb_layers = hyperparams["num_layers"]
        self.R = R

        fc = []
        for i in range(self.nb_layers):
            if i == 0:
                fc.append(nn.Linear(input_dim, hidden_dim))
            elif i == self.nb_layers - 1:
                fc.append(nn.Linear(hidden_dim, output_dim))
            else:
                fc.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc = nn.ModuleList(fc)

        self.drop = nn.Dropout(hyperparams["dropout"])

        if hyperparams["non_lin"] == "tanh":
            self.f = nn.Tanh()
        else:
            self.f = nn.ReLU()

    def forward(self, x, sigmoid=False, log_sigmoid=False):
        x = x.view(-1, 28 * 28 * 1)
        for i in range(self.nb_layers):
            if i == self.nb_layers - 1:
                if sigmoid:
                    x = nn.Sigmoid()(self.fc[i](x))
                elif log_sigmoid:
                    x = nn.LogSigmoid()(self.fc[i](x))
                else:
                    x = self.fc[i](x)
            else:
                x = self.f(self.fc[i](x))
                x = self.drop(x)

        if self.R is None:
            return x

        if self.training:
            constrained_out = x
        else:
            constrained_out = get_constr_out(x, self.R)
        return constrained_out


if not os.path.isfile("constraints/" + dataset_name + ".sdd") or not os.path.isfile(
    "constraints/" + dataset_name + ".vtree"
):
    # Compute matrix of ancestors R
    # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is ancestor of class j
    R = np.zeros(A.shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(A)
    for i in range(len(A)):
        descendants = list(nx.descendants(g, i))
        if descendants:
            R[i, descendants] = 1
    R = torch.tensor(R)

    # Transpose to get the ancestors for each node
    R = R.unsqueeze(0).to(device)

    # Uncomment below to compile the constraint
    R.squeeze_()
    mgr = SddManager(var_count=R.size(0), auto_gc_and_minimize=True)

    alpha = mgr.true()
    alpha.ref()
    for i in range(R.size(0)):

        beta = mgr.true()
        beta.ref()
        for j in range(R.size(0)):

            if R[i][j] and i != j:
                old_beta = beta
                beta = beta & mgr.vars[j + 1]
                beta.ref()
                old_beta.deref()

        old_beta = beta
        beta = -mgr.vars[i + 1] | beta
        beta.ref()
        old_beta.deref()

        old_alpha = alpha
        alpha = alpha & beta
        alpha.ref()
        old_alpha.deref()

    # Saving circuit & vtree to disk
    alpha.save(str.encode("constraints/" + dataset_name + ".sdd"))
    alpha.vtree().save(str.encode("constraints/" + dataset_name + ".vtree"))

    # Create circuit object
    cmpe = CircuitMPE(
        "constraints/" + dataset_name + ".vtree", "constraints/" + dataset_name + ".sdd"
    )

    if S > 0:
        cmpe.overparameterize(S=S)
        print("Done overparameterizing")

    # Create gating function
    gate = DenseGatingFunction(
        cmpe.beta, gate_layers=[67] + [67] * gates, num_reps=num_reps
    ).to(device)
    R = None

    print("CMPE e DenseGating function prepared...")

else:
    # Use fully-factorized sdd
    mgr = SddManager(var_count=A.shape[0], auto_gc_and_minimize=True)
    alpha = mgr.true()
    vtree = Vtree(var_count=A.shape[0], var_order=list(range(1, A.shape[0] + 1)))
    alpha.save(str.encode("ancestry.sdd"))
    vtree.save(str.encode("ancestry.vtree"))
    cmpe = CircuitMPE("ancestry.vtree", "ancestry.sdd")
    cmpe.overparameterize()

    # Gating function
    gate = DenseGatingFunction(cmpe.beta, gate_layers=[67]).to(device)
    R = None

# Create the model
model = ConstrainedFFNNModel(28 * 28 * 1, hidden_dim, 67, hyperparams, R)
model = model.to(device)

# Load MNIST dataset
train_loader = dataloaders["train_loader"]

# Obtain a batch of data
data, target = next(iter(train_loader))
data = data.to(device)

# Set requires_grad to True for data tensor
data.requires_grad_(True)

model.eval()
gate.eval()

# Forward pass
outputs = model(data.float())
thetas = gate(outputs.float())
cmpe.set_params(thetas)
# marginals
log_prob_ys = cmpe.get_marginals()[1:67]

# detect anomaly
torch.autograd.set_detect_anomaly(True)

gradXes = torch.autograd.grad(
    log_prob_ys,
    data,
    torch.ones_like(log_prob_ys),
)[0]

# Print the input gradients
print("Input Gradients:")
print(gradXes[0])
