from model import GCNNModel
import torch
import torch.nn as nn
from deepchem.molnet import load_qm9
from deepchem.models import TorchModel
import torch.optim as optim
import deepchem as dc
from rdkit import Chem
# Load QM9 dataset
def load_data():
    tasks, datasets, transformers = load_qm9(featurizer='GraphConv', split='random')
    def is_valid(mol):
        return mol is not None and Chem.SanitizeMol(mol, catchErrors=True) == 0

    train_dataset = train_dataset.select([is_valid(mol) for mol in train_dataset.X])
    valid_dataset = valid_dataset.select([is_valid(mol) for mol in valid_dataset.X])
    test_dataset = test_dataset.select([is_valid(mol) for mol in test_dataset.X])
    return (train_dataset, valid_dataset, test_dataset), transformers
datasets, transformers = load_data()
train_dataset, valid_dataset, test_dataset = datasets
    
model = TorchModel(GCNNModel(), loss=nn.MSELoss(), optimizer=optim.Adam)
model.fit(train_dataset, nb_epoch=10)
    
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
print("Validation score:", model.evaluate(valid_dataset, [metric], transformers))
