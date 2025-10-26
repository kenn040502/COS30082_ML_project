import torch.nn as nn

def get_losses():
    return {
        "classification": nn.CrossEntropyLoss(),
        "domain": nn.BCEWithLogitsLoss()
    }
