import time
import torch
import torch.nn as nn
import copy
import csv
from load_data import load_data
from tqdm import tqdm


def train_model(model, num_epochs, device, data_filename, verbose=True, prev_state=None):
    # Load data
    dataloaders = load_data(data_filename, device)
    dataset_sizes = {"train": len(dataloaders["train"]), "validation": len(dataloaders["validation"])}

    # Load Model Weights
    if prev_state is not None:
        model.load_state_dict(torch.load(prev_state))

    # Set up training objects
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        if verbose:
            print("EPOCH",epoch)
        # TRAIN
        model.train()
        for embedding, token in tqdm(dataloaders["train"], desc="Training", disable=not verbose):
            output = model(embedding)
            loss = criterion(output.squeeze(), token.squeeze())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # VALIDATE
        model.eval()
        val_loss = 0.0
        with torch.no_grad():  # No need to track gradients
            for embedding, token in tqdm(dataloaders["validation"], desc="Validation", disable=not verbose):
                output = model(embedding)
                loss = criterion(output.squeeze(), token.squeeze())
                val_loss += loss.item()
        if verbose:
            print("VALIDATION ACCURACY:", val_loss / dataset_sizes["validation"],"\n")

    return model



