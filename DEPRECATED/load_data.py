import csv
import torch
import numpy as np
from torch.utils.data import DataLoader


def load_data(data_filename, device):
    # Load data
    embeddings = []
    tokens = []
    with open("data/" + data_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            str_embedding = row[1:]
            float_embedding = []
            for i, y in enumerate(str_embedding):
                float_embedding.append(float(y))
            embeddings.append(float_embedding)
            tokens.append(float(row[0]))

    data = np.column_stack((tokens, embeddings))

    data_tensor = torch.tensor(
        data=data,
        dtype=torch.float,
        device=device
    )

    data_length = data_tensor.size(0)

    trainloader = torch.utils.data.DataLoader(
        [[data_tensor[i][1:], data_tensor[i][0]] for i in range(round(data_length * 0.8))], shuffle=True)
    testloader = torch.utils.data.DataLoader(
        [[data_tensor[i][1:], data_tensor[i][0]] for i in range(round(data_length * 0.8), data_length)], shuffle=True)

    dataloaders = {"train": trainloader, "validation": testloader}

    return dataloaders
