import torch
import torch.nn as nn
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1536, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out


def split_samples(df):
    test_data = df.sample(frac=0.2)
    cond = df['chunk_id'].isin(test_data['chunk_id'])
    df = df.drop(df[cond].index)
    return df, test_data


def train_model(model, num_epochs, samples, verbose=True):
    # Split into train and validation
    train_df, test_df = split_samples(samples)
    dataset_sizes = {"train": len(train_df), "validation": len(test_df)}
    token_total = sum(samples["length"])
    if verbose:
        print(f"Training with: {len(train_df)} samples\nValidating with: {len(test_df)} samples")

    # Set up training objects
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(num_epochs), desc="Epochs", disable=not verbose):
        if verbose:
            # print("EPOCH", epoch)
            pass
        # TRAIN
        model.train()
        for embedding, token in zip(train_df["embedding"], train_df["token_fraction"]):
            output = model(torch.FloatTensor(embedding))
            loss = criterion(output.squeeze(), torch.FloatTensor([token]).squeeze())
            # print("Prediction:", output.squeeze(), "Truth:", torch.FloatTensor([token / len(samples)]).squeeze())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # VALIDATE
        model.eval()
        val_loss = 0.0
        with torch.no_grad():  # No need to track gradients
            for embedding, token in tqdm(zip(test_df["embedding"], test_df["token_fraction"]), desc="Validation",
                                         disable=verbose):
                output = model(torch.FloatTensor(embedding))
                loss = criterion(output.squeeze(), torch.FloatTensor([token]).squeeze())
                # print("Prediction:", output.squeeze(), "Truth:", torch.FloatTensor([token / len(samples)]).squeeze())
                # print("Loss:", loss.item())
                # print("Error (in tokens):", loss.item() * token_total)
                val_loss += loss.item()

        # NEW TRAIN TEST SPLIT
        train_df, test_df = split_samples(samples)

    if verbose:
        print("AVERAGE VALIDATION LOSS:", val_loss / dataset_sizes["validation"], "\n")
        print("AVERAGE VALIDATION ERROR (in tokens):", (val_loss / dataset_sizes["validation"]) * token_total, "\n")

    return model


def retrieve_node(window_size, tokens, query_embedding, model):
    # GENERATE PREDICTED TOKEN
    model.eval()
    predicted_token_fraction = model(torch.FloatTensor(query_embedding)).item()
    print("Model Output:", predicted_token_fraction)
    tokens_length = len(tokens)
    target_index = round(predicted_token_fraction * tokens_length)
    print("Predicted Location:", target_index)

    # RETRIEVE PREDICTED CHUNK
    trailing_chunk = " ".join(tokens[target_index:target_index + window_size])

    return trailing_chunk
