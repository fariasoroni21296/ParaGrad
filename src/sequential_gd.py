import torch
from torch.utils.data import DataLoader
import time

def train_sequential(model, dataset, epochs=2, lr=0.01):

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    start_time = time.time()

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for x, y in loader:

            optimizer.zero_grad()

            output = model(x)

            loss = loss_fn(output, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Sequential Epoch {epoch+1} Loss: {total_loss:.4f}")

    end_time = time.time()

    total_time = end_time - start_time

    return total_time