import torch
from torch.utils.data import DataLoader
import time

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def train_sequential(model, dataset, epochs=2, lr=0.01):

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    model = model.to(device)

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

            # MOVE DATA TO GPU
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(x)

            loss = loss_fn(output, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1} Loss: {total_loss:.4f}"
        )

    end_time = time.time()

    total_time = end_time - start_time

    return total_time