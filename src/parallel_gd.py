import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import time
import copy

def worker(model_state, dataset, return_dict, idx, epochs, lr):

    from model import SimpleModel

    model = SimpleModel()
    model.load_state_dict(model_state)

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

    model.train()

    for epoch in range(epochs):

        for x, y in loader:

            optimizer.zero_grad()

            output = model(x)

            loss = loss_fn(output, y)

            loss.backward()

            optimizer.step()

    return_dict[idx] = copy.deepcopy(model.state_dict())

def average_models(model, state_dicts):

    avg_state = model.state_dict()

    for key in avg_state.keys():

        avg_state[key] = torch.stack(
            [state_dict[key].float() for state_dict in state_dicts],
            dim=0
        ).mean(dim=0)

    model.load_state_dict(avg_state)

    return model

def train_parallel(model, subsets, epochs=2, lr=0.01):

    manager = mp.Manager()

    return_dict = manager.dict()

    processes = []

    start_time = time.time()

    model_state = copy.deepcopy(model.state_dict())

    for i, subset in enumerate(subsets):

        p = mp.Process(
            target=worker,
            args=(
                model_state,
                subset,
                return_dict,
                i,
                epochs,
                lr
            )
        )

        p.start()

        processes.append(p)

    for p in processes:
        p.join()

    state_dicts = list(return_dict.values())

    model = average_models(model, state_dicts)

    end_time = time.time()

    total_time = end_time - start_time

    return total_time, model