# Training and testing functions for the model
import tqdm
import numpy as np
import torch
import torchmetrics

def train(model, dataloader, test_dataloader, optim, loss_fn, n_epochs, device):
    model.train()
    epoch_losses = []
    test_epoch_losses = []
    n_iter = 0
    for epoch in tqdm.tqdm(range(n_epochs)):
        losses = []
        for input, label in tqdm.tqdm(dataloader, total = len(dataloader)):
            n_iter += 1
            for key in input:
                input[key] = input[key].to(device)
            label = label.to(device)

            optim.zero_grad()
            output = model(**input)
            loss = loss_fn(output, label)
            loss.backward()

            losses.append(loss.item())
            if n_iter % 10 == 0:
                print("Iteration: {}, Loss: {}".format(n_iter, loss.item()))
            optim.step()
        
        epoch_losses.append(np.mean(losses))
        print("Epoch: {}, Train Loss: {}".format(epoch, np.mean(losses)))
        
        test_epoch_losses.append(test(model, test_dataloader, loss_fn, device))
        print("Epoch: {}, Test Loss: {}".format(epoch, test_epoch_losses[-1]))

        if epoch % 5 == 0:
            torch.save(model.state_dict(), "Checkpoints/model{}.pt".format(epoch))
    return epoch_losses, test_epoch_losses

def test(model, dataloader, loss_fn, device):
    model.eval()
    losses = []
    # f1 = torchmetrics.F1Score(task = 'multiclass', num_labels = 4)

    for input, label in tqdm.tqdm(dataloader, total = len(dataloader)):
        for key in input:
            input[key] = input[key].to(device)
        output = model(**input)
        label = label.to(device)
        loss = loss_fn(output, label)
        losses.append(loss.item())
    return np.mean(losses)