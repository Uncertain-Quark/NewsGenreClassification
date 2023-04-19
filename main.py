from dataset.ag_news import ag_news_tok, ag_news_collate_fn_parallel
from models.bert import BertFinetuning
from torch.utils.data import DataLoader
from train import train, test
import torch
from transformers import BertTokenizerFast
import matplotlib.pyplot as plt

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    train_dataset = ag_news_tok()
    train_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn=ag_news_collate_fn_parallel, num_workers = 8)
    
    test_dataset = ag_news_tok(split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=128, collate_fn=ag_news_collate_fn_parallel, num_workers = 8)

    model = BertFinetuning()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.to(device)

    n_epochs = 500

    # Train
    train_epoch_loss, test_loss = train(model, train_dataloader, test_dataloader, optimizer, loss_fn, n_epochs, device)

    fig = plt.figure(0)
    plt.plot(train_epoch_loss, label="Train")
    plt.plot(test_loss, label="Test")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.grid()
    plt.savefig("loss.png")
    