from transformers import BertModel, BertTokenizerFast
from torch import nn
import torch
torch.manual_seed(0)

class BertFinetuning(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, **x):
        x = self.bert(**x).pooler_output
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    model = BertFinetuning()
    
    input = ["Hello, my dog", "Hello world, I am still alive"]
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    input = tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output = model(input)
    print(output.shape)