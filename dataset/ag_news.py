from datasets import load_dataset
import torch.nn
from transformers import BertTokenizerFast

class ag_news(torch.nn.Module):
	def __init__(self, split: str = "train"):
		super().__init__()
		self.dataset_name = "ag_news"
		self.dataset = load_dataset(self.dataset_name)
		
		if split == "train":
			self.dataset_split = self.dataset[split][:10000]
		else :
			self.dataset_split = self.dataset[split][:1000]

	def __getitem__(self, index):
		text = self.dataset_split["text"][index]
		label = self.dataset_split["label"][index]
		return text, label

	def __len__(self):
		return len(self.dataset_split["text"])

def ag_news_collate_fn(batch, tokenizer):
	input = [item[0] for item in batch]
	input = tokenizer(input, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
	label = torch.tensor([item[1] for item in batch])
	return input, label

class ag_news_tok(torch.nn.Module):
	def __init__(self, split: str = "train"):
		super().__init__()
		self.dataset_name = "ag_news"
		self.dataset = load_dataset(self.dataset_name)
		self.dataset = self.dataset.shuffle()
		if split == "train":
			self.dataset_split = self.dataset[split][:10000]
		else :
			self.dataset_split = self.dataset[split][:1000]
		self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

	def __getitem__(self, index):
		text = self.dataset_split["text"][index]
		label = self.dataset_split["label"][index]
		# text = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
		return text, label

	def __len__(self):
		return len(self.dataset_split["text"])

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
def ag_news_collate_fn_parallel(batch):
	input = [item[0] for item in batch]
	
	input_tensor = tokenizer(input, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
	# input_tensor = {}
	# for key in input[0].keys():
	# 	input_tensor[key] = torch.cat([item[key] for item in input])
	label = torch.tensor([item[1] for item in batch])
	return input_tensor, label

if __name__ == '__main__':
	dataset = ag_news_tok()
	print(len(dataset))
	print(dataset.__getitem__(0))
