import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from mixture_of_experts import HeirarchicalMoE, MoE

from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import wandb

embedding_size = 512

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class EmbeddingMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(tokenizer.vocab_size, embedding_size)
        # HierarchicalMoE leads to the variable overwriting error in cell 33 here: https://github.com/t-vi/acdl2020/blob/master/pytorch_introduction_slides.pdf
        # self.moe = HeirarchicalMoE(
        #     dim = embedding_size,
        #     num_experts = (4, 4),       # 4 gates on the first layer, then 4 experts on the second, equaling 16 experts
        # )
        self.moe1 = MoE(
            dim=embedding_size,
            num_experts=4,
        )
        self.moe2 = MoE(
            dim=embedding_size,
            num_experts=4,
        )
        self.mlm = nn.Linear(embedding_size, tokenizer.vocab_size)
    def forward(self, x):
        outputs = self.embedding(x)
        outputs, aux_loss1 = self.moe1(outputs)
        outputs, aux_loss2 = self.moe2(outputs)
        outputs = self.mlm(outputs)
        aux_loss = aux_loss1 + aux_loss2
        return outputs, aux_loss

class MLMDataset(Dataset):
    def __init__(self):
        wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")
        wikitext_train = wikitext["train"]
        self.tokenized_dataset = wikitext_train.map(lambda x: tokenizer(
            x["text"], max_length=512, padding="max_length"
        ), batched=True)

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        return self.tokenized_dataset[idx]

    def get_batch(self, size):
        batch = self.tokenized_dataset[:size]

        ids = torch.Tensor(batch["input_ids"]).long()

        mask = (torch.rand(ids.shape) > 0.85).long()
        mask = mask * torch.Tensor(batch["attention_mask"]).long()

        masked_ids = (mask * tokenizer.mask_token_id) + ( (1 - mask) * ids)

        return ids.to("cuda"), masked_ids.to("cuda"), mask.to("cuda")

def get_mlm_loss(preds, ids, mask):
    preds = preds.permute([0, 2, 1]) # [batch, vocab_size, seq_len]
    ids = (1 - mask) * (-100) + (mask * ids) # [batch, seq_len]
    loss = F.cross_entropy(preds, ids, ignore_index=-100, reduction="mean") # [batch, seq_len]
    return loss


def train(steps, batch_size, lr):
    device = torch.device("cuda")
    model = EmbeddingMoE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = MLMDataset()
    wandb.init(
        project="moe",
        entity="jaredtnielsen",
        config={
            "batch_size": batch_size,
            "steps": steps,
            "lr": lr,
        },
    )
    for i in tqdm(range(steps)):
        ids, masked_ids, mask = dataset.get_batch(batch_size)
        optimizer.zero_grad()
        preds, aux_loss = model(masked_ids)
        optimizer.zero_grad()
        mlm_loss = get_mlm_loss(preds, ids, mask)
        loss = mlm_loss + aux_loss
        loss.backward()
        optimizer.step()
        tqdm.write(f"Batch {i}, loss {loss:.3f}, mlm_loss {mlm_loss:.3f}, aux_loss {aux_loss:.3f}")
        wandb.log({
            "step": i,
            "loss": loss,
            "mlm_loss": mlm_loss,
            "aux_loss": aux_loss,
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(steps=args.steps, batch_size=args.batch_size, lr=args.lr)