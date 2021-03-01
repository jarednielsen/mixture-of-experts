import torch
from torch import nn
from mixture_of_experts import HeirarchicalMoE, MoE

# from transformers import
from transformers import AutoTokenizer
from datasets import load_dataset


embedding_size = 512

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class EmbeddingMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(tokenizer.vocab_size, embedding_size)
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
    def forward(self, x):
        outputs = self.embedding(x)
        outputs, aux_loss1 = self.moe1(outputs)
        outputs, aux_loss2 = self.moe2(outputs)
        aux_loss = aux_loss1 + aux_loss2
        return outputs, aux_loss

wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")
wikitext_train = wikitext["train"]
tokenized_dataset = wikitext_train.map(lambda x: tokenizer(
    x["text"], max_length=512, padding="max_length"
), batched=True)

def get_batch(size):
    return torch.Tensor(tokenized_dataset[:size]["input_ids"]).long()


model = EmbeddingMoE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

with torch.autograd.detect_anomaly():
    for i in range(5):
        batch = get_batch(5)
        optimizer.zero_grad()
        out, aux_loss = model(batch)
        optimizer.zero_grad()
        aux_loss.backward()
        optimizer.step()
        print(f"Batch {i}, loss {aux_loss}")

# import pdb; pdb.set_trace()






pass