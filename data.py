import json
import random
from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def set_seed(seed=42):
    """Seed Python, NumPy (via torch) and PyTorch for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_data(filename):
    with open(filename, "r") as f:
        raw_data = json.load(f)
    data = []
    for entry in raw_data:
        # When an entry has several gold queries, use the shortest as the
        # template the model is trained to predict.
        template = sorted(entry["sql"], key=lambda x: (len(x), x))[0]
        for sentence in entry["sentences"]:
            tokens = sentence["text"].split()
            tags = ["O"] * len(tokens)
            complete = template
            for variable, value in sentence["variables"].items():
                idx = tokens.index(variable)
                value_tokens = value.split()
                tokens = tokens[:idx] + value_tokens + tokens[idx + 1:]
                tags = tags[:idx] + [variable] * \
                    len(value_tokens) + tags[idx + 1:]
                complete = complete.replace(variable, value)
            data.append({"input": tokens, "tags": tags,
                        "template": template, "complete": complete})
    return data


def get_features(data):
    X, tags, templates, complete = [], [], [], []
    for item in data:
        X.append(item["input"])
        tags.append(item["tags"])
        templates.append(item["template"])
        complete.append(item["complete"])
    return X, tags, templates, complete


def build_vocab(sentences, min_freq=1):
    counter = Counter(word for sentence in sentences for word in sentence)
    special_tokens = ["<PAD>", "<UNK>"]
    words = [w for w, c in counter.items() if c >= min_freq]
    return {token: idx for idx, token in enumerate(special_tokens + words)}


def encode(sentence, vocab):
    if not isinstance(sentence, list):
        sentence = sentence.split()
    return [vocab.get(token, vocab["<UNK>"]) for token in sentence]


def collate_fn(batch):
    sentences, tags, templates = zip(*batch)
    sentences_padded = pad_sequence(
        sentences, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=-1)
    return sentences_padded, tags_padded, torch.stack(templates)


class SQLDataset(Dataset):
    def __init__(self, sentences, tags, templates, vocab, tag2idx, temp2idx):
        self.sentences = sentences
        self.tags = tags
        self.templates = templates
        self.vocab = vocab
        self.tag2idx = tag2idx
        self.temp2idx = temp2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = torch.tensor(encode(self.sentences[idx], self.vocab))
        tags = torch.tensor([self.tag2idx[t] for t in self.tags[idx]])
        template = torch.tensor(self.temp2idx[self.templates[idx]])
        return sentence, tags, template
