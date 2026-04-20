import json
from collections import Counter
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from tqdm import tqdm


def read_data(filename):
    with open(filename, 'r') as file:
        raw_data = json.load(file)
    data = []
    for dict in raw_data:
        templates = dict["sql"]
        templates_sorted = sorted(templates, key=lambda x: (len(x), x))
        template = templates_sorted[0]
        for sentence in dict['sentences']:
            text = sentence['text']
            tokens = text.split()
            tags = ['O'] * len(tokens)
            complete = template
            for variable, value in sentence['variables'].items():
                idx = tokens.index(variable)
                value_tokens = value.split()
                tokens = tokens[:idx] + value_tokens + tokens[idx+1:]
                tags = tags[:idx] + [variable] * \
                    len(value_tokens) + tags[idx+1:]
                complete = complete.replace(variable, value)
            item = {'input': tokens, 'tags': tags,
                    'template': template, 'complete': complete}
            data.append(item)
    return data


def get_features(data):
    X, tags, templates, complete = [], [], [], []
    for item in data:
        X.append(item['input'])
        tags.append(item['tags'])
        templates.append(item['template'])
        complete.append(item['complete'])
    return X, tags, templates, complete


def build_vocab(sentences, min_freq=1):
    counter = Counter(word for sentence in sentences for word in sentence)
    special_tokens = ["<PAD>", "<UNK>"]
    words = [w for w, c in counter.items() if c >= min_freq]
    vocab = {token: idx for idx, token in enumerate(special_tokens + words)}
    return vocab


def encode(sentence, vocab):
    if not isinstance(sentence, list):
        sentence = sentence.split()
    return [vocab.get(token, vocab["<UNK>"]) for token in sentence]


def collate_fn(batch):
    sentences, tags, templates = zip(*batch)
    lengths = torch.tensor([len(s) for s in sentences])
    sentences_padded = pad_sequence(
        sentences, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=-1)
    templates = torch.stack(templates)
    return sentences_padded, tags_padded, templates, lengths


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


class LSTM_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim_tag, output_dim_sql):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            layer_dim, batch_first=True)
        self.fc_tag = nn.Linear(hidden_dim, output_dim_tag)
        self.fc_sql = nn.Linear(hidden_dim, output_dim_sql)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        tag_out = self.fc_tag(out)
        sql_out = self.fc_sql(hn[-1])
        return tag_out, sql_out, hn, cn


def train(model, dataloader, optimizer, scheduler, criterion_tags,
          criterion_template, num_epochs, device, dev_sentences, complete_dev,
          vocab, idx2tag, idx2template):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = total_loss_tags = total_loss_template = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for sentences, tags, templates, lengths in progress_bar:
            sentences = sentences.to(device)
            tags = tags.to(device)
            templates = templates.to(device)

            optimizer.zero_grad()
            per_word_preds, template_pred, hn, cn = model(sentences)

            loss_tags = criterion_tags(
                per_word_preds.view(-1, model.fc_tag.out_features), tags.view(-1))
            loss_template = criterion_template(template_pred, templates)
            loss = loss_tags + 5*loss_template

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_loss_tags += loss_tags.item()
            total_loss_template += loss_template.item()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "tag_loss": f"{loss_tags.item():.4f}",
                "sql_loss": f"{loss_template.item():.4f}"
            })

        dev_acc = evaluate(model, dev_sentences, complete_dev,
                           vocab, idx2tag, idx2template, device)
        avg_loss = total_loss / len(dataloader)
        avg_loss_tags = total_loss_tags / len(dataloader)
        avg_loss_template = total_loss_template / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Tag Loss: {avg_loss_tags:.4f} | "
              f"SQL Loss: {avg_loss_template:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Val acc: {dev_acc:.6f}")


def evaluate(model, dev_sentences, complete_dev, vocab, idx2tag, idx2template, device):
    complete_pred = predict_complete(
        model, dev_sentences, vocab, idx2tag, idx2template, device)
    accuracy = sum(pred == true for pred, true in zip(
        complete_pred, complete_dev)) / len(complete_pred)
    return accuracy


def evaluate_detailed(model, dev_sentences, tags_dev, templates_dev, vocab, idx2tag, idx2template, device):
    template_correct = 0
    tag_correct = 0
    tag_total = 0

    model.eval()
    for sentence, true_tags, true_template in zip(dev_sentences, tags_dev, templates_dev):
        with torch.no_grad():
            tensor = torch.tensor(encode(sentence, vocab)
                                  ).unsqueeze(0).to(device)
            tag_per_word, template_pred, _, _ = model(tensor)

        pred_template = idx2template[torch.argmax(
            template_pred.squeeze(0), dim=-1).item()]
        pred_tags = [idx2tag[i.item()]
                     for i in torch.argmax(tag_per_word.squeeze(0), dim=-1)]

        if pred_template == true_template:
            template_correct += 1

        for pred_tag, true_tag in zip(pred_tags, true_tags):
            tag_correct += pred_tag == true_tag
            tag_total += 1

    template_acc = template_correct / len(dev_sentences)
    tag_acc = tag_correct / tag_total
    return template_acc, tag_acc


def predict_complete(model, sentence, vocab, idx2tag, idx2template, device):
    if isinstance(sentence, list):
        return [predict_complete(model, sent, vocab, idx2tag, idx2template, device) for sent in sentence]

    model.eval()
    with torch.no_grad():
        encoded = encode(sentence, vocab)
        tensor = torch.tensor(encoded).unsqueeze(0).to(device)
        tag_per_word, template_pred, hn, cn = model(tensor)

    tags_pred = torch.argmax(tag_per_word.squeeze(0), dim=-1)
    template_idx = torch.argmax(template_pred.squeeze(0), dim=-1).item()
    template = idx2template[template_idx]
    tags = [idx2tag[idx.item()] for idx in tags_pred]

    tag_indices = {}
    for i, tag in enumerate(tags):
        if tag != 'O':
            tag_indices.setdefault(tag, []).append(i)

    for tag, indices in tag_indices.items():
        if len(indices) >= 2:
            if all(indices[i] + 1 == indices[i+1] for i in range(len(indices)-1)):
                variable = sentence.split()[indices[0]:indices[-1]+1]
                template = template.replace(tag, " ".join(variable))
            else:
                max_idx = tags_pred[indices].argmax()
                variable = sentence.split()[max_idx]
                template = template.replace(tag, variable)
        elif len(indices) == 1:
            variable = sentence.split()[indices[0]]
            template = template.replace(tag, variable)

    return template


def main_LSTM():
    train_data = read_data('data/geography.train.json')
    dev_data = read_data('data/geography.dev.json')

    x_train, tags_train, templates_train, complete_train = get_features(
        train_data)
    x_dev, tags_dev, templates_dev, complete_dev = get_features(dev_data)

    vocab = build_vocab(x_train, 1)

    all_tags = set(tag for tags in tags_train for tag in tags)
    all_templates = set(templates_train)

    tag2idx = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
    temp2idx = {template: idx for idx,
                template in enumerate(sorted(all_templates))}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    idx2template = {idx: template for template, idx in temp2idx.items()}

    train_dataset = SQLDataset(
        x_train, tags_train, templates_train, vocab, tag2idx, temp2idx)
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    vocab_size = len(vocab)
    embedding_dim = 128 * 2 * 2 * 2
    hidden_dim = 128 * 2 * 2 * 2
    layer_dim = 1
    output_dim_tag = len(tag2idx)
    output_dim_sql = len(temp2idx)
    epochs = 25
    step_size = 5
    gamma = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_model(vocab_size, embedding_dim, hidden_dim,
                       layer_dim, output_dim_tag, output_dim_sql).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)
    criterion_tags = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_template = nn.CrossEntropyLoss()

    dev_sentences = [" ".join(x) for x in x_dev]
    train(model, train_loader, optimizer, scheduler, criterion_tags, criterion_template,
          num_epochs=epochs, device=device, dev_sentences=dev_sentences,
          complete_dev=complete_dev, vocab=vocab, idx2tag=idx2tag, idx2template=idx2template)

    complete_pred = predict_complete(
        model, dev_sentences, vocab, idx2tag, idx2template, device)
    complete_acc = sum(pred == true for pred, true in zip(
        complete_pred, complete_dev)) / len(complete_pred)

    template_acc, tag_acc = evaluate_detailed(
        model, dev_sentences, tags_dev, templates_dev,
        vocab, idx2tag, idx2template, device
    )
    print("\n Results for LSTM model:")
    print(f"\n--- Dev Set Evaluation ---")
    print(f"Complete SQL Accuracy:  {complete_acc:.4f}")
    print(f"Template Accuracy:      {template_acc:.4f}")
    print(f"Tag Accuracy:           {tag_acc:.4f}")

    sentences = []
    with open("data/test.txt", 'r') as file:
        for line in file:
            sentences.append(line.strip())

    complete_sql = predict_complete(
        model, sentences, vocab, idx2tag, idx2template, device)
    with open("q5_predictions.txt", 'w') as file:
        for sent, sql in zip(sentences, complete_sql):
            file.write(sent + "|" + sql + "\n")
