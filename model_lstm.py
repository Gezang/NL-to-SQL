from torch import nn
from torch.utils.data import DataLoader
import torch

from data import (set_seed, read_data, get_features, build_vocab, encode,
                  collate_fn, SQLDataset)


class LSTMTagger(nn.Module):
    """Sequence-labelling model with two heads (per-word tags and SQL template).

    Setting ``bidirectional=True`` makes it a BiLSTM; otherwise a plain LSTM.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers,
                 num_tags, num_templates, bidirectional=False, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=bidirectional)
        # The tag/template heads read from num_directions * hidden_dim because
        # forward and backward states are concatenated when bidirectional.
        out_dim = self.num_directions * hidden_dim
        self.fc_tag = nn.Linear(out_dim, num_tags)
        self.fc_sql = nn.Linear(out_dim, num_templates)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        num_states = self.num_directions * self.num_layers
        h0 = torch.zeros(num_states, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros_like(h0)
        out, (hn, _) = self.lstm(self.dropout(self.embedding(x)), (h0, c0))
        if self.bidirectional:
            # hn[-2] = last forward layer, hn[-1] = last backward layer.
            final_hidden = torch.cat((hn[-2], hn[-1]), dim=-1)
        else:
            final_hidden = hn[-1]
        return self.fc_tag(self.dropout(out)), self.fc_sql(self.dropout(final_hidden))


def _fill_template(template, tags, tags_tensor, sentence):
    tokens = sentence.split()
    tag_indices = {}
    for i, tag in enumerate(tags):
        if tag != "O":
            tag_indices.setdefault(tag, []).append(i)

    for tag, indices in tag_indices.items():
        if len(indices) >= 2:
            if all(indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1)):
                template = template.replace(tag, " ".join(
                    tokens[indices[0]:indices[-1] + 1]))
            else:
                max_idx = indices[tags_tensor[indices].argmax().item()]
                template = template.replace(tag, tokens[max_idx])
        elif len(indices) == 1:
            template = template.replace(tag, tokens[indices[0]])
    return template


def _predict_single(model, sentence, vocab, idx2tag, idx2template, device):
    tensor = torch.tensor(encode(sentence, vocab)).unsqueeze(0).to(device)
    tag_per_word, template_pred = model(tensor)
    tags_tensor = torch.argmax(tag_per_word.squeeze(0), dim=-1)
    template = idx2template[torch.argmax(
        template_pred.squeeze(0), dim=-1).item()]
    tags = [idx2tag[i.item()] for i in tags_tensor]
    return template, tags, tags_tensor


def predict_complete(model, sentence, vocab, idx2tag, idx2template, device):
    if isinstance(sentence, list):
        return [predict_complete(model, s, vocab, idx2tag, idx2template, device) for s in sentence]
    model.eval()
    with torch.no_grad():
        pred_template, pred_tags, tags_tensor = _predict_single(
            model, sentence, vocab, idx2tag, idx2template, device)
    return _fill_template(pred_template, pred_tags, tags_tensor, sentence)


def evaluate_all(model, dev_sentences, tags_dev, templates_dev, complete_dev,
                 vocab, idx2tag, idx2template, device):
    template_correct = tag_correct = complete_correct = tag_total = 0

    model.eval()
    with torch.no_grad():
        for sentence, true_tags, true_template, true_complete in zip(
                dev_sentences, tags_dev, templates_dev, complete_dev):
            pred_template, pred_tags, tags_tensor = _predict_single(
                model, sentence, vocab, idx2tag, idx2template, device)

            template_correct += pred_template == true_template
            for pt, tt in zip(pred_tags, true_tags):
                tag_correct += pt == tt
                tag_total += 1

            pred_complete = _fill_template(
                pred_template, pred_tags, tags_tensor, sentence)
            complete_correct += pred_complete == true_complete

    n = len(dev_sentences)
    return complete_correct / n, template_correct / n, tag_correct / tag_total


def train(model, dataloader, optimizer, criterion_tags, criterion_template,
          num_epochs, device, template_loss_weight, dev_sentences, tags_dev,
          templates_dev, complete_dev, vocab, idx2tag, idx2template):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = total_loss_tags = total_loss_template = 0

        for sentences, tags, templates in dataloader:
            sentences, tags, templates = sentences.to(
                device), tags.to(device), templates.to(device)

            optimizer.zero_grad()
            per_word_preds, template_pred = model(sentences)

            loss_tags = criterion_tags(
                per_word_preds.view(-1, model.fc_tag.out_features), tags.view(-1))
            loss_template = criterion_template(template_pred, templates)
            loss = loss_tags + template_loss_weight * loss_template

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_loss_tags += loss_tags.item()
            total_loss_template += loss_template.item()

        complete_acc, _, _ = evaluate_all(
            model, dev_sentences, tags_dev, templates_dev, complete_dev,
            vocab, idx2tag, idx2template, device)

        n = len(dataloader)
        print(f"  Epoch {epoch + 1:02d}/{num_epochs} | "
              f"Loss: {total_loss / n:.4f}  "
              f"Tag: {total_loss_tags / n:.4f}  "
              f"SQL: {total_loss_template / n:.4f}  |  "
              f"Val: {complete_acc:.4f}")


def run_lstm(bidirectional, embedding_dim, hidden_dim, num_layers, dropout,
             template_loss_weight, num_epochs, batch_size, lr,
             predictions_path, title):
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

    set_seed()

    train_data = read_data("data/geography.train.json")
    dev_data = read_data("data/geography.dev.json")

    x_train, tags_train, templates_train, _ = get_features(train_data)
    x_dev, tags_dev, templates_dev, complete_dev = get_features(dev_data)

    vocab = build_vocab(x_train)

    all_tags = sorted({tag for tags in tags_train for tag in tags})
    all_templates = sorted(set(templates_train))
    tag2idx = {tag: idx for idx, tag in enumerate(all_tags)}
    temp2idx = {tmpl: idx for idx, tmpl in enumerate(all_templates)}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    idx2template = {idx: tmpl for tmpl, idx in temp2idx.items()}

    train_dataset = SQLDataset(
        x_train, tags_train, templates_train, vocab, tag2idx, temp2idx)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMTagger(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_tags=len(tag2idx),
        num_templates=len(temp2idx),
        bidirectional=bidirectional,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_tags = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_template = nn.CrossEntropyLoss()

    dev_sentences = [" ".join(x) for x in x_dev]
    train(model, train_loader, optimizer, criterion_tags, criterion_template,
          num_epochs=num_epochs, device=device,
          template_loss_weight=template_loss_weight,
          dev_sentences=dev_sentences, tags_dev=tags_dev,
          templates_dev=templates_dev, complete_dev=complete_dev,
          vocab=vocab, idx2tag=idx2tag, idx2template=idx2template)

    complete_acc, template_acc, tag_acc = evaluate_all(
        model, dev_sentences, tags_dev, templates_dev, complete_dev,
        vocab, idx2tag, idx2template, device)

    print()
    print("  Dev Set Results")
    print("  " + "-" * 30)
    print(f"  Complete SQL Accuracy : {complete_acc:.4f}")
    print(f"  Template Accuracy     : {template_acc:.4f}")
    print(f"  Tag Accuracy          : {tag_acc:.4f}")
    print("=" * 60)

    test_sentences = [line.strip() for line in open("data/test.txt")]
    complete_sql = predict_complete(
        model, test_sentences, vocab, idx2tag, idx2template, device)
    with open(predictions_path, "w") as f:
        for sent, sql in zip(test_sentences, complete_sql):
            f.write(sent + "|" + sql + "\n")


def main_lstm():
    run_lstm(
        bidirectional=False,
        embedding_dim=512,
        hidden_dim=512,
        num_layers=1,
        dropout=0.0,
        template_loss_weight=5,
        num_epochs=25,
        batch_size=32,
        lr=0.001,
        predictions_path="results/predictions_lstm.txt",
        title="LSTM Model",
    )


def main_bilstm():
    run_lstm(
        bidirectional=True,
        embedding_dim=512 * 2,
        hidden_dim=512 * 2,
        num_layers=2,
        dropout=0.3,
        template_loss_weight=10,
        num_epochs=25,
        batch_size=32,
        lr=0.001,
        predictions_path="results/predictions_bilstm.txt",
        title="BiLSTM Model",
    )
