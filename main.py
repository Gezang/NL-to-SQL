from collections import Counter
import json
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from collections import Counter


def read_data(filename):
    """Read the data from a json file.

    Keyword arguments:
    filename -- the name of a json file
    """
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
            tags = ['O']*len(tokens)
            complete = template
            for variable, value in sentence['variables'].items():

                idx = tokens.index(variable)
                value_tokens = value.split()
                tokens = tokens[:idx] + value_tokens + tokens[idx+1:]
                tags = tags[:idx] + [variable]*len(value_tokens) + tags[idx+1:]
                complete = complete.replace(variable, value)

            item = {'input': tokens,
                    'tags': tags,
                    'template': template,
                    'complete': complete}
            data.append(item)

    return data


class Data_processor:
    def __init__(self, train_data, dev_data):
        self.train_data = train_data
        self.dev_data = dev_data

    def fit_transform_preprocess(self):
        X_train, tags_train, templates_train, complete_train = self.get_features(
            self.train_data)

        X_dev, tags_dev, templates_dev, complete_dev = self.get_features(
            self.dev_data)

        train_sentances = [" ".join(x) for x in X_train]
        dev_sentances = [" ".join(x) for x in X_dev]

        self.vectorizer = CountVectorizer()
        self.X_counts = self.vectorizer.fit_transform(train_sentances)

        self.transformer = TfidfTransformer()
        self.train_tfidf = self.transformer.fit_transform(self.X_counts)
        print(self.train_tfidf.shape)

        dev_counts = self.vectorizer.transform(dev_sentances)
        self.dev_tfidf = self.transformer.transform(dev_counts)

        temp_set = set(templates_train)
        self.temp_mapping = {tag: i for i, tag in
                             enumerate(temp_set)}
        self.reverse_temp_mapping = {
            i: tag for tag, i in self.temp_mapping.items()}
        self.y_templates_train = [self.temp_mapping[temp]
                                  for temp in templates_train]
        self.y_templates_dev = [self.temp_mapping.get(
            temp, -1) for temp in templates_dev]

        self.all_words_train = [
            word for sentence in X_train for word in sentence]
        self.y_tags_train = [
            tag for sentence in tags_train for tag in sentence]

        self.v = CountVectorizer()
        self.all_words_dev = [word for sentence in X_dev for word in sentence]
        self.X_tags_train = self.v.fit_transform(self.all_words_train)
        self.X_tags_dev = self.v.transform(self.all_words_dev)
        self.y_tags_dev = [tag for sentence in tags_dev for tag in sentence]

    def get_features(self, data):
        X = []
        tags = []
        templates = []
        complete = []
        for dict in data:
            X.append(dict['input'])
            tags.append(dict['tags'])
            templates.append(dict['template'])
            complete.append(dict['complete'])

        return X, tags, templates, complete

    def get_test_data(self):
        self.test_sentances = []
        with open("data/test.txt", 'r') as file:
            for line in file:
                self.test_sentances.append(line.strip())

    def transform_test_data(self):
        # For the template model:
        test_counts = self.vectorizer.transform(self.test_sentances)
        self.test_tfidf = self.transformer.transform(test_counts)


class Linear_model:
    def __init__(self, train_data, dev_data):
        self.template_model = LinearSVC()
        self.tag_model = LogisticRegression()
        self.data = Data_processor(train_data, dev_data)
        self.data.fit_transform_preprocess()

    def train_template_model(self):
        self.template_model.fit(self.data.train_tfidf,
                                self.data.y_templates_train)

    def train_tag_model(self):
        self.tag_model.fit(self.data.X_tags_train, self.data.y_tags_train)
        print(self.data.y_tags_train)

    def evaluate_template_model(self):
        dev_predictions = self.template_model.predict(self.data.dev_tfidf)
        correct = sum(1 for pred, true in zip(
            dev_predictions, self.data.y_templates_dev) if pred == true)
        total = len(self.data.y_templates_dev)
        accuracy = correct / total
        print(f"Template Prediction Accuracy: {accuracy:.2f}")

    def evaluate_tag_model(self):
        tag_predictions = self.tag_model.predict(self.data.X_tags_dev)
        tag_correct = sum(1 for pred, true in zip(
            tag_predictions, self.data.y_tags_dev) if pred == true)
        tag_total = len(self.data.y_tags_dev)
        tag_accuracy = tag_correct / tag_total
        print(f"Tag Prediction Accuracy: {tag_accuracy:.2f}")

    def predict_complete_sql(self, sentence):

        if type(sentence) == list:
            list_of_complete_sql = []
            for sent in sentence:
                list_of_complete_sql.append(self.predict_complete_sql(sent))
            return list_of_complete_sql

        template_pred = self.template_model.predict(
            self.data.transformer.transform(self.data.vectorizer.transform([sentence])))[0]
        tags_pred = self.tag_model.predict(
            self.data.v.transform(sentence.split()))

        template = self.data.reverse_temp_mapping[template_pred]

        # Check if there are 2 or more of any tags predicted
        # Record their idx and the tag itself in a dictionary
        # print("Predicted test tags:", tags_pred)
        tag_indices = {}
        for i, tag in enumerate(tags_pred):
            if tag != 'O':
                if tag not in tag_indices:
                    tag_indices[tag] = []
                tag_indices[tag].append(i)

        for tag, indices in tag_indices.items():
            if len(indices) >= 2:
                # Longer word
                if all(indices[i] + 1 == indices[i+1] for i in range(len(indices)-1)):
                    variable = sentence.split()[indices[0]:indices[-1]+1]
                    template = template.replace(tag, " ".join(variable))

                # if not consecutive, use the one with highest probability
                else:
                    probabilities = self.tag_model.predict_proba(
                        self.data.v.transform([sentence.split()[idx] for idx in indices]))
                    max_idx = indices[probabilities.argmax()]
                    variable = sentence.split()[max_idx]
                    template = template.replace(tag, variable)
            if len(indices) == 1:
                variable = sentence.split()[indices[0]]
                template = template.replace(tag, variable)
        complete_sql = template

        return complete_sql


def linear_model():
    train_data = read_data('data/geography.train.json')
    dev_data = read_data('data/geography.dev.json')

    model = Linear_model(train_data, dev_data)
    model.train_template_model()
    model.train_tag_model()
    model.evaluate_template_model()
    model.evaluate_tag_model()
    sentences = []
    with open("test.txt", 'r') as file:
        for line in file:
            sentences.append(line.strip())

    complete_sql = model.predict_complete_sql(sentences)

    with open("q4_predictions.txt", 'w') as file:
        for sent, sql in zip(sentences, complete_sql):
            # print(sent + "|" + sql)
            file.write(sent + "|" + sql + "\n")


class LSTM_model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim_tag, output_dim_sql):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # Embedding layer to convert word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            layer_dim, batch_first=True)
        self.fc_tag = nn.Linear(hidden_dim, output_dim_tag)
        self.fc_sql = nn.Linear(hidden_dim, output_dim_sql)
        # self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(
                0), self.hidden_dim).to(x.device)

        x = self.embedding(x)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Apply the tag prediction to each time step
        tag_out = self.fc_tag(out)
        # Apply the SQL prediction to the last time step
        sql_out = self.fc_sql(hn[-1])
        return tag_out, sql_out, hn, cn


def get_features(data):
    X = []
    tags = []
    templates = []
    complete = []
    for dict in data:
        X.append(dict['input'])
        tags.append(dict['tags'])
        templates.append(dict['template'])
        complete.append(dict['complete'])

    return X, tags, templates, complete


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

    # Get lengths before padding
    lengths = torch.tensor([len(s) for s in sentences])

    # Pad sentences and tags to longest in batch
    sentences_padded = pad_sequence(
        sentences, batch_first=True, padding_value=0)
    # -1 so we can ignore pad positions in loss
    tags_padded = pad_sequence(tags,      batch_first=True, padding_value=-1)
    templates = torch.stack(templates)

    return sentences_padded, tags_padded, templates, lengths


def train(model, dataloader, optimizer,
          scheduler, criterion_tags,
          criterion_template, num_epochs, device,
          dev_sentences, complete_dev):
    model.to(device)
    # Set Optimizer, loss and scheduler:

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()

        total_loss = 0
        total_loss_tags = 0
        total_loss_template = 0

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
            loss = loss_tags + loss_template

            loss.backward()
            # prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_loss_tags += loss_tags.item()
            total_loss_template += loss_template.item()

            # Update tqdm bar with current batch loss
            progress_bar.set_postfix({
                "loss":     f"{loss.item():.4f}",
                "tag_loss": f"{loss_tags.item():.4f}",
                "sql_loss": f"{loss_template.item():.4f}"
            })

        # Step scheduler once per epoch
        # scheduler.step()

        dev_acc = evaluate(model, dev_sentences, complete_dev)

        # print(dev_acc)
        # Print epoch summary
        avg_loss = total_loss / len(dataloader)
        avg_loss_tags = total_loss_tags / len(dataloader)
        avg_loss_template = total_loss_template / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Tag Loss: {avg_loss_tags:.4f} | "
              f"SQL Loss: {avg_loss_template:.4f} | "
              f"LR: {current_lr:.6f}| "
              f"Val acc: {dev_acc:.6f}")


def evaluate(model, dev_sentences, complete_dev):
    complete_pred = predict_complete(
        model, dev_sentences, vocab, idx2tag, idx2template, device)
    accuracy = (sum(pred == true for pred, true in zip(
        complete_pred, complete_dev))/len(complete_pred))
    return accuracy


def predict_complete(model, sentence, vocab, idx2tag, idx2template, device):
    if type(sentence) == list:
        list_of_complete_sql = []
        for sent in sentence:
            list_of_complete_sql.append(predict_complete(
                model, sent, vocab, idx2tag, idx2template, device))
        return list_of_complete_sql

    model.eval()
    encoded_sentence = encode(sentence, vocab)
    with torch.no_grad():
        encoded_sentence = encode(sentence, vocab)
        tensor = torch.tensor(encoded_sentence).unsqueeze(
            0).to(device)  # (1, seq_len)
        tag_per_word, template_pred, hn, cn = model(tensor)

    tags_pred = torch.argmax(tag_per_word.squeeze(0), dim=-1)  # (seq_len,)
    template_idx = torch.argmax(
        template_pred.squeeze(0), dim=-1).item()  # scalar
    template = idx2template[template_idx]
    tags = [idx2tag[idx.item()] for idx in tags_pred]
    # Check if there are 2 or more of any tags predicted
    # Record their idx and the tag itself in a dictionary
    tag_indices = {}
    for i, tag in enumerate(tags):
        if tag != 'O':
            if tag not in tag_indices:
                tag_indices[tag] = []
            tag_indices[tag].append(i)

    for tag, indices in tag_indices.items():
        if len(indices) >= 2:
            # Longer word
            if all(indices[i] + 1 == indices[i+1] for i in range(len(indices)-1)):
                variable = sentence.split()[indices[0]:indices[-1]+1]
                template = template.replace(tag, " ".join(variable))

            # if not consecutive, use the one with highest score
            else:
                max_idx = tags_pred[indices].argmax()
                variable = sentence.split()[max_idx]
                template = template.replace(tag, variable)
        if len(indices) == 1:
            variable = sentence.split()[indices[0]]
            template = template.replace(tag, variable)
    complete_sql = template

    return complete_sql


def main_LSTM():
    # Load data and build vocab
    train_data = read_data('data/geography.train.json')
    dev_data = read_data('data/geography.dev.json')
    x_train, tags_train, templates_train, complete_train = get_features(
        train_data)
    x_dev, tags_dev, templates_dev, complete_dev = get_features(dev_data)
    # print(x_dev)
    # Count word frequencies
    global vocab
    global idx2tag, idx2template, device

    vocab = build_vocab(x_train, 1)

    # Set tag2idx and idx2tag
    all_tags = set(tag for tags in tags_train for tag in tags)
    all_templates = set(templates_train)

    tag2idx = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
    temp2idx = {template: idx for idx,
                template in enumerate(sorted(all_templates))}

    # Build inverse mappings
    idx2tag = {idx: tag for tag,      idx in tag2idx.items()}
    idx2template = {idx: template for template, idx in temp2idx.items()}

    # Create datasets and dataloaders
    train_dataset = SQLDataset(
        x_train, tags_train, templates_train, vocab, tag2idx, temp2idx)
    # dev_dataset = SQLDataset(
    #    x_dev, tags_dev, templates_dev, vocab, tag2idx, temp2idx)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    # dev_loader = DataLoader(dev_dataset, batch_size=32,
    #                        collate_fn=collate_fn)

    # Set hyperparameters
    vocab_size = len(vocab)
    embedding_dim = 128*2*2*2
    hidden_dim = 128*2*2*2
    layer_dim = 1
    output_dim_tag = len(train_dataset.tag2idx)
    output_dim_sql = len(train_dataset.temp2idx)
    epochs = 45
    step_size = 5
    gamma = 0.5
    # Initialize model, loss functions, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM_model(vocab_size, embedding_dim, hidden_dim,
                       layer_dim, output_dim_tag, output_dim_sql).to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=1.0, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)

    criterion_tags = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_template = nn.CrossEntropyLoss()

    dev_sentences = [" ".join(x) for x in x_dev]
    train(model, train_loader, optimizer, scheduler, criterion_tags,
          criterion_template, num_epochs=epochs, device=device, dev_sentences=dev_sentences,
          complete_dev=complete_dev)

    # Check accuracy on validation complete

    complete_pred = predict_complete(
        model, dev_sentences, vocab, idx2tag, idx2template, device)
    print(sum(pred == true for pred, true in zip(
        complete_pred, complete_dev))/len(complete_pred))

    sentences = []
    with open("test.txt", 'r') as file:
        for line in file:
            sentences.append(line.strip())

    complete_sql = predict_complete(
        model, sentences, vocab, idx2tag, idx2template, device)

    with open("q5_predictions.txt", 'w') as file:
        for sent, sql in zip(sentences, complete_sql):
            # print(sent + "|" + sql)
            file.write(sent + "|" + sql + "\n")


if __name__ == "__main__":
    # linear_model()
    main_LSTM()
