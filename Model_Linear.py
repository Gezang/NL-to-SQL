import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def read_data(filename):
    with open(filename, "r") as f:
        raw_data = json.load(f)
    data = []
    for entry in raw_data:
        template = sorted(entry["sql"], key=lambda x: (len(x), x))[0]
        for sentence in entry["sentences"]:
            tokens = sentence["text"].split()
            tags = ["O"] * len(tokens)
            complete = template
            for variable, value in sentence["variables"].items():
                idx = tokens.index(variable)
                value_tokens = value.split()
                tokens = tokens[:idx] + value_tokens + tokens[idx + 1:]
                tags = tags[:idx] + [variable] * len(value_tokens) + tags[idx + 1:]
                complete = complete.replace(variable, value)
            data.append({"input": tokens, "tags": tags, "template": template, "complete": complete})
    return data


class DataProcessor:
    def __init__(self, train_data, dev_data):
        self.train_data = train_data
        self.dev_data = dev_data

    def fit_transform_preprocess(self):
        X_train, tags_train, templates_train, _ = self._get_features(self.train_data)
        X_dev, tags_dev, templates_dev, complete_dev = self._get_features(self.dev_data)

        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()
        self.train_tfidf = self.transformer.fit_transform(
            self.vectorizer.fit_transform([" ".join(x) for x in X_train]))
        self.dev_tfidf = self.transformer.transform(
            self.vectorizer.transform([" ".join(x) for x in X_dev]))

        temp_set = set(templates_train)
        self.temp_mapping = {tmpl: i for i, tmpl in enumerate(temp_set)}
        self.reverse_temp_mapping = {i: tmpl for tmpl, i in self.temp_mapping.items()}
        self.y_templates_train = [self.temp_mapping[t] for t in templates_train]
        self.y_templates_dev = [self.temp_mapping.get(t, -1) for t in templates_dev]

        all_words_train = [word for sentence in X_train for word in sentence]
        self.y_tags_train = [tag for sentence in tags_train for tag in sentence]

        self.tag_vectorizer = CountVectorizer()
        self.X_tags_train = self.tag_vectorizer.fit_transform(all_words_train)
        self.X_tags_dev = self.tag_vectorizer.transform(
            [word for sentence in X_dev for word in sentence])
        self.y_tags_dev = [tag for sentence in tags_dev for tag in sentence]

        self.X_dev = X_dev
        self.complete_dev = complete_dev

    def _get_features(self, data):
        X, tags, templates, complete = [], [], [], []
        for item in data:
            X.append(item["input"])
            tags.append(item["tags"])
            templates.append(item["template"])
            complete.append(item["complete"])
        return X, tags, templates, complete


class LinearModel:
    def __init__(self, train_data, dev_data):
        self.template_model = LinearSVC()
        self.tag_model = LogisticRegression()
        self.data = DataProcessor(train_data, dev_data)
        self.data.fit_transform_preprocess()

    def train(self):
        self.template_model.fit(self.data.train_tfidf, self.data.y_templates_train)
        self.tag_model.fit(self.data.X_tags_train, self.data.y_tags_train)

    def evaluate(self):
        template_preds = self.template_model.predict(self.data.dev_tfidf)
        template_acc = sum(p == t for p, t in zip(template_preds, self.data.y_templates_dev))
        template_acc /= len(self.data.y_templates_dev)

        tag_preds = self.tag_model.predict(self.data.X_tags_dev)
        tag_acc = sum(p == t for p, t in zip(tag_preds, self.data.y_tags_dev))
        tag_acc /= len(self.data.y_tags_dev)

        dev_sentences = [" ".join(tokens) for tokens in self.data.X_dev]
        complete_preds = self.predict_complete_sql(dev_sentences)
        complete_acc = sum(p == t for p, t in zip(complete_preds, self.data.complete_dev))
        complete_acc /= len(self.data.complete_dev)

        print()
        print("  Dev Set Results")
        print("  " + "-" * 30)
        print(f"  Complete SQL Accuracy : {complete_acc:.4f}")
        print(f"  Template Accuracy     : {template_acc:.4f}")
        print(f"  Tag Accuracy          : {tag_acc:.4f}")
        print("=" * 60)

    def predict_complete_sql(self, sentence):
        if isinstance(sentence, list):
            return [self.predict_complete_sql(s) for s in sentence]

        template_idx = self.template_model.predict(
            self.data.transformer.transform(self.data.vectorizer.transform([sentence])))[0]
        tags_pred = self.tag_model.predict(
            self.data.tag_vectorizer.transform(sentence.split()))
        template = self.data.reverse_temp_mapping[template_idx]

        tag_indices = {}
        for i, tag in enumerate(tags_pred):
            if tag != "O":
                tag_indices.setdefault(tag, []).append(i)

        tokens = sentence.split()
        for tag, indices in tag_indices.items():
            if len(indices) >= 2:
                if all(indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1)):
                    template = template.replace(tag, " ".join(tokens[indices[0]:indices[-1] + 1]))
                else:
                    probabilities = self.tag_model.predict_proba(
                        self.data.tag_vectorizer.transform([tokens[idx] for idx in indices]))
                    max_idx = indices[probabilities.argmax()]
                    template = template.replace(tag, tokens[max_idx])
            elif len(indices) == 1:
                template = template.replace(tag, tokens[indices[0]])

        return template


def linear_model():
    print("=" * 60)
    print("  Linear Model")
    print("=" * 60)

    train_data = read_data("data/geography.train.json")
    dev_data = read_data("data/geography.dev.json")

    model = LinearModel(train_data, dev_data)
    model.train()
    model.evaluate()

    test_sentences = [line.strip() for line in open("data/test.txt")]
    complete_sql = model.predict_complete_sql(test_sentences)
    with open("results/predictions_linear.txt", "w") as f:
        for sent, sql in zip(test_sentences, complete_sql):
            f.write(sent + "|" + sql + "\n")
