import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


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
        # print(self.train_tfidf.shape)

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

        self.X_dev = X_dev
        self.complete_dev = complete_dev

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
        # print(self.data.y_tags_train)

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

    def evaluate_complete_sql(self):
        dev_sentences = [" ".join(tokens) for tokens in self.data.X_dev]
        predictions = self.predict_complete_sql(dev_sentences)
        correct = sum(pred == true for pred, true in zip(
            predictions, self.data.complete_dev))
        accuracy = correct / len(self.data.complete_dev)
        print(f"Complete SQL Accuracy: {accuracy:.2f}")

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
    print("\n Results for Linear Model:")
    model = Linear_model(train_data, dev_data)
    model.train_template_model()
    model.train_tag_model()
    model.evaluate_template_model()
    model.evaluate_tag_model()
    model.evaluate_complete_sql()
    sentences = []
    with open("data/test.txt", 'r') as file:
        for line in file:
            sentences.append(line.strip())

    complete_sql = model.predict_complete_sql(sentences)

    with open("data/q4_predictions.txt", 'w') as file:
        for sent, sql in zip(sentences, complete_sql):
            # print(sent + "|" + sql)
            file.write(sent + "|" + sql + "\n")
