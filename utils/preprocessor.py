import json

format_prompt_dict = {
    'esnli': '{text1} </s> {text2}',
    'ecqa': '{text1} </s> {text2}',
    'cose': '{text1} </s> {text2}',
    'comve': '{text1} </s> {text2}',
}

label_dict = {
    'esnli': {'entailment': 0, 'neutral': 1, 'contradiction': 2},
    'ecqa': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4},
    'cose': {'A': 0, 'B': 1, 'C': 2},
    'comve': {'1': 0, '2': 1},
}


class Esnli_sample:
    def __init__(self, premise, hypothesis, label, explain, highlight):
        self.source_text = format_prompt_dict['esnli'].format(text1=premise, text2=hypothesis)
        self.target_text = label + ' ' + explain
        self.high_tokens = highlight
        self.sent1 = premise
        self.sent2 = hypothesis
        self.label = label_dict['esnli'][label]
        self.training_text = self.source_text + ' [Explanation] ' + self.target_text


class Esnli_preprocessor:
    def __init__(self, train_file):
        val_file = train_file.replace('train', 'val')
        test_file = train_file.replace('train', 'test')
        self.sample_list_train = self.from_json(train_file)
        self.sample_list_val = self.from_json(val_file)
        self.sample_list_test = self.from_json(test_file)

    def from_json(self, file_name):
        sample_list = []
        with open(file_name, 'r') as file:
            samples = json.load(file)
            for s in samples:
                premise = s['text1']
                hypothesis = s['text2']
                label = s['label']
                tokens = s['tokens']
                exp = s['exp1']
                if type(exp) == str:
                    sample = Esnli_sample(premise, hypothesis, label, exp, tokens)
                    sample_list.append(sample)
        return sample_list


class EsnliSingle_sample:
    """e-SNLI sample for the decoder-only (Llama) path.

    Uses the single-file annotation format where highlights are stored as two
    per-sentence token-index lists and each gold explanation yields its own
    training instance.
    """

    def __init__(self, premise, hypothesis, label, explain, highlight):
        self.source_text = format_prompt_dict['esnli'].format(text1=premise, text2=hypothesis)
        self.target_text = label + ' ' + explain
        self.high_tokens = highlight
        self.sent1 = premise
        self.sent2 = hypothesis
        self.label = label_dict['esnli'][label]
        self.training_text = self.source_text + ' [Explanation] ' + self.target_text


class EsnliSingle_preprocessor:
    """Single-file e-SNLI loader (random split done downstream by the caller)."""

    def __init__(self, filename):
        self.sample_list = self.from_json(filename)

    def from_json(self, file_name):
        sample_list = []
        with open(file_name, 'r') as file:
            samples = json.load(file)
            for s in samples:
                premise = s['text1']
                hypothesis = s['text2']
                label = s['label']
                for exp in [s['exp1'], s['exp2'], s['exp3']]:
                    sample = EsnliSingle_sample(premise, hypothesis, label, exp,
                                                [s['text1_tokens'], s['text2_tokens']])
                    sample_list.append(sample)
        return sample_list


class Ecqa_sample:
    def __init__(self, premise, hypothesis, label, explain, highlight):
        hypothesis = hypothesis.replace(',', ', ')
        self.source_text = format_prompt_dict['ecqa'].format(text1=premise, text2=hypothesis)
        self.target_text = label + ' explanation: ' + explain
        self.high_tokens = highlight
        self.sent1 = premise
        self.sent2 = hypothesis
        self.label = label
        self.training_text = self.source_text + ' [Explanation] ' + self.target_text


class Ecqa_preprocessor:
    def __init__(self, train_file):
        val_file = train_file.replace('train', 'val')
        test_file = train_file.replace('train', 'test')
        self.sample_list_train = self.from_json(train_file)
        self.sample_list_val = self.from_json(val_file)
        self.sample_list_test = self.from_json(test_file)

    def from_json(self, file_name):
        sample_list = []
        with open(file_name, 'r') as file:
            samples = json.load(file)
            for s in samples:
                premise = s['text1']
                hypothesis = s['text2']
                label = s['label']
                tokens = s['tokens']
                exp = s['exp3']
                sample = Ecqa_sample(premise, hypothesis, label, exp, tokens)
                sample_list.append(sample)
        return sample_list


class Comve_sample:
    def __init__(self, premise, hypothesis, label, explain, highlight):
        hypothesis = hypothesis.replace(',', ', ')
        self.source_text = format_prompt_dict['comve'].format(text1=premise, text2=hypothesis)
        self.target_text = label + ' explanation: ' + explain
        self.high_tokens = highlight
        self.sent1 = premise
        self.sent2 = hypothesis
        self.label = label
        self.training_text = self.source_text + ' [Explanation] ' + self.target_text


class Comve_preprocessor:
    def __init__(self, train_file):
        val_file = train_file.replace('train', 'dev')
        test_file = train_file.replace('train', 'test')
        self.sample_list_train = self.from_json(train_file)
        self.sample_list_val = self.from_json(val_file)
        self.sample_list_test = self.from_json(test_file)

    def from_json(self, file_name):
        sample_list = []
        with open(file_name, 'r') as file:
            samples = json.load(file)
            for s in samples:
                premise = s['text1']
                hypothesis = s['text2']
                label = s['label']
                tokens = s['tokens']
                exp = s['exp3']
                sample = Comve_sample(premise, hypothesis, label, exp, tokens)
                sample_list.append(sample)
        return sample_list


class Cose_sample:
    def __init__(self, premise, hypothesis, label, explain, highlight):
        self.source_text = format_prompt_dict['cose'].format(text1=premise, text2=hypothesis)
        self.target_text = label + ' explanation: ' + explain
        self.high_tokens = highlight
        self.sent1 = premise
        self.sent2 = hypothesis
        self.label = label
        self.training_text = self.source_text + ' [Explanation] ' + self.target_text


class Cose_preprocessor:
    def __init__(self, train_file):
        val_file = train_file.replace('train', 'val')
        test_file = train_file.replace('train', 'test')
        self.sample_list_train = self.from_json(train_file)
        self.sample_list_val = self.from_json(val_file)
        self.sample_list_test = self.from_json(test_file)

    def from_json(self, file_name):
        sample_list = []
        with open(file_name, 'r') as file:
            samples = json.load(file)
            for s in samples:
                premise = s['text1']
                hypothesis = s['text2']
                label = s['label']
                tokens = s['tokens']
                exp = s['exp1']
                sample = Cose_sample(premise, hypothesis, label, exp, tokens)
                sample_list.append(sample)
        return sample_list
