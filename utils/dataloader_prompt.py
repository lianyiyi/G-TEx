from utils.preprocessor import Esnli_preprocessor, Ecqa_preprocessor, Cose_preprocessor, label_dict, Comve_preprocessor
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import random
import torch
from itertools import combinations


class ExpDataset(Dataset):
    def __init__(self, data_list, tokenizer, model_max_length, top_percent=0.3):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.k = top_percent
        self.model_max_length = model_max_length
        self.max_input_len = self._max_source_leng()
        self.max_output_len = self._max_target_leng()

    def __len__(self):
        return len(self.data_list)

    def _max_source_leng(self):
        max_input_len = 0
        for data in self.data_list:
            input_ids = self.tokenizer.encode(data.source_text)
            prompt_length = len(self.tokenizer.encode(' The most important tokens are: '))
            length_input = int(len(input_ids) * 1.3 + prompt_length)
            max_input_len = max(length_input, max_input_len)
        return min(max_input_len, self.model_max_length)

    def _max_target_leng(self):
        max_input_len = 0
        for data in self.data_list:
            input_ids = self.tokenizer.encode(data.target_text)
            max_input_len = max(len(input_ids), max_input_len)
        return min(max_input_len, self.model_max_length)

    def _tokenize_output(self, text):
        model_input = self.tokenizer(text=text, max_length=self.max_output_len, padding='max_length', truncation=True, return_tensors="pt")
        return model_input.input_ids, model_input.attention_mask

    def _get_model_input(self, text, tokens):
        model_input = self.tokenizer(text=text, max_length=self.max_input_len, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = model_input.input_ids
        attention_mask = model_input.attention_mask

        pad_id = self.tokenizer.pad_token_id
        token_ids = torch.reshape(input_ids, (-1,)).tolist()  # turn token ids into a list
        token_ids = token_ids[:token_ids.index(pad_id)] if pad_id in token_ids else token_ids  # truncate pad tokens
        last_token_index = len(token_ids) - 1

        sep_token = self.tokenizer.convert_tokens_to_ids('</s>')
        sep_token_position = token_ids.index(sep_token)

        high_tokens = tokens[:int(len(tokens) * self.k)]

        # Append the highlighted tokens to the input as an explicit prompt, then
        # re-tokenize before building the highlight graph.
        high_token_ids = [token_ids[x] for x in high_tokens if x < len(token_ids)]
        high_token_text = ' '.join([self.tokenizer.decode(i) for i in high_token_ids])
        input_text_with_prompt = text + ' The most important tokens are: ' + high_token_text

        model_input = self.tokenizer(text=input_text_with_prompt, max_length=self.max_input_len, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = model_input.input_ids
        attention_mask = model_input.attention_mask

        combis = list(combinations(high_tokens, 2))
        edge_tuples = []
        for x1, x2 in combis:
            edge_tuples.append([x1, x2])
        for t in high_tokens:
            edge_tuples.append([t, sep_token_position])
            edge_tuples.append([t, last_token_index])

        edge_index = torch.tensor(edge_tuples, dtype=torch.long).t().contiguous()
        return input_ids, attention_mask, edge_index, input_text_with_prompt

    def __getitem__(self, index):
        text = self.data_list[index].source_text
        target = self.data_list[index].target_text
        tokens = self.data_list[index].high_tokens
        label = self.data_list[index].label

        input_ids, attention_mask, edge_index, input_text_with_prompt = self._get_model_input(text, tokens)
        target_ids, target_attention_mask = self._tokenize_output(target)

        return Data(num_nodes=len(input_ids), input_ids=input_ids, attention_mask=attention_mask, edge_index=edge_index,
                    target_input_ids=target_ids, target_attention_mask=target_attention_mask, input_text=input_text_with_prompt, target_text=target, label=label)


class ExpDataModule(LightningDataModule):
    def __init__(self, task_name, file_name, batch_size, tokenizer, model_max_length=512, few_shot=None):
        super().__init__()
        self.task_name = task_name
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.setup(file_name, few_shot)

    def setup(self, filename, few_shot, stage=None):
        if self.task_name == 'esnli':
            esnli = Esnli_preprocessor(filename)
            if not few_shot:
                self.train_set = esnli.sample_list_train
            else:
                all = esnli.sample_list_train
                entail = [s for s in all if s.label == 0]
                neutral = [s for s in all if s.label == 1]
                contra = [s for s in all if s.label == 2]
                self.train_set = random.sample(entail, few_shot) + random.sample(neutral, few_shot) + random.sample(contra, few_shot)
            self.val_set = esnli.sample_list_val
            self.test_set = esnli.sample_list_test
        elif self.task_name == 'ecqa':
            ecqa = Ecqa_preprocessor(filename)
            self.train_set = ecqa.sample_list_train
            self.val_set = ecqa.sample_list_val
            self.test_set = ecqa.sample_list_test
        elif self.task_name == 'cose':
            cose = Cose_preprocessor(filename)
            self.train_set = cose.sample_list_train
            self.val_set = cose.sample_list_val
            self.test_set = cose.sample_list_test
        elif self.task_name == 'comve':
            comve = Comve_preprocessor(filename)
            self.train_set = comve.sample_list_train
            self.val_set = comve.sample_list_val
            self.test_set = comve.sample_list_test

    def train_dataloader(self):
        train_dataset = ExpDataset(self.train_set, self.tokenizer, self.model_max_length)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        val_dataset = ExpDataset(self.val_set, self.tokenizer, self.model_max_length)
        return DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False, num_workers=10)

    def test_dataloader(self):
        test_dataset = ExpDataset(self.test_set, self.tokenizer, self.model_max_length)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=10)
