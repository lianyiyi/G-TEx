from utils.preprocessor import EsnliSingle_preprocessor, label_dict
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import random
import torch


class ExpDataset(Dataset):
    def __init__(self, data_list, tokenizer, model_max_length, topk=3):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]', 'additional_special_tokens': ['[Explanation]']})
        self.model_max_length = model_max_length
        self.max_input_len = self._max_training_leng()
        self.max_val_len = self._max_val_leng()
        self.k = topk

    def __len__(self):
        return len(self.data_list)

    def _max_training_leng(self):
        max_input_len = 0
        for data in self.data_list:
            input_ids = self.tokenizer.encode(data.training_text)
            max_input_len = max(len(input_ids), max_input_len)
        return min(max_input_len, self.model_max_length)

    def _max_val_leng(self):
        max_input_len = 0
        for data in self.data_list:
            input_ids = self.tokenizer.encode(data.source_text)
            max_input_len = max(len(input_ids), max_input_len)
        return min(max_input_len, self.model_max_length)

    def _tokenize_val_input(self, text):
        self.tokenizer.padding_side = "left"
        model_input = self.tokenizer(text=text, max_length=self.max_val_len, padding='max_length', truncation=True, return_tensors="pt")
        return model_input.input_ids, model_input.attention_mask

    def _get_model_input(self, text, source_text, source_ids, word_interaction):
        self.tokenizer.padding_side = "left"
        model_input = self.tokenizer(text=text, max_length=self.max_input_len, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = model_input.input_ids
        attention_mask = model_input.attention_mask

        pad_id = self.tokenizer.pad_token_id
        train_input_ids = torch.reshape(input_ids, (-1,)).tolist()  # turn token ids into a list
        val_input_ids = torch.reshape(source_ids, (-1,)).tolist()

        def make_dict(inputs):
            input_id_index = {}
            for i, id_index in enumerate(inputs):
                input_id_index[i] = id_index
            return input_id_index

        train_input_id_index, val_input_id_index = make_dict(train_input_ids), make_dict(val_input_ids)

        last_train_position = len(train_input_ids) - 1
        last_val_position = len(val_input_ids) - 1

        # Connect every non-pad token to the final position, separately for the
        # training (teacher-forced) and validation (generation) inputs.
        train_edge_tuples = []
        val_edge_tuples = []
        for k, v in train_input_id_index.items():
            if v != pad_id:
                train_edge_tuples.append([k, last_train_position])
        for k, v in val_input_id_index.items():
            if v != pad_id:
                val_edge_tuples.append([k, last_val_position])

        train_edge_index = torch.tensor(train_edge_tuples, dtype=torch.long).t().contiguous()
        val_edge_index = torch.tensor(val_edge_tuples, dtype=torch.long).t().contiguous()
        return input_ids, attention_mask, train_edge_index, val_edge_index

    def __getitem__(self, index):
        input_text = self.data_list[index].source_text + ' [Explanation]'
        target_text = self.data_list[index].target_text + '<|end_of_text|>'
        text = self.data_list[index].training_text + '<|end_of_text|>'
        interaction = self.data_list[index].high_tokens
        label = self.data_list[index].label

        val_input_ids, val_attention_mask = self._tokenize_val_input(input_text)
        input_ids, attention_mask, edge_index, val_edge_index = self._get_model_input(text, input_text, val_input_ids, interaction)

        return Data(num_nodes=len(input_ids), input_ids=input_ids, attention_mask=attention_mask, edge_index=edge_index, val_edge_index=val_edge_index,
                    input_text=text, target_text=target_text, val_input_text=input_text, val_input_ids=val_input_ids, val_attention_mask=val_attention_mask, label=label)


class ExpDataModule(LightningDataModule):
    def __init__(self, task_name, file_name, batch_size, tokenizer, model_max_length=4096):
        super().__init__()
        self.task_name = task_name
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.setup(file_name)

    def setup(self, filename, stage=None):
        if self.task_name == 'esnli':
            sample_list = EsnliSingle_preprocessor(filename).sample_list
            random.shuffle(sample_list)
            length_train = len(sample_list) * 6 // 10
            length_test = len(sample_list) * 2 // 10
            self.train_set = sample_list[:length_train]
            self.val_set = sample_list[length_train:length_train + length_test]
            self.test_set = sample_list[length_train + length_test:]

    def train_dataloader(self):
        train_dataset = ExpDataset(self.train_set, self.tokenizer, self.model_max_length)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        val_dataset = ExpDataset(self.val_set, self.tokenizer, self.model_max_length)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=10)

    def test_dataloader(self):
        test_dataset = ExpDataset(self.test_set, self.tokenizer, self.model_max_length)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=10)
