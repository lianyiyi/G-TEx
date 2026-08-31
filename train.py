import argparse
import os
import random
import string

import evaluate
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam, AdamW, RAdam
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

from models.bart_gnn import BartForConditionalGenerationwithGNN
from models.t5_gnn import T5ForConditionalGenerationGNN
from utils.preprocessor import label_dict

OPTIMIZERS = {'Adam': Adam, 'AdamW': AdamW, 'RAdam': RAdam}

# Each mode maps to (main dataloader module, faithfulness dataloader module).
MODE_DATALOADERS = {
    'high_tokens': ('utils.dataloader', 'utils.dataloader_for_faith'),
    'span_pair': ('utils.dataloader_span_pair', 'utils.dataloader_for_faith'),
    'token_pair': ('utils.dataloader_token_pair', 'utils.dataloader_for_faith'),
    'prompt': ('utils.dataloader_prompt', 'utils.dataloader_for_faith'),
}

# Each model family maps to (tokenizer, plain LM, GNN-fused LM, config).
MODEL_FAMILIES = {
    't5': (T5Tokenizer, T5ForConditionalGeneration, T5ForConditionalGenerationGNN, T5Config),
    'bart': (BartTokenizer, BartForConditionalGeneration, BartForConditionalGenerationwithGNN, BartConfig),
}


def _load_datamodule(module_path):
    module = __import__(module_path, fromlist=['ExpDataModule'])
    return module.ExpDataModule


def _model_family(model_name):
    if 't5' in model_name.lower():
        return 't5'
    if 'bart' in model_name.lower():
        return 'bart'
    raise ValueError(f"Cannot infer model family from {model_name!r} (expected 't5' or 'bart').")


class ExpGeneration(LightningModule):
    def __init__(self, model_name_or_path, task_name, exp_type, tokenizer, run_id, optimizer,
                 learning_rate, warmup_steps, training_steps, do_schedule, num_beam, max_text_length,
                 gnn, generated_files):
        super().__init__()

        self.run_id = run_id
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        self.model_name_or_path = model_name_or_path
        self.optimizer = optimizer
        self.do_schedule = do_schedule
        self.num_beam = num_beam
        self.max_text_length = max_text_length

        self.task_name = task_name
        self.label_map = label_dict[task_name]
        self.tokenizer = tokenizer
        self.generated_files = generated_files

        self.save_hyperparameters()

        self.gnn = gnn
        self.exp_type = exp_type

        _, plain_cls, gnn_cls, config_cls = MODEL_FAMILIES[_model_family(model_name_or_path)]
        config = config_cls.from_pretrained(model_name_or_path)

        if self.exp_type == 'gnn':
            self.model = gnn_cls.from_pretrained(self.model_name_or_path, config=config, num_beam=num_beam, gnn=gnn)
        elif self.exp_type in ('base', 'prompt'):
            self.model = plain_cls.from_pretrained(self.model_name_or_path, config=config)
            self.gnn = 'none'
        elif self.exp_type == 'gnn-peft':
            # Freeze everything except the injected GNN layer.
            self.model = gnn_cls.from_pretrained(self.model_name_or_path, config=config, num_beam=num_beam, gnn=gnn)
            for name, param in self.model.named_parameters():
                if "gnn" not in name:
                    param.requires_grad = False
        else:
            raise ValueError(f"Unknown exp_type: {self.exp_type!r}")

        self.model.resize_token_embeddings(len(self.tokenizer))

        self.bleu = evaluate.load('bleu', experiment_id=run_id)
        self.accuracy = evaluate.load("accuracy", experiment_id=run_id)

    def forward(self, input_ids, attention_mask, labels, edge_index):
        if 'gnn' in self.exp_type:
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, edge_index=edge_index)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def _generate(self, input_ids, attention_mask, edge_index):
        if 'gnn' in self.exp_type:
            return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, edge_index=edge_index,
                                       num_beams=self.num_beam, min_length=20, max_length=self.max_text_length, early_stopping=True)
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                   num_beams=self.num_beam, min_length=20, max_length=self.max_text_length, early_stopping=True)

    def _decode(self, generated_ids):
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [x if x.strip() else 'None' for x in generated_text]

    def _predicted_labels(self, generated_text, label):
        if self.task_name == 'esnli':
            generated_labels = [x.split()[0] for x in generated_text]
            generated_labels = [self.label_map[l.lower()] if l in self.label_map.keys() else 3 for l in generated_labels]
        elif self.task_name in ('ecqa', 'comve'):
            generated_labels = [x.split('explanation:')[0].strip() for x in generated_text]
            generated_labels = [self.tokenizer.encode(l, add_special_tokens=False)[0] if l else 0 for l in generated_labels]
            label = [self.tokenizer.encode(l, add_special_tokens=False)[0] for l in label]
        return generated_labels, label

    def training_step(self, batch, batch_idx):
        output = self(batch.input_ids, batch.attention_mask, batch.target_input_ids, batch.edge_index)
        loss = output.loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        generated_ids = self._generate(batch.input_ids, batch.attention_mask, batch.edge_index)
        generated_text = self._decode(generated_ids)
        generated_labels, label = self._predicted_labels(generated_text, batch.label)

        acc = self.accuracy.compute(predictions=generated_labels, references=label)['accuracy']
        bleu_score = self.bleu.compute(predictions=generated_text, references=batch.target_text)["bleu"]

        self.log("val_bleu", bleu_score, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_accuracy", acc, prog_bar=True, logger=True, sync_dist=True)
        return {"val_bleu": bleu_score, 'val_accuracy': acc}

    def test_step(self, batch, batch_idx):
        input_texts = list(batch.input_text)
        references = list(batch.target_text)
        words = list(batch.word) if 'faith' in self.generated_files else []

        generated_ids = self._generate(batch.input_ids, batch.attention_mask, batch.edge_index)
        generated_text = self._decode(generated_ids)
        generated_labels, label = self._predicted_labels(generated_text, batch.label)

        acc = self.accuracy.compute(predictions=generated_labels, references=label)['accuracy']

        dir_name = self.model_name_or_path + '_' + self.exp_type + '_' + self.gnn + str(self.learning_rate)
        new_dir = os.path.join(os.getcwd(), self.generated_files, self.task_name, dir_name)
        os.makedirs(new_dir, exist_ok=True)

        with open(os.path.join(new_dir, 'hypos_' + str(self.learning_rate) + self.run_id + '.txt'), 'a') as f:
            for h in generated_text:
                f.write(h + '\n')
        with open(os.path.join(new_dir, 'refs_' + str(self.learning_rate) + self.run_id + '.txt'), 'a') as r:
            for ref in references:
                r.write(ref.replace('\n', ' ') + '\n')
        with open(os.path.join(new_dir, 'inputs_' + str(self.learning_rate) + self.run_id + '.txt'), 'a') as i:
            if 'faith' in self.generated_files:
                for inp, w in zip(input_texts, words):
                    i.write(inp + ' ' + str(w) + '\n')
            else:
                for inp in input_texts:
                    i.write(inp + '\n')

        bleu_score = self.bleu.compute(predictions=generated_text, references=references)["bleu"]
        self.log("test_bleu", bleu_score, prog_bar=True, logger=True, sync_dist=True)
        self.log("test_accuracy", acc, prog_bar=True, logger=True, sync_dist=True)
        return {"test_bleu": bleu_score, "test_accuracy": acc}

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        if self.do_schedule:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.training_steps)
            return [optimizer], [scheduler]
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--mode", type=str, default='do_train',
                            help="do_train | do_test | do_test_faith")
        parser.add_argument("--graph_mode", type=str, default='high_tokens',
                            choices=list(MODE_DATALOADERS.keys()),
                            help="Highlight graph construction mode.")
        parser.add_argument("--exp_type", type=str, default='base',
                            help="gnn | base | prompt | gnn-peft")
        parser.add_argument("--gnn", type=str, default='gcn', help="gcn | gat | sage")
        parser.add_argument("--task_name", type=str, default='esnli')

        parser.add_argument("--epochs", type=int, default=1, help="training epochs")
        parser.add_argument("--learning_rate", type=float, default=3e-5, help="learning rate")
        parser.add_argument("--batch_size", type=int, default=24, help="batch size")
        parser.add_argument("--optimizer", type=str, default='Adam', help="optimizer")
        parser.add_argument("--model_name", type=str, default='google-t5/t5-large', help="name of the model")

        parser.add_argument("--num_gpu", type=int, default=1, help="number of gpus")
        parser.add_argument("--random_seed", type=int, default=0, help="random_seed")

        parser.add_argument("--project_name", type=str, default='gtex', help="name of the project")
        parser.add_argument("--checkpoint", type=str, default='')
        parser.add_argument("--checkpoint_dir", type=str, default='checkpoints')

        parser.add_argument("--do_schedule", type=bool, default=False)
        parser.add_argument("--warm_up_steps", type=int, default=0, help="number of warm up steps")
        parser.add_argument("--early_stop", type=int, default=15)

        parser.add_argument("--num_beam", type=int, default=3)
        parser.add_argument("--max_text_length", type=int, default=120)
        parser.add_argument("--few_shot", type=int, default=None)
        parser.add_argument("--top_k", type=float, default=0.3, help="fraction of highlighted tokens kept")

        parser.add_argument("--data_file", type=str, required=True, help="training data JSON (train split).")
        parser.add_argument("--data_file_2", type=str, default=None, help="faithfulness test data JSON.")
        parser.add_argument("--generated_files", type=str, default='generated_text/high_tokens',
                            help="output directory for generations.")
        parser.add_argument("--faith_generated_files", type=str, default='generated_text/high_tokens_for_faithfulness',
                            help="output directory for faithfulness-run generations.")
        return parser


def main(args):
    seed_everything(args.random_seed)
    torch.set_float32_matmul_precision('high')

    family = _model_family(args.model_name)
    tokenizer_cls = MODEL_FAMILIES[family][0]
    tokenizer = tokenizer_cls.from_pretrained(args.model_name)

    main_loader_path, faith_loader_path = MODE_DATALOADERS[args.graph_mode]
    ExpDataModuleMain = _load_datamodule(main_loader_path)
    ExpDataModuleFaith = _load_datamodule(faith_loader_path)

    run_id = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))
    optimizer = OPTIMIZERS[args.optimizer]

    icldata = ExpDataModuleMain(args.task_name, args.data_file, args.batch_size, tokenizer, few_shot=args.few_shot)
    tokenizer = icldata.tokenizer
    training_steps = len(icldata.train_dataloader()) * args.epochs

    model = ExpGeneration(args.model_name, args.task_name, args.exp_type, tokenizer, run_id, optimizer,
                          args.learning_rate, args.warm_up_steps, training_steps, args.do_schedule,
                          args.num_beam, args.max_text_length, args.gnn, args.generated_files)

    wandb_logger = WandbLogger(project=args.project_name)
    early_stop_callback = EarlyStopping(monitor="val_bleu", patience=args.early_stop, mode="max")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=args.project_name + str(args.random_seed) + args.optimizer + str(args.learning_rate),
        monitor="val_bleu", mode="max", save_top_k=1)

    if args.num_gpu == 1:
        trainer = Trainer(callbacks=[checkpoint_callback, early_stop_callback], max_epochs=args.epochs,
                          accelerator='gpu', devices=[0], logger=wandb_logger)
    else:
        trainer = Trainer(callbacks=[checkpoint_callback, early_stop_callback], max_epochs=args.epochs,
                          strategy='ddp_find_unused_parameters_true', accelerator="gpu", devices=-1, logger=wandb_logger)

    def run_faithfulness_test(ckpt_path=None):
        model.generated_files = args.faith_generated_files
        faith_data = ExpDataModuleFaith(args.task_name, args.data_file_2, args.batch_size, tokenizer, top_k=args.top_k)
        if ckpt_path:
            trainer.test(model, faith_data.test_dataloader(), ckpt_path=ckpt_path)
        else:
            trainer.test(model, faith_data.test_dataloader())

    if args.checkpoint:
        if args.mode == 'do_train':
            trainer.fit(model, train_dataloaders=icldata.train_dataloader(), val_dataloaders=icldata.val_dataloader(), ckpt_path=args.checkpoint)
            trainer.test(dataloaders=icldata.test_dataloader(), ckpt_path='best')
        elif args.mode == 'do_test':
            trainer.test(model, icldata.test_dataloader(), ckpt_path=args.checkpoint)
        elif args.mode == 'do_test_faith':
            run_faithfulness_test(ckpt_path=args.checkpoint)
    else:
        if args.mode == 'do_train':
            trainer.fit(model, train_dataloaders=icldata.train_dataloader(), val_dataloaders=icldata.val_dataloader())
            trainer.test(dataloaders=icldata.test_dataloader(), ckpt_path='best')
            if args.data_file_2:
                run_faithfulness_test()
        elif args.mode == 'do_test':
            trainer.test(model, icldata.test_dataloader())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = ExpGeneration.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)
