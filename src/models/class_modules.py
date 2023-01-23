import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
from argparse import ArgumentParser
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torchmetrics import Accuracy
from ..utils import my_test_binary, get_kl_diver_loss, patch_related_args
from ..dataset.fever_dataloader import FeverData
from .patch import ModifyLinearInput, ModifyLinearOutput, ModuleDetector
from torch.optim.optimizer import Optimizer
from typing import Any, Callable, Optional, Union


class Editor(nn.Module):

    def __init__(self,
                 model, hidden_size=768,
                 activate_loss='non_use', memory_loss='non_use',
                 device=None, freeze_model=True, memory_size=50000,
                 amplify_v=False, freeze_k=False,
                 freeze_a=False, amplify_con=10.0,
                 drop_num=0, drop_rate=0.5,
                 act_margin_val=0.0, margin_val1=0.0, margin_val2=0.0
                 ):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.original_model = copy.deepcopy(model)
        self.hidden_size = hidden_size
        self.device = device
        self.memory_size = memory_size
        self.activate_loss = activate_loss
        self.memory_loss = memory_loss
        self.freeze_model = freeze_model
        self.freeze_a = freeze_a
        self.amplify_con = amplify_con
        if self.freeze_model:
            for p in self.model.parameters():
                p.requires_grad = False
        # initialization, may be re-initialized after every edit
        self.model_named_modules = None
        self.get_named_modules()

        self.editors = []
        self.detectors = []

        self.train_memories = {}
        self.val_memories = {}

        self.amplify_v = amplify_v
        self.freeze_k = freeze_k
        self.freeze_k = freeze_k

        self.drop_num = drop_num
        self.drop_rate = drop_rate

        self.act_margin_val = act_margin_val
        self.margin_val1 = margin_val1
        self.margin_val2 = margin_val2

        self.detected_modules = {'model.model.encoder.layer.11': 'intermediate'}

        self.name_edit_type = {
            'model.model.encoder.layer.11.output.dense': 'input',
            'model.model.encoder.layer.11.intermediate.dense': 'output'
        }

    def reset_model(self, model, clear_memory):
        self.model = copy.deepcopy(model)
        if self.freeze_model:
            for p in self.model.parameters():
                p.requires_grad = False
        self.model_named_modules = None
        self.get_named_modules()
        self.editors = []
        for ms in [self.train_memories, self.val_memories]:
            for m in ms.values():
                del m[-1 * clear_memory:]

    def clear_memory(self):
        # clear all memories
        self.memories = {}

    def clear_detectors(self):
        for d in self.detectors:
            self.model_named_modules[d['module']]._modules[d['child']] = d['original_module']
        self.detectors = []

    def clear_editors(self):
        for e in self.editors:
            self.model_named_modules[e['module']]._modules[e['child']] = e['original_module']
        self.editors = []

    def set_editors(self, batch=None, init_weights=None, error_count=0, select_index=0):
        # before every turn, we call this function to set the editors
        self.get_editors(batch=batch, init_weights=dict() if init_weights is None else init_weights)
        for e in self.editors:
            self.model_named_modules[e['module']]._modules[e['child']] = e['editor']

    def set_detectors(self):
        for d in self.detectors:
            self.model_named_modules[d['module']]._modules[d['child']] = d['detector']

    def step(self):
        # we assign the trained linear layer to the edit target
        for e in self.editors:
            self.model_named_modules[e['module']]._modules[e['child']] = e['editor'].assign_layer()
        self.editors = []

    def lock_hidden_detectors(self):
        for d in self.detectors:
            d['detector'].turn_off_hidden()

    def unlock_hidden_detectors(self):
        for d in self.detectors:
            d['detector'].turn_on_hidden()

    def lock_memory_detectors(self):
        for d in self.detectors:
            d['detector'].turn_off_memory()

    def unlock_memory_detectors(self):
        for d in self.detectors:
            d['detector'].turn_on_memory()

    def insert_hidden_detector(self):
        self.get_detectors(detected_modules=self.detected_modules)
        self.set_detectors()

    def get_hidden(self,):
        res = dict()
        for d in self.detectors:
            k = d['module'] + '.' + d['child']
            v = d['detector'].get_hidden()
            res[k] = v
        return res

    def feed_one_memory(self, m):
        for k, v in m.items():
            assert k in self.train_memories
            self.train_memories[k] += v
            self.val_memories[k] += [m]
            print("This is a update and now we have {} train memories and {} val_memories for module {}".format(
                len(self.train_memories[k]), len(self.val_memories[k]), k
            ))

    def construct_memory(self, data: DataLoader, memory_size, device, update=False, memory_use='train'):
        self.detectors = []
        self.model.eval()
        self.model.to(device)
        self.get_detectors(
            detected_modules={'model.model.encoder.layer.11': 'intermediate'})
        self.set_detectors()
        for d in self.detectors:
            if not update:
                d['detector'].turn_on_memory()
            else:
                d['detector'].turn_on_hidden()
        # bar = tqdm(enumerate(data), total=len(data.dataset) // data.batch_size)
        for batch_id, batch in enumerate(data):
            self.model(
                batch["src_input_ids"].to(device),
                batch["src_attention_mask"].to(device),
                batch["labels"].to(device)
            )
        for d in self.detectors:
            name = d['module']+'.'+d['child']
            if not update:
                # this is for construction
                if memory_use == 'train':
                    self.train_memories[name] = d['detector'].get_memory()
                else:
                    self.val_memories[name] = d['detector'].get_memory()
            else:
                assert name in self.train_memories
                self.train_memories[name] += [d['detector'].get_hidden()]
                self.val_memories[name] += [d['detector'].get_hidden()]
                print("This is a update and now we have {} train memories and {} val_memories".format(
                    len(self.train_memories[name]), len(self.val_memories[name])
                ))
            self.model_named_modules[d['module']]._modules[d['child']] = d['original_module']
        self.detectors = []

    def get_named_modules(self):
        # For now we just edit one linear layer once
        self.model_named_modules = {x[0]: x[1] for x in self.model.named_modules()}

    def get_editors(self, *args, **kwargs):
        init_weights = kwargs.get("init_weights")
        name_edit_type = self.name_edit_type
        for name, edit_type in name_edit_type.items():
            e_tmp = dict()
            n = name.rsplit('.', 1)
            e_tmp['module'], e_tmp['child'] = n[0], n[-1]
            if edit_type == 'input':
                e_tmp['editor'] = ModifyLinearInput(
                    self.model_named_modules[n[0]].__getattr__(n[-1]),
                    amplify=self.amplify_v, freeze_a=self.freeze_a, amplify_con=self.amplify_con,
                )
            else:
                init_weight = init_weights[n[0]] if n[0] in init_weights.keys() else None
                train_memo, val_memo = None, None
                if n[0] in self.train_memories.keys():
                    train_memo = self.train_memories[n[0]]
                    val_memo = self.val_memories[n[0]]
                e_tmp['editor'] = ModifyLinearOutput(
                    self.model_named_modules[n[0]].__getattr__(n[-1]),
                    init_weight=init_weight,  freeze=self.freeze_k,
                    activate_loss=self.activate_loss, memory_loss=self.memory_loss,
                    train_memories=train_memo, val_memories=val_memo,
                    drop_num=self.drop_num, drop_rate=self.drop_rate,
                    act_margin_val=self.act_margin_val, margin_val1=self.margin_val1,
                    margin_val2=self.margin_val2
                )
            e_tmp['original_module'] = self.model_named_modules[n[0]].__getattr__(n[-1])
            self.editors.append(e_tmp)

    def get_detectors(self, *args, **kwargs):
        detected_modules = kwargs["detected_modules"]
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'input'
        for module_name, child in detected_modules.items():
            detector = ModuleDetector(
                model=self.model_named_modules[module_name + '.' + child],
                memory_size=self.memory_size, mode=mode
            )
            self.detectors.append({
                'module': module_name, 'child': child,
                'detector': detector, 'original_module': self.model_named_modules[module_name + '.' + child]
            })

    def get_act_loss(self):
        act_loss = 0
        for e in self.editors:
            if isinstance(e['editor'], ModifyLinearOutput):
                act_loss_ = e['editor'].get_act_loss()
                if act_loss_ is not None:
                    act_loss += act_loss_
        return act_loss

    def get_memo_loss(self):
        memo_loss = 0
        for e in self.editors:
            if isinstance(e['editor'], ModifyLinearOutput):
                memo_loss_ = e['editor'].get_memo_loss()
                if memo_loss_ is not None:
                    memo_loss += memo_loss_
        return memo_loss

    def feed_kl_input(self, memo_loader, his_edit_data, total_loc_num):
        self.memo_loader = memo_loader
        self.total_loc_num = total_loc_num
        self.his_edit_data = his_edit_data

    def forward(self, input_ids, attention_mask, labels=None):
        # gen = self.drop_num > 0 and self.training
        # labels = torch.repeat_interleave(labels, self.drop_num + 1, dim=0) if gen else labels
        res = self.model(input_ids, attention_mask, labels)
        if self.activate_loss != 'non_use':
            res['act_loss'] = self.get_act_loss()
        if self.memory_loss != 'non_use' and not self.memory_loss.startswith('kl'):
            res['memo_loss'] = self.get_memo_loss()
        elif self.memory_loss.startswith('kl') and self.training:
            res['memo_loss'] = get_kl_diver_loss(
                original_model=self.original_model, post_model=self.model, memo_loader=self.memo_loader,
                device=self.device, total_loc_num=self.total_loc_num, his_edit_data=self.his_edit_data
            )
        return res


class BertBinaryEditor(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return patch_related_args(parser)

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        add_neuron_num = kwargs.get('add_neuron_num')
        self.current_device = torch.device('cuda', self.hparams.gpus[0])
        self.editor = Editor(
            model=BertBinary.load_from_checkpoint(self.hparams.model_path),
            freeze_model=self.hparams.freeze_model, memory_size=self.hparams.memory_size,
            amplify_v=self.hparams.amplify_v == 1, activate_loss=self.hparams.activate_loss,
            freeze_k=self.hparams.freeze_k == 1, memory_loss=self.hparams.memory_loss,
            freeze_a=self.hparams.freeze_a == 1,
            act_margin_val=self.hparams.act_margin_val, margin_val1=self.hparams.margin_val1,
            margin_val2=self.hparams.margin_val2, device=self.current_device
        )
        # if this is a continuation of previous experiments, we need extend the size of the edited layer
        if add_neuron_num is not None:
            for _ in range(add_neuron_num):
                self.editor.set_editors()
                self.editor.step()
        self.val_loader = None
        self.valid_memory_loss = []
        self.valid_metric = []
        self.save_ckpt = 0
        self.start_example_editing = False
        self.stop_editing = False
        self.BIG_CONSTANT = 10000
        self.has_stepped = False

    def on_train_start(self):
        self.valid_memory_loss = []
        self.valid_metric = []
        self.save_ckpt = 0
        self.start_example_editing = False
        self.stop_editing = False
        self.has_stepped = False

    @staticmethod
    def early_stop_editing(metrics, mode='min', thd=0.0, patience=1):
        best_step = 0
        for step, vm in enumerate(metrics):
            if mode == 'min':
                if vm < metrics[best_step] - thd:
                    best_step = step
            else:
                if vm > metrics[best_step] + thd:
                    best_step = step

        if best_step < len(metrics) - patience:
            return True
        else:
            return False

    @staticmethod
    def save_editing_ckpt(metrics, mode='min'):
        # save the model if new val metric is attained
        return (mode == 'min' and min(metrics) == metrics[-1]) or (mode == 'max' and max(metrics) == metrics[-1])

    def fed_val_loader(self, dl):
        self.val_loader = dl

    def reset(self, clear_memory):
        self.editor.reset_model(
            BertBinary.load_from_checkpoint(self.hparams.model_path),
            clear_memory=clear_memory
        )

    def forward(self, input_ids, attention_mask, labels):
        res = self.editor(input_ids, attention_mask, labels)
        return res

    def joint_training(self, batch, batch_idx=None):
        input_ids, attention_mask, labels = batch["src_input_ids"], batch["src_attention_mask"], batch["labels"]
        res = self.editor(input_ids, attention_mask, labels)
        loss = res['loss']
        if self.hparams.activate_loss != "non_use":
            self.log("al", res['act_loss'], on_step=True, on_epoch=False, prog_bar=True, batch_size=input_ids.size(0))
            loss = loss + self.hparams.alc * res['act_loss']
        if self.hparams.memory_loss != 'non_use':
            self.log("ml", res['memo_loss'], on_step=True, on_epoch=False, prog_bar=True, batch_size=input_ids.size(0))
            loss = loss + self.hparams.mlc * res['memo_loss']

        return {"loss": loss}

    def training_step(self, batch, batch_idx=None, optimizer_idx=None):
        return self.joint_training(batch, batch_idx)

    def joint_validation(self, batch, batch_idx=None):
        stop_editing, save_ckpt = False, False
        input_ids, attention_mask, labels = batch["src_input_ids"], batch["src_attention_mask"], batch["labels"]
        b_size = input_ids.size(0)
        res = self.editor(input_ids, attention_mask, labels)
        self.log("val_loss", res['loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=b_size)
        if self.current_epoch >= self.hparams.start_val_epoch:
            if self.hparams.use_val == 1:
                if res['metric'] == 1:
                    # val_metric = my_test_binary(self.editor.model, self.val_loader, self.device)[0]
                    # self.valid_metric.append(val_metric)
                    self.valid_memory_loss.append(res['memo_loss'])
                    stop_editing = float(self.early_stop_editing(
                        metrics=self.valid_memory_loss, thd=0.001,
                        patience=self.hparams.early_patience, mode='min'
                    ))
                    save_ckpt = self.save_editing_ckpt(metrics=self.valid_memory_loss, mode='min') or stop_editing == 1
            else:
                stop_editing = float(res['metric'] == 1)
                save_ckpt = res['metric'] == 1

            if self.hparams.memory_loss != 'non_use' and not self.hparams.memory_loss.startswith('kl'):
                self.log("v_ml", res['memo_loss'], on_step=False, on_epoch=True, prog_bar=True, batch_size=b_size)

        self.save_ckpt += float(save_ckpt)
        self.stop_editing = stop_editing
        if stop_editing == 1:
            self.editor.step()
            self.has_stepped = True
        self.log("stop_editing", stop_editing, on_step=False, on_epoch=True, batch_size=b_size, prog_bar=True)
        self.log("save_ckpt", self.save_ckpt, on_step=False, on_epoch=True, batch_size=b_size, prog_bar=True)

    def validation_step(self, batch, batch_idx=None):
        self.joint_validation(batch, batch_idx)

    def test_step(self, batch, batch_idx=None):
        input_ids, attention_mask, labels = batch["src_input_ids"], batch["src_attention_mask"], batch["labels"]
        b_size = input_ids.size(0)
        res = self.editor(input_ids, attention_mask, labels)
        test_loss = res['loss']
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=b_size)

    def memorize(self, train_memory_data, device, update, val_memory_data: DataLoader = None):
        self.editor.construct_memory(
            data=train_memory_data, device=device,
            memory_size=self.hparams.memory_size, update=update, memory_use='train'
        )
        if not update:
            self.editor.construct_memory(
                data=val_memory_data, device=device,
                memory_size=self.hparams.memory_size, update=update, memory_use='val'
            )

    def get_optimizer(self, params, lr=None, optim=None):
        if lr is None:
            lr = self.hparams.lr
        if optim is None:
            optim = self.hparams.optim
        if optim == "adam":
            return torch.optim.Adam(params=params, lr=lr, weight_decay=self.hparams.weight_decay)
        if optim == 'rmsprop':
            return torch.optim.RMSprop(params=params, lr=lr)
        if optim == 'sgd':
            return torch.optim.SGD(params=params, lr=lr, weight_decay=self.hparams.weight_decay, momentum=0.9)


    def configure_optimizers(self):
        # for the joint editing style, we just need one parameter
        parameters = [p for p in self.editor.parameters() if p.requires_grad]
        optimizer = self.get_optimizer(parameters)

        lr_scheduler = ReduceLROnPlateau(
            optimizer=optimizer, mode='min',
            factor=self.hparams.lr_scheduler_factor,
            patience=self.hparams.lr_scheduler_patience,
            threshold=0.05
        )

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch", "frequency": self.hparams.check_val_every_n_epoch,
            "monitor": "val_loss", "strict": True
        }
        optimizer_list = [optimizer]
        lr_scheduler_config_list = [lr_scheduler_config]

        return optimizer_list, lr_scheduler_config_list


class BertBinary(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            "--train_data_path", type=str,
            default='data/fever_data/fever-train.jsonl'
        )
        parser.add_argument(
            "--dev_data_path", type=str,
            default='data/fever_data/fever-val.jsonl'
        )
        parser.add_argument(
            "--test_data_path", type=str,
            default='data/fever_data/fever-dev-kilt.jsonl'
        )
        # parser.add_argument("--add_pooling_layer", type=bool, default=False)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=3e-5)
        parser.add_argument("--max_length", type=int, default=32)
        parser.add_argument("--weight_decay", type=int, default=0.01)
        parser.add_argument("--total_num_steps", type=int, default=10000)
        parser.add_argument("--warmup_updates", type=int, default=500)
        parser.add_argument("--num_workers", type=int, default=32)
        parser.add_argument("--model_name", type=str, default="bert-base-uncased")
        parser.add_argument("--eps", type=float, default=0.1)

        return parser

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        model_dir = os.path.join(self.hparams.cache_dir, self.hparams.model_name) \
            if "cache_dir" in self.hparams else self.hparams.model_name
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_dir)
            self.model = BertClassifier(model_dir)
        except:
            print("The transformer cache can not be used")
            self.tokenizer = BertTokenizer.from_pretrained(self.hparams.model_name)  # have internet
            self.model = BertClassifier(self.hparams.model_name)  # have internet
        self.train_acc = Accuracy(threshold=0.0)
        self.valid_acc = Accuracy(threshold=0.0)

    def train_dataloader(self, shuffle=True):
        if not hasattr(self, "train_dataset") or not hasattr(self, "train_loader"):
            self.train_dataset = FeverData(
                tokenizer=self.tokenizer,
                data_path=self.hparams.train_data_path, max_length=self.hparams.max_length
            )
            print("The training dataset has {} data\n".format(len(self.train_dataset)))
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.train_dataset.collate_fn,
                num_workers=self.hparams.num_workers,
                shuffle=shuffle,
            )
        return self.train_loader

    def val_dataloader(self):
        if not hasattr(self, "val_dataset") or not hasattr(self, "val_loader"):
            self.val_dataset = FeverData(
                tokenizer=self.tokenizer,
                data_path=self.hparams.dev_data_path, max_length=self.hparams.max_length
            )
            print("The validation dataset has {} data\n".format(len(self.val_dataset)))
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.val_dataset.collate_fn,
                num_workers=self.hparams.num_workers,
            )
        return self.val_loader

    def test_dataloader(self):
        if not hasattr(self, "test_dataset") or not hasattr(self, "test_loader"):
            self.test_dataset = FeverData(
                tokenizer=self.tokenizer,
                data_path=self.hparams.test_data_path, max_length=self.hparams.max_length
            )
            print("The test dataset has {} data\n".format(len(self.test_dataset)))
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.test_dataset.collate_fn,
                num_workers=self.hparams.num_workers,
            )
        return self.test_loader

    def forward(self, input_ids, attention_mask, labels):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cross_entropy = F.binary_cross_entropy_with_logits(logits, labels)
        # todo KnowledgeEditor这里加入了这个entropy，不知道为什么，需要研究一下
        entropy = torch.distributions.Bernoulli(logits=logits).entropy().mean(-1)
        loss = cross_entropy - self.hparams.eps * entropy
        metric = self.train_acc(logits, labels.int())

        return {"loss": loss, "logits": logits, "metric": metric}

    def training_step(self, batch, batch_idx=None):
        logits = self.model(input_ids=batch["src_input_ids"], attention_mask=batch["src_attention_mask"])
        cross_entropy = F.binary_cross_entropy_with_logits(logits, batch['labels'])
        # todo KnowledgeEditor这里加入了这个entropy，不知道为什么，需要研究一下
        entropy = torch.distributions.Bernoulli(logits=logits).entropy().mean(-1)

        loss = cross_entropy - self.hparams.eps * entropy
        self.log("CE", cross_entropy, on_step=True, on_epoch=False, prog_bar=True, batch_size=logits.size(0))
        self.log("E", entropy, on_step=True, on_epoch=False, prog_bar=True, batch_size=logits.size(0))
        self.train_acc(logits, batch["labels"].int())
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True, batch_size=logits.size(0))
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx=None):
        logits = self.model(input_ids=batch["src_input_ids"], attention_mask=batch["src_attention_mask"])
        self.valid_acc(logits, batch["labels"].int())
        self.log("valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=logits.size(0))
        return {"logits": logits}

    def sample(self, sentences, **kwargs):
        with torch.no_grad():
            return self.model(
                **{
                    k: v.to(self.device)
                    for k, v in self.tokenizer(
                        sentences,
                        return_tensors="pt",
                        padding=True,
                        max_length=self.hparams.max_length,
                        truncation=True,
                    ).items()
                }
            )

    def test_step(self, batch, batch_idx=None):
        logits = self.model(input_ids=batch["src_input_ids"], attention_mask=batch["src_attention_mask"])
        metric = self.valid_acc(logits, batch["labels"].int())
        self.log("metric", metric, batch_size=logits.size(0))  # , on_step=False, on_epoch=True, prog_bar=True)
        return {"logits": logits, "metric": metric}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            # bias 和 LayerNorm.weight 使用的是不同的weight_decay
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            }
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.hparams.total_num_steps,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]


class BertClassifier(torch.nn.Module):
    def __init__(self, model_name, hidden_dim=768):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, *args, **kwargs):
        return self.classifier(self.model(*args, **kwargs)[1]).squeeze(-1)


### useless codes
"""
    
    def freeze_modify_module(self, module_type='output'):
        for e in self.editors:
            if module_type == 'input':
                if isinstance(e['editor'], ModifyLinearInput):
                    e['editor'].freeze_self()
            else:
                if isinstance(e['editor'], ModifyLinearOutput):
                    e['editor'].freeze_self()

    def unfreeze_modify_module(self, module_type='output'):
        for e in self.editors:
            if module_type == 'input':
                if isinstance(e['editor'], ModifyLinearInput):
                    e['editor'].unfreeze_self()
            else:
                if isinstance(e['editor'], ModifyLinearOutput):
                    e['editor'].unfreeze_self()
    
"""

