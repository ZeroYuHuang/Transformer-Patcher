import os
import pickle
import random
import numpy as np
from .fever_dataloader import FeverData, FeverEditData
from .zsre_dataloader import Seq2SeqData
from torch.utils.data import DataLoader, Subset, random_split, Dataset


class PeusodoData(Dataset):

    def __init__(self, d):
        super().__init__()
        self.d = [d]

    def __len__(self):
        return len(self.d)

    def __getitem__(self, item):
        return self.d[item]

    def collate_fn(self, batch):
        return batch[0]


class SeqEditDataSet(object):

    def __init__(self, task_name='fever', tokenizer=None, data_path=None,
                 train_sub_size=10000, memory_size=20000, edit_folder_num=20, edit_data=None,
                 batch_size=128, num_workers=1, loss_as_val_metric=True, example_repeat=16):

        self.tokenizer = tokenizer
        self.train_sub_size = train_sub_size
        self.memory_size = memory_size
        self.edit_folder_num = edit_folder_num
        self.batch_size = batch_size
        self.task_name = task_name
        self.num_workers = num_workers
        self.example_repeat = example_repeat

        train_path = os.path.join(data_path, '{}-train.jsonl'.format(task_name))
        edit_path = os.path.join(data_path, '{}-edit.jsonl'.format(task_name))
        val_path = os.path.join(data_path, '{}-val.jsonl'.format(task_name))
        dev_path = os.path.join(data_path, '{}-dev-kilt.jsonl'.format(task_name))

        # self.train_path = train_path
        # self.edit_path = edit_path
        # self.val_path = val_path
        # self.dev_path = dev_path

        # Creating datasets
        if task_name == 'fever':
            self.train_data_as_val = FeverData(tokenizer=tokenizer, data_path=train_path)
            self.train_data_as_memo = self.train_data_as_val
            self.edit_test_data = FeverData(tokenizer=tokenizer, data_path=edit_path)
            self.edit_data = FeverEditData(tokenizer=tokenizer, data_path=edit_path, example_repeat=self.example_repeat)
            # Different editing method may require different data processing method
            # self.edit_data = edit_data
            self.dev_data = FeverData(tokenizer=tokenizer, data_path=dev_path)
            self.val_data = FeverData(tokenizer=tokenizer, data_path=val_path)
        elif task_name == 'zsre':
            self.train_data_as_val = Seq2SeqData(tokenizer=tokenizer, data_path=train_path, validation=True)
            self.train_data_as_memo = Seq2SeqData(tokenizer=tokenizer, data_path=train_path, validation=False)
            self.edit_test_data = Seq2SeqData(tokenizer=tokenizer, data_path=edit_path, validation=True)
            self.edit_data = Seq2SeqData(tokenizer=tokenizer, data_path=edit_path, edit=True, validation=True, example_repeat=self.example_repeat)
            self.dev_data = Seq2SeqData(tokenizer=tokenizer, data_path=dev_path, validation=True)
            # self.val_data = Seq2SeqData(tokenizer=tokenizer, data_path=val_path, validation=not loss_as_val_metric)
            # we always use the validation_loss for seq2seq task as the validation metric
            self.val_data = Seq2SeqData(tokenizer=tokenizer, data_path=val_path, validation=False)

        # Splitting edit into 'edit_folder_num' subsets
        self.edit_folder = self.split_edit_into_folder()

        # Sample a testing subset and memory of training set
        self.train_sub, self.memory_set = self.get_train_sub_and_memory(
            self.train_data_as_val, self.train_data_as_memo, self.train_sub_size, self.memory_size
        )

        # creating DataLoaders
        # self.train_loader = DataLoader(self.train_data, batch_size=256, collate_fn=self.train_data.collate_fn)
        try:
            self.train_sub_loader = DataLoader(
                self.train_sub, batch_size=self.batch_size, collate_fn=self.train_data_as_val.collate_fn, num_workers=num_workers
            )
            self.memory_loader = DataLoader(
                self.memory_set, batch_size=self.batch_size, collate_fn=self.train_data_as_memo.collate_fn, num_workers=num_workers, shuffle=True
            )
            self.val_loader = DataLoader(
                self.val_data, batch_size=batch_size, collate_fn=self.val_data.collate_fn, num_workers=num_workers
            )
        except:
            print(self.train_sub)
            print(self.memory_set)
            print(self.val_data)

        self.edit_test_loader = DataLoader(
            self.edit_test_data, batch_size=self.batch_size, collate_fn=self.train_data_as_val.collate_fn, num_workers=num_workers
        )
        self.edit_folder_loader = [
            DataLoader(dataset=e, batch_size=1, collate_fn=self.edit_data.collate_fn, shuffle=True, num_workers=num_workers)
            for e in self.edit_folder
        ]
        self.dev_loader = DataLoader(
            self.dev_data, batch_size=self.batch_size, collate_fn=self.dev_data.collate_fn, num_workers=num_workers
        )

    def reset_example_repeat(self, er):
        self.edit_data.example_repeat = er

    def re_split_train_sub_and_memory(self, t_size, m_size):
        self.train_sub_size, self.memory_size = t_size, m_size
        self.train_sub, self.memory_set = self.get_train_sub_and_memory(
            self.train_data_as_val, self.train_data_as_memo, self.train_sub_size, self.memory_size
        )
        self.train_sub_loader = DataLoader(
            self.train_sub, batch_size=self.batch_size, collate_fn=self.train_data_as_val.collate_fn, num_workers=self.num_workers
        )
        self.memory_loader = DataLoader(
            self.memory_set, batch_size=self.batch_size, collate_fn=self.train_data_as_memo.collate_fn, num_workers=self.num_workers
        )

    def re_set_loaders(self, new_num_workers, new_batch_size):
        self.num_workers, self.batch_size = new_num_workers, new_batch_size
        self.train_sub_loader = DataLoader(
            self.train_sub, batch_size=self.batch_size, collate_fn=self.train_data_as_val.collate_fn, num_workers=self.num_workers
        )

        self.memory_loader = DataLoader(
            self.memory_set, batch_size=self.batch_size, collate_fn=self.train_data_as_memo.collate_fn, num_workers=self.num_workers
        )
        self.edit_test_loader = DataLoader(
            self.edit_test_data, batch_size=self.batch_size, collate_fn=self.train_data_as_val.collate_fn, num_workers=self.num_workers
        )
        self.edit_folder_loader = [
            DataLoader(dataset=e, batch_size=1, collate_fn=self.edit_data.collate_fn, shuffle=True, num_workers=self.num_workers)
            for e in self.edit_folder
        ]
        self.dev_loader = DataLoader(
            self.dev_data, batch_size=self.batch_size, collate_fn=self.dev_data.collate_fn, num_workers=self.num_workers
        )
        self.val_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, collate_fn=self.val_data.collate_fn, num_workers=self.num_workers
        )

    def shuffle_memory_loader(self):
        self.memory_loader = DataLoader(
            self.memory_set, batch_size=self.batch_size, collate_fn=self.train_data_as_memo.collate_fn,
            num_workers=self.num_workers, shuffle=True
        )

    def split_edit_into_folder(self):
        len_per_folder = len(self.edit_data) // self.edit_folder_num + 1
        lengths, i = [], 0
        while i + len_per_folder < len(self.edit_data):
            lengths.append(len_per_folder)
            i += len_per_folder
        lengths.append(len(self.edit_data) - i)
        edit_folder = random_split(dataset=self.edit_data, lengths=lengths)
        return edit_folder

    @staticmethod
    def get_train_sub_and_memory(train_data_as_val, train_data_as_memo, train_sub_size, memory_size):
        if train_data_as_val is None or train_data_as_memo is None:
            return None, None
        assert len(train_data_as_val) == len(train_data_as_memo)
        indices = [i for i in range(len(train_data_as_val))]
        random.shuffle(indices)
        if 0 < train_sub_size < 1:
            train_sub_size *= len(train_data_as_val)
        if 0 < memory_size < 1:
            memory_size *= len(train_data_as_memo)
        train_sub = Subset(dataset=train_data_as_val, indices=indices[:train_sub_size])
        memory_set = Subset(dataset=train_data_as_memo, indices=indices[train_sub_size:train_sub_size+memory_size])

        return train_sub, memory_set

    @staticmethod
    def get_subset(all_set, subset_size=200):
        indices = [i for i in range(len(all_set))]
        random.shuffle(indices)
        return Subset(dataset=all_set, indices=indices[:subset_size])


class SeqEditResOutput(object):

    def __init__(self, edit_folder_num=20, save_dir='./'):
        self.save_dir = save_dir
        self.edit_folder_num = edit_folder_num
        self.init_metric = {}
        # record edit is successful or not
        self.edit = {f: [] for f in range(edit_folder_num)}
        self.ber = {f: [] for f in range(edit_folder_num)}  # before_editing_rephrases
        self.aer = {f: [] for f in range(edit_folder_num)}  # after_editing_rephrases
        # self.gen = {f: [] for f in range(edit_folder_num)}
        # record the metric on learnt training data and public test data
        self.test = {f: [] for f in range(edit_folder_num)}
        self.train = {f: [] for f in range(edit_folder_num)}
        # record the metric on historical edit data
        self.his = {f: [] for f in range(edit_folder_num)}
        self.his_re = {f: []for f in range(edit_folder_num)}
        # record the add neuron num
        self.add_neuron_num = {f: [] for f in range(edit_folder_num)}

    def feed_ts(self, ts):
        self.ts = ts

    def normalize(self, task_type=None):
        if task_type:
            if 'bart' in task_type.lower() or 'zsqa' in task_type.lower():
                init_metric = {'test': 0.23055174946784973, 'train': 0.566100001335144}
            else:
                init_metric = {'test': 0.7688624858856201, 'train': 0.9406999945640564}
        else:
            init_metric = self.init_metric
        # we normalize the metric
        for n_key, n_object in (("test", self.test), ("train", self.train)):
            for f_k, f_v in n_object.items():
                for i in range(len(f_v)):
                    f_v[i] /= self.init_metric[n_key]

    def save_as_file(self):
        output = open(os.path.join(self.save_dir, 'res.pkl'), 'wb')  # 若已经存在，则覆盖写
        s = pickle.dumps(self)
        output.write(s)
        output.close()

    def get_res(self):
        edit_suc, edit_num = 0, 0
        for e in self.edit.values():
            edit_suc += np.sum(e)
            edit_num += len(e)
        # calculating GR
        edit_gen, edit_gen_number = 0, 0
        for e in self.aer.values():
            for (acc, num) in e:
                edit_gen += acc * num
                edit_gen_number += num
        return {
            # 'edit': self.average_dict(self.edit),
            'SR_t': self.average_dict(self.edit), 'SR': edit_suc / edit_num,
            'GR_t': self.average_dict(self.aer), 'GR': edit_gen / edit_gen_number,
            'LRR_t': self.average_dict(self.train), 'LRR': self.average_final(self.ts['train'], self.train),
            'GRR_t': self.average_dict(self.test), 'GRR': self.average_final(self.ts['test'], self.test),
            'ERR_t': self.average_dict(self.his), "ERR": self.average_final(self.ts['his'], self.his),
            # 'ERR_et': self.average_dict(self.his_re),
            # "ERR_e": self.average_final(self.ts, self.his_re),
        }
    @staticmethod
    def average_final(ts, metric: dict):
        average = []
        assert len(ts) == len(metric)
        for f, m in metric.items():
            if ts[f] is not None:
                average.append(m[ts[f]])
        return np.mean(average)

    def average_dict(self, d):
        res = []
        # ll = sorted([len(v) for v in d.values()])[self.edit_folder_num // 3 * 2]
        # ll = min([len(v) if v not in self.absent_folders else 100000 for v in d.values()])
        # ll = max([len(v) for v in d.values()])
        L = int(np.mean([len(v) for v in d.values()]))
        for i in range(L):
            # print(d)
            for jj in d.values():
                if len(jj) > 0:
                    istuple = isinstance(jj[0], tuple)
            if not istuple:
                tmp = [d[f][i] for f in range(self.edit_folder_num) if i < len(d[f])]
            else:
                tmp = [d[f][i][0] for f in range(self.edit_folder_num) if i < len(d[f])]
            res.append((np.mean(tmp), np.sqrt(np.var(tmp))))
        return res





