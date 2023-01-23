import jsonlines
import torch
import random
import numpy.random as rand
from torch.utils.data import Dataset
REPHRASE_CONTAIN_ORIGINAL_SENT = 0


def data_split(dataset, ratio, shuffle=True):
    res, start = {}, 0
    offsets = {n: int(len(dataset)*r) for n, r in ratio.items()}

    if shuffle:
        random.shuffle(dataset)
        toy_list = [i for i in range(len(dataset))]
        random.shuffle(toy_list)
        print("The dataset has been shuffled and the first 5 item of toy list is {}".format(toy_list[:5]))

    for n, offset in offsets.items():
        res[n] = dataset[start:start + offset]
        start += offset
    return res


class FeverData(Dataset):
    
    def __init__(self, tokenizer=None, data_path=None, max_length=32):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        self.data_path = data_path
        self.max_length = max_length
        if self.data_path is not None:
            with jsonlines.open(self.data_path) as input_file:
                for data_point in input_file:
                    self.data.append(
                        {
                            "input": data_point["input"],
                            "label": data_point["output"][0]["answer"]
                        }
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            "input": self.data[item]["input"],
            "label": self.data[item]["label"] == "SUPPORTS",
        }

    def collate_fn(self, batch):
        tokenized_inputs = self.tokenizer(
            [b["input"] for b in batch],
            return_tensors="pt",
            padding=True,
            max_length=self.max_length,
            truncation=True,    
        )
        '''
        tokenized_inputs
        {'input_ids':...    'token_type_ids':...    'attention_mask':...}
        '''
        batches = dict()
        for key, value in tokenized_inputs.items():
            batches['src_' + key] = value
        batches["labels"] = torch.tensor([b["label"] for b in batch]).float()
        batches["raw"] = batch
        return batches


class FeverEditData(Dataset):

    def __init__(self, tokenizer=None, data_path=None, max_length=32, all_rephrase=True, example_repeat=16):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = []
        self.not_data = 0
        self.data_path = data_path
        self.max_length = max_length
        self.all_rephrase = all_rephrase
        self.example_repeat = example_repeat
        if self.data_path is not None:
            with jsonlines.open(self.data_path) as f:
                for d in f:
                    if len(d["rephrases"]) > 0:
                        self.data.append({
                            "input": d["input"],
                            "rephrases": d["rephrases"],
                            "label": d["output"][0]["answer"]
                        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {
            "input": self.data[i]["input"],
            "label": self.data[i]["label"] == "SUPPORTS",
            "rephrase": self.data[i]["rephrases"] if self.all_rephrase else rand.choice(self.data[i]["rephrases"])
        }

    def collate_fn(self, b):
        """
        :return: batches = {
                    "src_input_ids": 1 x padded_sent_len,
                    "src_rephrase_ids": 1 x rephrase_num x sent_len
                    "label": label x rephrase_num
                }
        The rephrased sentences does not contain original sentence.
        """
        # For editing dataset, we just consider one model at once
        if len(b) > 1:
            raise ValueError("For editing dataset, we just consider one data point at once")
        batches = dict()
        inp = b[0]["input"]
        inp_ids = self.tokenizer(
            inp, return_tensors="pt",
            padding=True, max_length=self.max_length, truncation=True
        )

        for k, v in inp_ids.items():
            key = '{}_{}'.format('src', k)
            if self.example_repeat == 1:
                batches[key] = v
            else:
                v_ = [v for _ in range(self.example_repeat)]
                batches[key] = torch.cat(v_, dim=0)

        rephrase = b[0]["rephrase"]
        rephrase_ids = self.tokenizer(
            rephrase, return_tensors="pt", padding=True,
            max_length=self.max_length, truncation=True
        )
        for k, v in rephrase_ids.items():
            key = '{}_{}'.format('re_src', k)
            batches[key] = v
        labels = torch.tensor([b_["label"] for b_ in b]).float()
        if self.example_repeat == 1:
            batches["labels"] = labels
        else:
            batches["labels"] = torch.cat([labels for _ in range(self.example_repeat)], dim=0)
        batches["re_labels"] = torch.cat([labels for _ in range(len(rephrase))], dim=0)
        batches["raw"] = b
        return batches


class HisData(FeverEditData):
    
    def __init__(self, tokenizer):
        super(HisData, self).__init__(tokenizer=tokenizer)

    def add(self, d):
        if isinstance(d, list):
            self.data.extend(d)
        else:
            self.data.append(d)

    def clear(self):
        self.data = []


if __name__ == "__main__":
    # First we need split the entire training data in three parts, training set, validation set and edit set.
    # and we need to sample some examples randomly for testing the model on the
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--edit_ratio", type=float, default=0.1)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()
    assert args.val_ratio + args.edit_ratio + args.train_ratio == 1

    data_path = 'data/fever_data/fever-train-kilt.jsonl'
    train_data_path = 'data/fever_data/fever-train.jsonl'
    edit_data_path = 'data/fever_data/fever-edit.jsonl'
    val_data_path = 'data/fever_data/fever-val.jsonl'
    paths = {"train": train_data_path, "val": val_data_path, "edit": edit_data_path}
    all_data = []
    # if not os.path.exists(edit_data_path) and not os.path.exists(val_data_path):
    print("Loading all data")
    with jsonlines.open(data_path) as train_file:
        for data in train_file:
            all_data.append(data)

    print("Splitting data into three parts according the ratios")
    data_splits = data_split(
        all_data,
        ratio={"train": args.train_ratio, "val": args.val_ratio, "edit": args.edit_ratio}
    )

    for k, v in data_splits.items():
        print("For {} data, we got {} data points".format(k, len(v)))

    for name, path in paths.items():
        with jsonlines.open(path, 'w') as w:
            for t_d in data_splits[name]:
                w.write(t_d)
