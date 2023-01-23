import jsonlines
import random
import torch
from torch.utils.data import Dataset


def data_split(dataset, ratio, shuffle=True):
    res, start = {}, 0
    offsets = {n: int(len(dataset)*r) for n, r in ratio.items()}
    if shuffle:
        random.shuffle(dataset)
    for n, offset in offsets.items():
        res[n] = dataset[start:start + offset]
        start += offset
    return res


class Seq2SeqData(Dataset):
    def __init__(self, tokenizer, data_path, max_length=32, example_repeat=16,
                 all_views=False, return_view=5, validation=False, edit=False):
        """
        :param tokenizer:
        :param data_path:
        :param max_length:
        :param validation:
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.validation = validation
        self.edit = edit
        self.data = []
        self.example_repeat = example_repeat

        with jsonlines.open(data_path) as f:
            for d in f:
                # we only edit the data with the only one answer
                if len(d["output"]) == 1:
                    if validation:
                        self.data.append(d)
                    else:
                        for o in d["output"]:
                            self.data.append({
                                "input": d["input"], "output": o, "rephrases": d["rephrases"]
                            })
                            break

        self.max_length = max_length
        self.all_views = all_views
        self.return_view = return_view

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return {
            "src": self.data[item]["input"],
            "trg": self.data[item]["output"],
            "rephrases": random.sample(
                self.data[item]["rephrases"],
                k=min(self.return_view, len(self.data[item]["rephrases"]))) if not self.all_views else self.data[item]["rephrases"]
        }

    def collate_fn(self, batch):
        batches = {}
        for name in ("src", ) + (() if self.validation else ("trg", )):
            tokenizer_input = [b[name] for b in batch]
            tokenizer_output = self.tokenizer(
                tokenizer_input, return_tensors="pt",
                padding=True, max_length=self.max_length,
                truncation=True,
            )
            for k, v in tokenizer_output.items():
                if name == 'src' and self.edit and self.example_repeat > 1:
                    v_ = [v for _ in range(self.example_repeat)]
                    batches["{}_{}".format(name, k)] = torch.cat(v_, dim=0)
                else:
                    batches["{}_{}".format(name, k)] = v
        if self.edit:
            assert len(batch) == 1
            tokenizer_trg = self.tokenizer(
                [b["trg"][0] for b in batch], return_tensors="pt",
                padding=True, max_length=self.max_length,
                truncation=True,
            )
            for k, v in tokenizer_trg.items():
                if self.example_repeat == 1:
                    batches["{}_{}".format("trg", k)] = v
                else:
                    v_ = [v for _ in range(self.example_repeat)]
                    batches["{}_{}".format("trg", k)] = torch.cat(v_, dim=0)

            tokenize_rephrases = self.tokenizer(
                batch[0]["rephrases"],
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            )
            for k, v in tokenize_rephrases.items():
                batches['{}_{}'.format('re_src', k)] = v

        if "trg_input_ids" in batches:
            # batches["trg_input_ids"][:, 0] = self.tokenizer.eos_token_id
            b_size = batches["trg_input_ids"].size(0)
            eos = torch.tensor([[self.tokenizer.eos_token_id] for _ in range(b_size)])
            mask = torch.tensor([[1] for _ in range(b_size)])
            batches["trg_input_ids"] = torch.cat((eos, batches["trg_input_ids"]), dim=-1)
            batches["trg_attention_mask"] = torch.cat((mask, batches["trg_attention_mask"]), dim=-1)

        batches["raw"] = batch
        return batches


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--val_ratio", type=float, default=0.075)
    parser.add_argument("--edit_ratio", type=float, default=0.025)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()
    assert args.val_ratio + args.edit_ratio + args.train_ratio == 1

    p_data_path = 'data/zsre_data/structured_zeroshot-train-new_annotated_final.jsonl'
    p_test_data_path = 'data/zsre_data/structured_zeroshot-dev-new_annotated_final.jsonl'
    train_data_path = 'data/zsre_data/zsre-train.jsonl'
    edit_data_path = 'data/zsre_data/zsre-edit.jsonl'
    val_data_path = 'data/zsre_data/zsre-val.jsonl'
    test_data_path = 'data/zsre_data/zsre-dev-kilt.jsonl'
    paths = {
        "train": train_data_path, "val": val_data_path,
        "edit": edit_data_path, "test": test_data_path
    }
    data, test_data = [], []
    print("Loading data")
    with jsonlines.open(p_data_path) as data_file:
        for d in data_file:
            new_d = {
                "input": d["input"],
                "output": [o["answer"] for o in d["output"] if "answer" in o and "provenance" in o],
                "rephrases": d["rephrases"]
            }
            data.append(new_d)

    print("Loading test data")
    with jsonlines.open(p_test_data_path) as test_file:
        for d in test_file:
            new_d = {
                "input": d["input"],
                "output": [o["answer"] for o in d["output"] if "answer" in o and "provenance" in o],
                "rephrases": d["rephrases"]
            }
            test_data.append(new_d)

    print("Splitting the existing data according to the ratios")
    data_splits = data_split(
        data, ratio={"train": args.train_ratio, "val": args.val_ratio, "edit": args.edit_ratio}
    )

    data_splits["test"] = test_data

    for k, v in data_splits.items():
        print("For {} data, we got {} data points".format(k, len(v)))
    for name, path in paths.items():
        with jsonlines.open(path, 'w') as w:
            for t_d in data_splits[name]:
                w.write(t_d)


"""

Loading data
Loading test data
Splitting the existing data according to the ratios
For train data, we got 228912 data points
For val data, we got 12208 data points
For edit data, we got 3052 data points
For test data, we got 27644 data points

train-new.jsonl
{
'input': 'Adoration of the Trinity [SEP] creator', 
'output': [{'answer': 'Albrecht Dürer'}],
'meta': {
    'template_questions': ['Who is Adoration of the Trinity by?']}, 
    'rephrases': ['Who Is Worship of the Trinity Through?', 'Who is Adoration of the Trinity Through?', 
                  'Who is worshiping the Trinity through?', 'Who is worship of the Trinity by?', 
                  'Who is through the worship of the Trinity?', 'Who is the worship of the Trinity through?', 
                  'Who is through worship of the Trinity?', 'Who through the worship of the Trinity?', 
                  'Who is by worship of the Trinity?', 'Who is the Worship of the Trinity?', 
                  'Who Is Worship of the Trinity?', 'Who is worshipping the Trinity?', 'Who is worshiping the Trinity?', 
                  'Who is Adoration of the Trinity by?']
}

dev-new.jsonl
{
'input': 'Watts Humphrey [SEP] educated at', 
'output': [{'answer': 'Illinois Institute of Technology'}], 
'meta': {
    'template_questions': ['What university did Watts Humphrey attend?']}, 
    'rephrases': ['Which university did Watts Humphrey attend?', 'Which university has Watts Humphrey attended?', 
                  'Which university did Watts Humphrey go to?', 'Which university has Watts Humphrey visited?', 
                  'Which university attended Watts Humphrey?', 'Which university did Watts go to Humphrey?', 
                  'What university did Watts attend Humphrey at?', 'Which university did Watts attend Humphrey?', 
                  'What university did Watts attend Humphrey?', 'What university did Watts go to Humphrey?', 
                  'Which university did Watts Humphrey take part in?', 'What university did Watts Humphrey take part in?', 
                  'Which university did Watts Humphrey participate in?', 'Which university did Watts Humphrey study at?', 
                  'What university did Watts Humphrey study at?', 'What university did Watts Humphrey go to?', 
                  'What university did Watts Humphrey attend?'], 
}

train-new_annotated_final.jsonl
{
'input': 'Who is Adoration of the Trinity by?', 
'output': [{'answer': 'Albrecht Dürer'}], 
'rephrases': ['Who Is Worship of the Trinity Through?', 'Who is Adoration of the Trinity Through?', 'Who is worshiping the Trinity through?', 
              'Who is worship of the Trinity by?', 'Who is through the worship of the Trinity?', 'Who is the worship of the Trinity through?', 
              'Who is through worship of the Trinity?', 'Who through the worship of the Trinity?', 'Who is by worship of the Trinity?', 
              'Who is the Worship of the Trinity?', 'Who Is Worship of the Trinity?', 'Who is worshipping the Trinity?', 
              'Who is worshiping the Trinity?', 'Who is Adoration of the Trinity by?'], 
}


{
'input': 'What university did Watts Humphrey attend?', 
'output': [{'answer': 'Illinois Institute of Technology']}], 
'meta': {
    'template_questions': ['What university did Watts Humphrey attend?']}, 
    'rephrases': ['Which university did Watts Humphrey attend?', 'Which university has Watts Humphrey attended?', 
                  'Which university did Watts Humphrey go to?', 'Which university has Watts Humphrey visited?', 
                  'Which university attended Watts Humphrey?', 'Which university did Watts go to Humphrey?', 
                  'What university did Watts attend Humphrey at?', 'Which university did Watts attend Humphrey?', 
                  'What university did Watts attend Humphrey?', 'What university did Watts go to Humphrey?', 
                  'Which university did Watts Humphrey take part in?', 'What university did Watts Humphrey take part in?', 
                  'Which university did Watts Humphrey participate in?', 'Which university did Watts Humphrey study at?', 
                  'What university did Watts Humphrey study at?', 'What university did Watts Humphrey go to?', 
                  'What university did Watts Humphrey attend?'], 
}

"""
