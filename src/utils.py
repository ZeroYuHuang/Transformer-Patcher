import logging
import os
import torch
import random
import pickle
import torch.nn.functional as F
from torch.utils.data import random_split
from pytorch_lightning import LightningModule
NUM_BEAMS = 1


def main_args(parser):
    parser.add_argument('--task', type=str, default='fever', help='fever||zsre')
    parser.add_argument('--method', type=str, default='T-patch', help='T-patch||ft')
    parser.add_argument('--edit_folder_num', type=int, default=20)
    parser.add_argument('--process_folders', type=str, default='all_folders', help='all_folders||seg_10_20||[1,5,3]')
    parser.add_argument('--task_id', type=str, default=None, help='name for logging txt')
    parser.add_argument('--gpu_nums', type=int, default=8)
    parser.add_argument('--tasks_per_gpu', type=int, default=2)
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--log_name', type=str, default='log.txt')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_path', type=str, default='log/model.ckpt')
    parser.add_argument('--train_sub_size', type=int, default=10000)
    parser.add_argument('--memory_size', type=int, default=40000)
    parser.add_argument('--debug_mode', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--example_repeat', type=int, default=8, help='we concatenate many examples in to one batch')
    parser.add_argument('--temp_mode', type=int, default=0, help='We test after every edit if tmp_mode is set to 1')
    parser.add_argument('--get_heat_map', type=int, default=0)  # set to 1 if you want to save the activation values
    parser.add_argument('--max_edit_step', type=int, default=2000)

    return parser


def patch_related_args(parser):
    """
    For both fever and classification and seq2seq module
    """
    # early stopping hp
    parser.add_argument('--early_patience', type=int, default=1)
    parser.add_argument('--early_mode', type=str, default='max')
    parser.add_argument('--early_thd', type=float, default=0.01)
    parser.add_argument('--start_val_epoch', type=int, default=100)

    # checkpoint hp
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--ckpt_monitor", default="save_ckpt", type=str)
    parser.add_argument("--ckpt_metric_mode", default="max", type=str)

    # optimizer hp
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5)
    parser.add_argument('--lr_scheduler_patience', type=int, default=1)

    # initialization hp
    parser.add_argument('--use_init_weight', type=int, default=1)
    parser.add_argument('--amplify_v', type=int, default=1)
    parser.add_argument('--amplify_con', type=float, default=10.0)
    parser.add_argument('--freeze_a', type=int, default=0)
    parser.add_argument('--freeze_k', type=int, default=0)

    # trainer hp
    parser.add_argument('--check_val_every_n_epoch', type=int, default=25)

    # memory hp
    parser.add_argument('--memory_loss', type=str, default='top1000_exp+top1000_exp')
    parser.add_argument('--mlc', type=float, default=5)
    parser.add_argument('--update_memory', type=int, default=1)
    parser.add_argument('--margin_val1', type=float, default=3)
    parser.add_argument('--margin_val2', type=float, default=-3)

    # activate hp
    parser.add_argument('--activate_loss', type=str, default='top5_exp')  # margin|exp|non_use
    parser.add_argument('--act_loss_thd', type=float, default=0.1)  # the thd for stopping the training
    parser.add_argument('--alc', type=float, default=1.0)
    parser.add_argument('--act_margin_val', type=float, default=0.0)

    # freeze hp
    parser.add_argument('--freeze_model', type=bool, default=True)
    parser.add_argument('--val_metric_type', type=str, default='loss')
    parser.add_argument('--max_add_neuron_num', type=int, default=5)

    # use_val = 0: we edit just until the editing example is corrected
    # use_val = 1: we edit and use an external validation set to decide when to early stop
    parser.add_argument('--use_val', type=int, default=1)

    return parser


def ft_related_args(parser):
    # parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--ft_optim', type=str, default='adam')
    parser.add_argument('--ft_lr', type=float, default=1e-5)
    parser.add_argument('--use_kl', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--loc_num', type=int, default=512)
    parser.add_argument('--ft_update_memory', type=int, default=1)
    parser.add_argument('--layer', default='11', type=str)

    return parser


def save_obj(obj, name):
    output = open(name, 'wb')  # 若已经存在，则覆盖写
    s = pickle.dumps(obj)
    output.write(s)
    output.close()


def load_obj(name):
    with open(name, 'rb') as f:
        res = pickle.load(f)
    f.close()
    return res


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def lower_and_strip_list(inp):
    return [i.lower().strip() for i in inp]


def get_time_post_fix():
    import time
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())


def kl_loc_loss(pre, post, mask=None):
    pre = pre.to(torch.float32)
    post = post.to(torch.float32)

    sequence = pre.dim() == 3
    pre_ = pre.view(-1, pre.shape[-1])
    post_ = post.view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]

    if not sequence:
        if pre_.shape[-1] == 1:  # No masking needed for binary classification
            return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
                (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
    else:  # We have sequences of predictions; masking needed
        if pre_.shape[-1] > 1:
            # assert mask is not None
            kl = (pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))).sum(-1)
            mask_ = mask.contiguous().view(pre_.shape[0])
            return (kl * mask_).sum() / mask_.sum()

    raise NotImplementedError


def get_ol_and_pl(o_model, p_model, batch, device, mode='class'):
    if mode == 'seq2seq':
        with torch.no_grad():
            ol = o_model(
                batch['src_input_ids'].to(device), batch['src_attention_mask'].to(device),
                batch['trg_input_ids'][:, :-1].to(device), batch['trg_attention_mask'][:, :-1].to(device)
            )
        pl = p_model(
            batch['src_input_ids'].to(device), batch['src_attention_mask'].to(device),
            batch['trg_input_ids'][:, :-1].to(device), batch['trg_attention_mask'][:, :-1].to(device)
        )
    else:
        with torch.no_grad():
            ol = o_model(
                batch['src_input_ids'].to(device), batch['src_attention_mask'].to(device), batch['labels'].to(device)
            )['logits']
        pl = p_model(
            batch['src_input_ids'].to(device),
            batch['src_attention_mask'].to(device),
            batch['labels'].to(device)
        )['logits']
        ol = ol.unsqueeze(1) if ol.dim() == 1 else ol
        pl = pl.unsqueeze(1) if pl.dim() == 1 else pl
    torch.cuda.empty_cache()
    return ol, pl


def get_kl_diver_loss(original_model, post_model, memo_loader, device,
                      total_loc_num, his_edit_data):
    his_edit_len = len(his_edit_data) if his_edit_data else 0
    kl_loss, loc_num = 0, 0
    post_model.eval().to(device)
    original_model.eval().to(device)
    for batch_id, batch in enumerate(memo_loader):
        batch_size = batch['src_input_ids'].size(0)
        ol, pl = get_ol_and_pl(o_model=original_model, p_model=post_model, batch=batch, device=device,
                               mode='seq2seq' if 'trg_attention_mask' in batch.keys() else 'class')
        mask = batch['trg_attention_mask'][:, :-1].to(device) if 'trg_attention_mask' in batch.keys() else None
        kl_loss += kl_loc_loss(pre=ol, post=pl, mask=mask) * batch_size
        loc_num += batch_size
        if loc_num + batch_size >= total_loc_num - his_edit_len:
            break
        kl_loss /= loc_num
    if his_edit_len > 0:
        kl_his_loss = 0
        h = his_edit_data if len(his_edit_data) < 100 else random.sample(his_edit_data, 100)
        for batch in h:
            ol, pl = get_ol_and_pl(o_model=original_model, p_model=post_model, batch=batch, device=device,
                                   mode='seq2seq' if 'trg_attention_mask' in batch.keys() else 'class')
            mask = batch['trg_attention_mask'][:, :-1].to(device) if 'trg_attention_mask' in batch.keys() else None
            kl_his_loss += kl_loc_loss(pre=ol, post=pl, mask=mask)
        kl_his_loss /= his_edit_len
        kl_loss += kl_his_loss
    return kl_loss


def split_data_n_sets(d, n_set):
    data_len = len(d) // n_set
    data_size, i = [], 0
    while i + data_len < len(d):
        data_size.append(data_len)
        i += data_len
    data_size.append(len(d) - i)
    edit_sets_list = random_split(dataset=d, lengths=data_size)
    return edit_sets_list


def get_handler(path, log_name):
    log_file_path = os.path.join(path, log_name)
    try:
        if not os.path.exists(path):
            print("We are creating the logger files")
            os.makedirs(path)
    except:
        pass
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    return file_handler, stream_handler


def my_test_binary(model, data_loader, device, mode='original', tokenizer=None):
    # mode is 'original' or 'rephrases'
    with torch.no_grad():
        model.eval()
        model.to(device)
        data_len, correct_num = 0, 0
        for _, batch in enumerate(data_loader):
            if mode == 'rephrases':
                input_ids = batch["re_src_input_ids"].to(device)
                attention_mask, labels = batch["re_src_attention_mask"].to(device), batch["re_labels"].to(device)
            else:
                input_ids = batch["src_input_ids"].to(device)
                attention_mask, labels = batch["src_attention_mask"].to(device), batch["labels"].to(device)
            if isinstance(model, LightningModule):
                # print('This could be our BertBinary Class')
                logits = model.model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                # print('This could be the BertClassifer model')
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = logits > 0
            pred = pred.clone().cpu()
            correct_num += torch.sum(torch.eq(pred, labels.cpu()))
            data_len += labels.size(0)
            torch.cuda.empty_cache()
        return correct_num.cpu() / data_len, data_len


def my_test_seq2seq(model, data_loader, device, mode='original', tokenizer=None):
    correct_count, total_count = 0, 0
    import pdb
    if tokenizer is None:
        tokenizer = model.tokenizer
    with torch.no_grad():
        model.eval()
        model.to(device)
        for _, batch in enumerate(data_loader):
            trg = [b["trg"] for b in batch["raw"]]
            if mode == 'original':
                input_ids = batch["src_input_ids"].to(device)
                attention_mask = batch["src_attention_mask"].to(device)
            else:
                input_ids = batch["re_src_input_ids"].to(device)
                attention_mask = batch["re_src_attention_mask"].to(device)
                # trg: [["answer1", "answer2", "answer_n"]]
                trg = [b["trg"] for b in batch["raw"] for _ in range(len(b["rephrases"]))]
            if isinstance(model, LightningModule):
                model_gen = model.model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    min_length=0, num_beams=NUM_BEAMS, num_return_sequences=1, use_cache=True
                )
                pred = model.tokenizer.batch_decode(model_gen, skip_special_tokens=True)
            else:
                model_gen = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    min_length=0, num_beams=NUM_BEAMS, num_return_sequences=1, use_cache=True
                )
                pred = tokenizer.batch_decode(model_gen, skip_special_tokens=True)
            acc = torch.tensor(
                [
                    p.lower().strip() in [t_.lower().strip() for t_ in t]
                    for t, p in zip(trg, pred)
                ]
            ).long()

            correct_count += torch.sum(acc)
            total_count += acc.size(0)

        return correct_count / total_count, total_count


def edit_or_not_binary(model, data_point, device, args=None):
    with torch.no_grad():
        model.eval()
        model.to(device)
        batch = data_point
        if args is not None:
            if args.use_init_weight:
                model.unlock_hidden_detectors()
        res = model(
            batch["src_input_ids"][[0]].to(device), batch["src_attention_mask"][[0]].to(device),
            batch["labels"][[0]].to(device),
        )
        if args is not None:
            if args.use_init_weight:
                model.lock_hidden_detectors()

        re_res = model(
            batch["re_src_input_ids"].to(device), batch["re_src_attention_mask"].to(device),
            batch["re_labels"].to(device),
        )
        return 1 - res['metric'].cpu(), re_res['metric'].cpu(), re_res['logits'].size(0)


def edit_or_not_seq2seq(model, data_point, device, test_rephrases=True, args=None):

    with torch.no_grad():
        model.eval()
        model.to(device)
        batch = data_point

        prediction = model.model.generate(
            input_ids=batch["src_input_ids"][[0]].to(device),
            attention_mask=batch["src_attention_mask"][[0]].to(device),
            min_length=0, num_beams=NUM_BEAMS, num_return_sequences=1, use_cache=True
        )
        prediction = model.tokenizer.batch_decode(prediction, skip_special_tokens=True)
        prediction = lower_and_strip_list(prediction)
        targets = lower_and_strip_list(batch["raw"][0]["trg"])
        need_edit = 1 if prediction[0] not in targets else 0
        if test_rephrases:
            prediction_re = model.model.generate(
                input_ids=batch["re_src_input_ids"].to(device),
                attention_mask=batch["re_src_attention_mask"].to(device),
                min_length=0, num_beams=NUM_BEAMS, num_return_sequences=1, use_cache=True
            )
            prediction_re = lower_and_strip_list(model.tokenizer.batch_decode(prediction_re, skip_special_tokens=True))
            correct_count = 0
            for p in prediction_re:
                correct_count += float(p in targets)
            correct_count /= len(prediction_re)
            re_num = len(prediction_re)
        else:
            correct_count = 0
            re_num = 0
        return need_edit, correct_count, re_num


def count_error_nums(model, data_point, device):

    with torch.no_grad():
        model.eval()
        model.to(device)
        batch = data_point

        # model.unlock_hidden_detectors()
        # since the original batch may be repeated
        raw_logits = model(
            input_ids=batch["src_input_ids"][[0]].to(device),
            attention_mask=batch["src_attention_mask"][[0]].to(device),
            decoder_input_ids=batch["trg_input_ids"][[0], :-1].to(device),
            decoder_attention_mask=batch["trg_attention_mask"][[0], :-1].to(device)
        )
        pred = torch.argmax(raw_logits, dim=-1)
        trg = batch["trg_input_ids"][[0], 1:].to(device)
        select_index = [i for i, s in enumerate(list((pred != trg).cpu().squeeze())) if s]

    return len(select_index), select_index


def get_seq2seq_loss(model, data_loader, device):
    with torch.no_grad():
        model.eval()
        model.to(device)
        total_loss, total_tokens = 0, 0
        for batch_id, batch in enumerate(data_loader):
            input_ids = batch["src_input_ids"].to(device)
            attention_mask = batch["src_attention_mask"].to(device)
            decoder_input_ids = batch["trg_input_ids"][:, :-1].to(device)
            decoder_attention_mask = batch["trg_attention_mask"][:, :-1].to(device)
            logits = model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
            loss, _ = label_smoothed_nll_loss(
                logits.log_softmax(-1), batch["trg_input_ids"][:, 1:].to(device),
                epsilon=model.hparams.eps, ignore_index=model.tokenizer.pad_token_id,
            )
            total_loss += loss
            total_tokens += batch["trg_attention_mask"][:, 1:].sum()
        total_loss.cpu()
        total_tokens.cpu()

        return total_loss / total_tokens


def echo(log, info_dict):
    for k, v in info_dict.items():
        log.info("{}:{}".format(k, v))
















