import os
import sys
import torch
import argparse
import logging
import pickle
from copy import deepcopy
sys.path.append('.')
sys.path.append('..')
LOG = logging.getLogger(__name__)
LOG.setLevel(level=logging.DEBUG)


def get_optimizer(optim, params, lr):
    if optim == 'adam':
        return torch.optim.Adam(params=params, lr=lr)
    if optim == 'rmsprop':
        return torch.optim.RMSprop(params=params, lr=lr)
    if optim == 'sgd':
        return torch.optim.SGD(params=params, lr=lr)


def get_train_r_test_r():
    test_r = TEST_DICT[args.task](t_model, seq_edit_data.dev_loader, args.device)
    train_r = TEST_DICT[args.task](t_model, seq_edit_data.train_sub_loader, args.device)
    res_out.test[f].append(test_r[0])
    LOG.info(f"Test Retain Rate: {test_r[0]}")
    res_out.train[f].append(train_r[0])
    LOG.info(f"Train Retain Rate {train_r[0]}")


def get_er():
    if len(his_edit_data) > 0:
        LOG.info("Testing on the history edit dataset")
        er, _ = TEST_DICT[args.task](t_model, his_edit_data, args.device, 'original')
        res_out.his[f].append(er)
        LOG.info(f"Model attains {er} on past edit examples")


if __name__ == '__main__':
    from src.models.class_modules import BertBinary
    from src.models.seq2seq_modules import BartSeq2Seq
    from src.dataset.sme_dataset import SeqEditResOutput
    from src.utils import echo, get_handler, label_smoothed_nll_loss, load_obj, save_obj, main_args, ft_related_args, \
        my_test_seq2seq, my_test_binary, edit_or_not_seq2seq, edit_or_not_binary, get_kl_diver_loss

    TEST_DICT = {'zsqa': my_test_seq2seq, 'fever': my_test_binary, }
    EDIT_DICT = {'zsqa': edit_or_not_seq2seq, 'fever': edit_or_not_binary}

    parser = argparse.ArgumentParser()
    '''adding the common arguments'''
    parser = main_args(parser)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--gpus', type=list, default=[0])
    parser.add_argument('--fold_n', type=int, default=25)

    '''adding the args related to the fine_tune method'''
    parser = ft_related_args(parser)

    args, _ = parser.parse_known_args()

    args.gpus = [args.device]
    args.device = torch.device('cuda', args.device)
    args.log_path = os.path.join(args.log_path, 'fold_{}'.format(args.fold_n))
    f_h, s_h = get_handler(args.log_path, log_name='fine_tuning.txt')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

    for k, v in vars(args).items():
        LOG.info(f'{k}:{v}')

    # Loading edit data
    LOG.info('Loading data')
    with open(os.path.join(args.data_path, 'seq_edit_data_{}_{}.pkl'.format(args.task, args.edit_folder_num)), 'rb') as file:
        seq_edit_data = pickle.loads(file.read())
    if args.train_sub_size != len(seq_edit_data.train_sub) or args.memory_size != len(seq_edit_data.memory_set):
        seq_edit_data.re_split_train_sub_and_memory(args.train_sub_size, args.memory_size)
    if args.num_workers != seq_edit_data.num_workers or args.batch_size != seq_edit_data.batch_size:
        seq_edit_data.re_set_loaders(args.num_workers, args.batch_size)
    if args.example_repeat != seq_edit_data.example_repeat:
        seq_edit_data.reset_example_repeat(er=args.example_repeat)
    seq_edit_data.shuffle_memory_loader()
    echo(LOG, {f'The size of {k} is': len(getattr(seq_edit_data, k)) for k in ('train_sub', 'memory_set', 'edit_test_data', 'dev_data', 'val_data')})
    is_continue = False
    if os.path.exists(os.path.join(args.log_path, 'model.ckpt')) and os.path.exists(os.path.join(args.log_path, 'edit_schedule.txt')) and os.path.exists(os.path.join(args.log_path, 'res.pkl')):
        is_continue = True
        LOG.info('Loading result file')
        with open(os.path.join(args.log_path, 'res.pkl'), 'rb') as res_file:
            res_out = pickle.loads(res_file.read())
        res_file.close()
        LOG.info('Loading edit start index')
        with open(os.path.join(args.log_path, 'edit_schedule.txt')) as edit_schedule_file:
            edit_start_index = int(edit_schedule_file.readline())
            edit_times = int(edit_schedule_file.readline())
        edit_schedule_file.close()
        LOG.info('Loading model')
        model_to_edit = torch.load(os.path.join(args.log_path, 'model.ckpt'))
        LOG.info('Loading hih_edit_data')
        his_edit_data = load_obj(os.path.join(args.log_path, 'his_edit_data.pkl'))
    else:
        LOG.info('Loading model to edit and creating result class')
        if args.task == 'fever':
            model_to_edit = BertBinary.load_from_checkpoint(args.model_path)
        else:
            model_to_edit = BartSeq2Seq.load_from_checkpoint(args.model_path)
        res_out = SeqEditResOutput(edit_folder_num=args.edit_folder_num, save_dir=args.log_path)
        his_edit_data, edit_times, edit_start_index = [], 0, 0
    LOG.info('Freezing the original model')
    for p in model_to_edit.parameters():
        p.requires_grad = False

    # Print all tunable parameters
    not_ft_params_name = ('bias', 'norm', 'embeddings', 'classifier', 'pooler', 'shared', 'embed', 'positions')
    if args.task == 'fever':
        names = [
            n for n, p in model_to_edit.named_parameters()
            if ((args.layer != 'all' and f'.{args.layer}.' in n) or args.layer == 'all') and all(e not in n.lower() for e in not_ft_params_name)
        ]
    else:
        names = [
            n for n, p in model_to_edit.named_parameters()
            if ((args.layer != 'all' and f'decoder.layers.{args.layer}.' in n) or args.layer == 'all') and all(e not in n.lower() for e in not_ft_params_name)
        ]

    LOG.info('The name of the tunable parameters are as below')
    for n in names:
        LOG.info(n)

    model_to_edit.eval().to(args.device)

    LOG.info('Before editing, we get the acc on train, dev and edit set')
    t_model = deepcopy(model_to_edit)

    if args.debug_mode == 0 and not is_continue:
        res_out.init_metric['test'], _ = TEST_DICT[args.task](t_model, data_loader=seq_edit_data.dev_loader, device=args.device)
        res_out.init_metric['train'], _ = TEST_DICT[args.task](model=t_model, data_loader=seq_edit_data.train_sub_loader, device=args.device)
        edit_acc, _ = TEST_DICT[args.task](t_model, seq_edit_data.edit_test_loader, args.device)
        echo(LOG, {f"The acc on {k} is": res_out.init_metric[k] if k != 'edit' else edit_acc for k in ('test', 'train', 'edit')})

    f, edit_folder = args.fold_n, seq_edit_data.edit_folder_loader[args.fold_n]
    t_model = deepcopy(model_to_edit)
    LOG.info('\n\n')
    assert args.task in ['zsqa', 'fever']
    if args.task == 'zsqa':
        for n, p in t_model.named_parameters():
            if n in names:
                p.requires_grad = True
    else:
        for n, p in t_model.named_parameters():
            if n in names:
                p.requires_grad = True
    params = [p for p in t_model.parameters() if p.requires_grad]

    for j, d0 in enumerate(edit_folder):
        if j <= edit_start_index:
            continue
        need_edit, ber, re_num = EDIT_DICT[args.task](t_model, data_point=d0, device=args.device)
        # data_edited = [] if args.ft_update_memory == 1 else None
        if need_edit == 1:
            LOG.info('\n')
            LOG.info(f"Before editing, model attains {ber} on {re_num} rephrases")
            res_out.ber[f].append((ber, re_num))

            edit_times += 1
            LOG.info(f'This is the {edit_times}th edit for the {f+1}th folder')
            optimizer = get_optimizer(args.ft_optim, params, args.ft_lr)

            edit_step = 0
            while True:
                # valid: stop if we have correct prediction of the example
                # if edit_step > 0 and edit_step % args.check_val_every_n_epoch == 0:
                with torch.no_grad():
                    edit_is_not_suc, _, _ = EDIT_DICT[args.task](t_model, data_point=d0, device=args.device)
                    if edit_is_not_suc == 0:
                        break
                    torch.cuda.empty_cache()

                # early_stopping if the max_edit_step is attained
                if edit_step > args.max_edit_step:
                    break

                # training
                t_model.train()
                if args.task == 'fever':
                    loss = t_model(
                        d0['src_input_ids'].to(args.device),
                        d0['src_attention_mask'].to(args.device),
                        d0['labels'].to(args.device)
                    )['loss']
                else:
                    logits = t_model(
                        input_ids=d0['src_input_ids'].to(args.device),
                        attention_mask=d0['src_attention_mask'].to(args.device),
                        decoder_input_ids=d0['trg_input_ids'][:, :-1].to(args.device),
                        decoder_attention_mask=d0['trg_attention_mask'][:, :-1].to(args.device)
                    )
                    loss, _ = label_smoothed_nll_loss(
                        logits.log_softmax(-1), d0['trg_input_ids'][:, 1:].to(args.device),
                        epsilon=t_model.hparams.eps, ignore_index=t_model.tokenizer.pad_token_id,
                    )
                if args.use_kl == 1:
                    kl_loss = get_kl_diver_loss(
                        original_model=model_to_edit, post_model=t_model, memo_loader=seq_edit_data.memory_loader,
                        device=args.device, total_loc_num=args.loc_num, his_edit_data=his_edit_data
                    )
                    loss = loss + args.alpha * kl_loss
                loss.backward()
                optimizer.step()
                edit_step += 1
                optimizer.zero_grad()
            LOG.info(f'We spend {edit_step} steps to edit the target example to the model')

            # test if we have edited the model successfully
            suc_edit, aer, re_num = EDIT_DICT[args.task](t_model, data_point=d0, device=args.device)
            LOG.info(f'After editing, {suc_edit} and {aer} edit example and its rephrases')
            res_out.edit[f].append(suc_edit)
            res_out.aer[f].append((aer, re_num))

            if (args.debug_mode == 0 and args.temp_mode == 1) or (edit_times > 0 and edit_times % 50 == 0):
                get_lrr_grr()
                # test on the history edit data
                get_err()
            his_edit_data.append(d0)
            # 及时保存文件
            res_out.save_as_file()
            # we have finished one more edits, let us save the model_state_dict as file
            torch.save(model_to_edit, os.path.join(args.log_path, 'model.ckpt'))
            with open((os.path.join(args.log_path, 'edit_schedule.txt')), 'w') as edit_schedule_file:
                edit_schedule_file.write(str(j) + '\n')
                edit_schedule_file.write(str(edit_times) + '\n')
            edit_schedule_file.close()
            LOG.info('save the historical edit data as file')
            save_obj(his_edit_data, os.path.join(args.log_path, 'his_edit_data.pkl'))

    # The sentences that used to verify the whole script is run completely
    if args.temp_mode == 1:
        get_lrr_grr()
        get_err()
    LOG.info('***************************************************')
    LOG.info('Ohhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
    LOG.info('Thanks to god that folder {} is edited completely'.format(args.fold_n))
    LOG.info('***************************************************')













