import os
import sys
import torch
import pickle
import logging
import argparse
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
sys.path.append(".")
sys.path.append("..")
LOG = logging.getLogger(__name__)
LOG.setLevel(level=logging.DEBUG)
SELECT_ERROR_INDEX = []
SEQUENCE_LENGTH_ACCUMULATE = 0
EXAMPLE_TO_NEURON = {}


def get_callbacks(args_):
    lr_callback = LearningRateMonitor(logging_interval="step")
    early_stopping_callback = EarlyStopping(
        monitor="stop_editing",
        patience=100000,
        # patience=args_.early_patience,
        stopping_threshold=0.99,
        mode='max',
    )
    model_checkpoint_callback = ModelCheckpoint(
        monitor=args_.ckpt_monitor,
        mode=args_.ckpt_metric_mode,
        dirpath=args_.ckpt_path,
        save_top_k=1,
        filename="model",  # auto-filled,
        # save_weights_only=True
    )
    return [lr_callback, early_stopping_callback, model_checkpoint_callback]


def editing(e_t, a_t):
    global SELECT_ERROR_INDEX, SEQUENCE_LENGTH_ACCUMULATE, EXAMPLE_TO_NEURON

    init_weights = None
    error_count, select_index = 0, []
    if args.task == 'fever':
        if args.use_init_weight == 1:
            editor.editor.insert_hidden_detector()
        need_edit, ber, rephrase_num = edit_or_not_binary(editor.editor, data_point=d0, device=args.device, args=args)
        if args.use_init_weight == 1:
            init_weights = editor.editor.get_hidden()
            editor.editor.clear_detectors()
        if need_edit == 1:
            error_count, select_index = 1, [0]
    else:
        need_edit, ber, rephrase_num = edit_or_not_seq2seq(editor.editor.model, data_point=d0, device=args.device)
        if need_edit == 1:
            if args.use_init_weight == 1:
                editor.editor.insert_hidden_detector()
                editor.editor.unlock_hidden_detectors()
            error_count, select_index = count_error_nums(editor.editor.model, d0, device=args.device)
            if error_count > args.max_add_neuron_num:
                LOG.info(f"Too much neuron added: {error_count}, we just utilize {args.max_add_neuron_num} of them")
                error_count, select_index = args.max_add_neuron_num, select_index[:args.max_add_neuron_num]
            if args.use_init_weight == 1:
                init_weights = editor.editor.get_hidden(select_index)
                editor.editor.clear_detectors()
    a_t += error_count

    if need_edit == 1 and error_count > 0:
        SELECT_ERROR_INDEX += [si + SEQUENCE_LENGTH_ACCUMULATE - 1 for si in select_index]
        SEQUENCE_LENGTH_ACCUMULATE += d0['trg_input_ids'][[0], :-1].size(1) - 1 if not args.task == "fever" else 1
        EXAMPLE_TO_NEURON[e_t] = select_index

        LOG.info("\n")
        LOG.info(f"For this example, we add {error_count} neuron(s)")
        LOG.info(f"Before editing, model attains {ber} on {rephrase_num} rephrases")
        res_out.ber[f].append((ber, rephrase_num))
        e_t += 1
        LOG.info(f"This is the {e_t}th edit for the {f+1}th folder")

        # build the trainer for editor
        callbacks = get_callbacks(args)
        edit_trainer = Trainer(
            callbacks=callbacks, gpus=args.gpus, logger=TensorBoardLogger(log_dir, name=None),
            check_val_every_n_epoch=args.check_val_every_n_epoch, log_every_n_steps=args.check_val_every_n_epoch,
            max_epochs=args.max_edit_step, num_sanity_val_steps=0, enable_progress_bar=args.debug_mode == 1,
            gradient_clip_val=5.0, weights_summary=None
        )

        res_out.add_neuron_num[f].append(error_count)
        editor.editor.set_editors(init_weights=init_weights, error_count=error_count, select_index=select_index)
        if args.memory_loss.startswith('kl'):
            LOG.info("We are feeding kl input")
            editor.editor.feed_kl_input(
                memo_loader=seq_edit_data.memory_loader,
                total_loc_num=int(args.memory_loss[3:]),
                his_edit_data=his_edit_data
            )
        edit_trainer.fit(editor, train_dataloaders=dl, val_dataloaders=dl)
        if not editor.has_stepped:
            editor.editor.step()
        if args.update_memory == 1 and (args.memory_loss != 'non_use' and not args.memory_loss.startswith('kl')):
            editor.memorize(train_memory_data=dl, device=args.device, update=True)

    return e_t, a_t


def get_train_r_test_r():
    test_r = TEST_DICT[args.task](editor.editor.model, seq_edit_data.dev_loader, args.device)
    train_r = TEST_DICT[args.task](editor.editor.model, seq_edit_data.train_sub_loader, args.device)
    res_out.test[f].append(test_r[0])
    LOG.info(f"Test Retain Rate: {test_r[0]}")
    res_out.train[f].append(train_r[0])
    LOG.info(f"Train Retain Rate {train_r[0]}")


def get_er():
    if len(his_edit_data) > 0:
        LOG.info("Testing on the history edit dataset")
        er, _ = TEST_DICT[args.task](editor.editor.model, his_edit_data, args.device, 'original')
        # his_re, his_re_num = TEST_DICT[args.task](editor.editor.model, his_edit_data, args.device, 'rephrases')
        res_out.his[f].append(er)
        # res_out.his_re[f].append((his_re, his_re_num))
        LOG.info(f"Model attains {er} on past edit examples")


if __name__ == "__main__":
    from src.models.seq2seq_modules import BartSeq2SeqEditor
    from src.models.class_modules import BertBinaryEditor
    from src.dataset.sme_dataset import SeqEditResOutput
    from src.dataset.zsre_dataloader import Seq2SeqData
    from src.dataset.fever_dataloader import FeverData
    from src.utils import get_handler, split_data_n_sets, save_obj, load_obj, echo, main_args, \
        my_test_seq2seq, my_test_binary, edit_or_not_seq2seq, edit_or_not_binary, count_error_nums

    TEST_DICT = {'zsre': my_test_seq2seq, 'fever': my_test_binary}

    parser = argparse.ArgumentParser()
    parser = main_args(parser)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gpus", type=list, default=[0])
    parser.add_argument("--fold_n", type=int, default=0)
    """The BartSeq2SeqEditor and BertBinaryEditor share the same arguments"""
    parser = BartSeq2SeqEditor.add_model_specific_args(parent_parser=parser)
    args, _ = parser.parse_known_args()
    args.gpus = [args.device]
    args.device = torch.device('cuda', args.device)
    tb_logger = TensorBoardLogger(args.log_path, name=None, version="fold_{}".format(args.fold_n))
    log_dir = tb_logger.log_dir
    args.ckpt_path = log_dir
    f_h, s_h = get_handler(tb_logger.log_dir, log_name=args.log_name)
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)
    LOG.info("All hyper-parameters are as follws")
    echo(LOG, vars(args))

    seed_everything(args.seed)

    # Loading edit data
    LOG.info("Loading data")
    with open(os.path.join(args.data_path, f"seq_edit_data_{args.task}_{args.edit_folder_num}.pkl"), 'rb') as file:
        seq_edit_data = pickle.loads(file.read())
    if args.train_sub_size != len(seq_edit_data.train_sub) or args.memory_size != len(seq_edit_data.memory_set):
        seq_edit_data.re_split_train_sub_and_memory(args.train_sub_size, args.memory_size)
    if args.num_workers != seq_edit_data.num_workers or args.batch_size != seq_edit_data.batch_size:
        seq_edit_data.re_set_loaders(args.num_workers, args.batch_size)
    if args.example_repeat != seq_edit_data.example_repeat:
        seq_edit_data.reset_example_repeat(er=args.example_repeat)
    seq_edit_data.shuffle_memory_loader()
    echo(LOG, {f'The size of {k} is': len(getattr(seq_edit_data, k)) for k in ('train_sub', 'memory_set', 'edit_test_data', 'dev_data', 'val_data')})

    # let us check if this is a continuation of previous Experiments
    is_continue = False
    if os.path.exists(os.path.join(args.ckpt_path, 'last_model.ckpt')) \
            and os.path.exists(os.path.join(args.ckpt_path, 'edit_schedule.txt')) \
            and os.path.exists(os.path.join(args.ckpt_path, 'res.pkl')):
        is_continue = True
        LOG.info("Loading result file")
        with open(os.path.join(args.ckpt_path, 'res.pkl'), 'rb') as result_file:
            res_out = pickle.loads(result_file.read())
        result_file.close()

        LOG.info("Loading edit start index")
        with open(os.path.join(args.ckpt_path, 'edit_schedule.txt')) as edit_schedule_file:
            edit_start_index = int(edit_schedule_file.readline())
            edit_times = int(edit_schedule_file.readline())
            ADD_NEURON_COUNT = int(edit_schedule_file.readline())
        edit_schedule_file.close()

        LOG.info("Loading model")
        if args.task == 'fever':
            editor = BertBinaryEditor.load_from_checkpoint(os.path.join(args.ckpt_path, 'last_model.ckpt'), add_neuron_num=ADD_NEURON_COUNT)
        else:
            editor = BartSeq2SeqEditor.load_from_checkpoint(os.path.join(args.ckpt_path, 'last_model.ckpt'), add_neuron_num=ADD_NEURON_COUNT)
        his_edit_data = load_obj(os.path.join(args.ckpt_path, 'his_edit_data.pkl'))
    else:
        edit_start_index = -1
        LOG.info("Creating editor and result class")
        if args.task == 'fever':
            editor = BertBinaryEditor(**vars(args))
        else:
            editor = BartSeq2SeqEditor(**vars(args))
        res_out = SeqEditResOutput(edit_folder_num=args.edit_folder_num, save_dir=log_dir)
        his_edit_data, edit_times, ADD_NEURON_COUNT = [], 0, 0
    editor.fed_val_loader(seq_edit_data.val_loader)
    f, edit_folder = args.fold_n, seq_edit_data.edit_folder[args.fold_n]

    if args.memory_loss != 'non_use' and not args.memory_loss.startswith('kl'):
        LOG.info(f"We utilize the {args.memory_loss} memory loss and construct the memory on learnt training data")
        editor.memorize(
            seq_edit_data.memory_loader, device=args.device,
            update=False, val_memory_data=seq_edit_data.val_loader
        )
        train_memo_size = [len(value) for value in editor.editor.train_memories.values()]
        val_memo_size = [len(value) for value in editor.editor.val_memories.values()]
        LOG.info(f"The memories have been constructed in {args.memory_loss} method, the size of training and validation memory are {train_memo_size} and {val_memo_size}")

        # if this is a continuation of previous experiments, we need reload the memory
        if his_edit_data and args.update_memory == 1:
            for dl in his_edit_data:
                editor.memorize(train_memory_data=[dl], device=args.device, update=True)
            LOG.info("The saved history edit data is added into memory")
    torch.cuda.empty_cache()
    LOG.info("The model that we edited is {}".format(args.model_path))

    if args.debug_mode == 0 and not is_continue:
        res_out.init_metric['test'], _ = TEST_DICT[args.task](editor.editor.model, data_loader=seq_edit_data.dev_loader, device=args.device)
        res_out.init_metric['train'], _ = TEST_DICT[args.task](model=editor.editor.model, data_loader=seq_edit_data.train_sub_loader, device=args.device)
        edit_acc, _ = TEST_DICT[args.task](editor.editor.model, seq_edit_data.edit_test_loader, args.device)
        echo(LOG, {f"The acc on {k} is": res_out.init_metric[k] if k != 'edit' else edit_acc for k in ('test', 'train', 'edit')})

    LOG.info("\n\n")
    edit_sets = split_data_n_sets(edit_folder, len(edit_folder))

    # remove the model for checkpoint
    if os.path.exists(os.path.join(args.ckpt_path, "model.ckpt")):
        os.remove(os.path.join(args.ckpt_path, "model.ckpt"))
    s, ds = 0, None
    for s, ds in enumerate(edit_sets):
        if s <= edit_start_index:
            continue
        dl = DataLoader(dataset=ds, batch_size=1, collate_fn=seq_edit_data.edit_data.collate_fn)
        d0 = [j for j in dl][0]

        edit_times_tmp, ADD_NEURON_COUNT = editing(edit_times, ADD_NEURON_COUNT)

        if edit_times_tmp == edit_times + 1:
            edit_times = edit_times_tmp
            if args.task == 'fever':
                edit_is_not_suc, aer, re_num = edit_or_not_binary(editor.editor, data_point=d0, device=args.device, args=args)
            else:
                edit_is_not_suc, aer, re_num = edit_or_not_seq2seq(editor.editor.model, data_point=d0, device=args.device)
            LOG.info(f"After editing, {1 - edit_is_not_suc} and {aer} edit example and its rephrases")
            if edit_is_not_suc == 1:
                LOG.info("T-Patch has failed an edit.")
            res_out.edit[f].append((1 - edit_is_not_suc))
            res_out.aer[f].append((aer, re_num))

            if (args.debug_mode == 0 and args.temp_mode == 1) or (edit_times > 0 and edit_times % 50 == 0):
                get_train_r_test_r()
                get_er()
            his_edit_data.append(d0)
            res_out.save_as_file()

            if os.path.exists(os.path.join(args.ckpt_path, "model.ckpt")):
                LOG.info("We rename the model.ckpt for the new model checkpoint")
                if os.path.exists(os.path.join(args.ckpt_path, "last_model.ckpt")):
                    os.remove(os.path.join(args.ckpt_path, "last_model.ckpt"))
                os.rename(os.path.join(args.ckpt_path, "model.ckpt"), os.path.join(args.ckpt_path, "last_model.ckpt"))
            with open((os.path.join(args.ckpt_path, 'edit_schedule.txt')), 'w') as edit_schedule_file:
                edit_schedule_file.write(str(s) + '\n')
                edit_schedule_file.write(str(edit_times) + '\n')
                edit_schedule_file.write(str(ADD_NEURON_COUNT))
            edit_schedule_file.close()
            LOG.info("save the historical edit data as file")
            save_obj(his_edit_data, os.path.join(args.ckpt_path, 'his_edit_data.pkl'))

            if (args.debug_mode == 1 and edit_times >= 3) or (args.get_heat_map == 1 and ADD_NEURON_COUNT >= 100):
                break
    # The sentences that used to verify the whole script is run completely
    if args.temp_mode == 0 and args.debug_mode == 0:
        LOG.info("temp_mode is 0, we get the final metric")
        get_train_r_test_r()
        get_er()

    if s == len(edit_sets):
        LOG.info("Ohhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
        LOG.info(f"Folder {args.fold_n} is edited completely")
        LOG.info("***************************************************")
    else:
        LOG.info(f"Folder {args.fold_n} is edited completely")
    # Since the edit is Done we can delete the model checkpoints
    if args.get_heat_map == 0:
        os.remove(os.path.join(args.ckpt_path, "last_model.ckpt"))

    if args.get_heat_map == 1:
        # In the end, we want to get the act values for each edit data of the final model
        # prepare data
        if args.task == 'fever':
            dev_data = seq_edit_data.dev_data
        else:
            dev_data = Seq2SeqData(tokenizer=seq_edit_data.tokenizer, data_path=os.path.join(args.data_path, '{}-dev-kilt.jsonl'.format(args.task)), validation=False)
        dev_data = DataLoader(dev_data, batch_size=128, collate_fn=dev_data.collate_fn, num_workers=0, shuffle=True)
        set_to_loader = {
            'his_edit': his_edit_data,
            'test': dev_data,
        }
        for s, dl in set_to_loader.items():
            LOG.info(f"Recording memory of {s}")
            editor.editor.detectors = []
            editor.editor.model.eval()
            editor.editor.model.to(args.device)
            if args.task == 'zsqa':
                editor.editor.get_detectors(
                    detected_modules={'model.model.decoder.layers.5': 'fc1'},
                    memory_loc='bart_seq', hidden_loc='bart_seq', mode='output'
                )
            else:
                editor.editor.get_detectors(detected_modules={'model.model.encoder.layer.11': 'intermediate'}, mode='output')
            editor.editor.set_detectors()
            for d in editor.editor.detectors:
                d['detector'].turn_on_memory()
            for i, batch in enumerate(dl):
                LOG.info("Sending batches...")
                if args.task == 'zsqa':
                    input_ids = batch["src_input_ids"][[0]].to(args.device) if s == 'his_edit' else batch["src_input_ids"].to(args.device)
                    attention_mask = batch["src_attention_mask"][[0]].to(args.device) if s == 'his_edit' else batch["src_attention_mask"].to(args.device)
                    decoder_input_ids = batch["trg_input_ids"][[0]].to(args.device) if s == 'his_edit' else batch["trg_input_ids"].to(args.device)
                    decoder_attention_mask = batch["trg_attention_mask"][[0]].to(args.device) if s == 'his_edit' else batch["trg_attention_mask"].to(args.device)
                    for d in editor.editor.detectors:
                        d['detector'].feed_memory_mask(decoder_attention_mask[:, :-1])
                    editor.editor.model(
                        input_ids, attention_mask,
                        decoder_input_ids[:, :-1], decoder_attention_mask[:, :-1]
                    )
                else:
                    editor.editor.model(
                        batch["src_input_ids"][[0]].to(args.device) if s == 'his_edit' else batch["src_input_ids"].to(args.device),
                        batch["src_attention_mask"][[0]].to(args.device) if s == 'his_edit' else batch["src_attention_mask"].to(args.device),
                        batch["labels"][[0]].to(args.device) if s == 'his_edit' else batch["labels"].to(args.device)
                    )
            LOG.info(f"Recording memory of {s}")
            for d in editor.editor.detectors:
                # name = d['module'] + '.' + d['child']
                acts = d['detector'].get_memory()
                LOG.info(f"Got {len(acts)} memories from {d['module'] + '.' + d['child']}")
                if s == 'his_edit' and args.task != 'fever':
                    filter_act = []
                    for sri in SELECT_ERROR_INDEX:
                        filter_act.append(acts[sri])
                    acts = filter_act
                editor.editor.model_named_modules[d['module']]._modules[d['child']] = d['original_module']
            editor.editor.detectors = []
            res_acts = [a.cpu() for a in acts]
            acts = res_acts
            LOG.info(f"Saving {len(acts)} memories for {s}")
            LOG.info(acts)
            save_obj(acts, os.path.join(args.ckpt_path, f'{args.task_id}_{s}_acts.pkl'))
        save_obj(EXAMPLE_TO_NEURON, os.path.join(args.ckpt_path, f'{args.task_id}_example_to_neuron.pkl'))




    







