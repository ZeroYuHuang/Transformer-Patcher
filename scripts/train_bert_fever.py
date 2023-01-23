import os
import sys
sys.path.append(".")

if __name__ == "__main__":
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.utilities.seed import seed_everything
    from src.models.class_modules import BertBinary
    from src.dataset.fever_dataloader import FeverData
    from torch.utils.data import DataLoader

    parser = ArgumentParser()

    parser.add_argument("--log_path", type=str, default="./log/models/bert_binary")
    parser.add_argument("--save_top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=list, default=[0])
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--cache_dir", type=str, default='./hugging_cache')

    parser = BertBinary.add_model_specific_args(parser)

    args, _ = parser.parse_known_args()
    seed_everything(seed=args.seed)
    logger = TensorBoardLogger(args.log_path, name=None)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_acc", mode="max",
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        save_top_k=args.save_top_k, save_weights_only=True,
        filename="bert-binary-{epoch:02d}-{valid_acc:.4f}"
    )
    lr_callback = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks, max_epochs=args.max_epochs)
    model = BertBinary(**vars(args))
    trainer.fit(model)
    trainer.test()
    edit_dataloader = DataLoader(
        dataset=FeverData(
            tokenizer=model.tokenizer, data_path='./src/dataset/fever_data/fever-edit.jsonl',
            max_length=model.hparams.max_length
        ),
        batch_size=model.hparams.batch_size,
        collate_fn=model.train_dataset.collate_fn,
        num_workers=model.hparams.num_workers
    )

    trainer.test(dataloaders=edit_dataloader)


