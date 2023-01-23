import os
import sys
from argparse import ArgumentParser
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
sys.path.append('.')

if __name__ == "__main__":

    from src.models.seq2seq_modules import BartSeq2Seq

    parser = ArgumentParser()
    parser.add_argument("--log_path", type=str, default="./log/models/bart_seq2seq")
    parser.add_argument("--save_top_k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--gpus", type=list, default=[0])
    parser.add_argument("--cache_dir", type=str, default='./hugging_cache')

    parser = BartSeq2Seq.add_model_specific_args(parser)

    args, _ = parser.parse_known_args()
    args.gpus = [args.device]
    seed_everything(args.seed)
    logger = TensorBoardLogger(args.log_path, name=None)

    callbacks = [
        ModelCheckpoint(
            monitor="valid_acc", mode="max",
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            save_weights_only=True, save_top_k=args.save_top_k,
            filename="bart-seq2seq-{epoch:02d}-{valid_acc:.4f}"
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks,  max_epochs=15)
    model = BartSeq2Seq(**vars(args))
    trainer.fit(model)

