import dataclasses
import logging
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn

from sklearn.metrics import classification_report

from transformers import DistilBertModel, DistilBertTokenizer, DistilBertPreTrainedModel, DistilBertConfig
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers.data.metrics import simple_accuracy

from vector_dataset import VectorDataTrainingArguments, VectorDataset

sys.path.insert(0, "../../grabo/sti1")
from model import IntentClassifier


logger = logging.getLogger(__name__)


@dataclass
class Config:
    output_dim = 31
    input_dim = 100
    seq_classif_dropout = 0.2
    dim = 768


class Speech2IntentModel(nn.Module):

    def __init__(self, config):
        super(Speech2IntentModel, self).__init__()

        self.classifier = IntentClassifier(
            input_dim=config.dim, 
            output_dim=config.output_dim,
            dropout_rate=config.seq_classif_dropout
        )
        
    def forward(self, input_vectors=None,
        attention_mask=None,
        labels=None,
    ):

        input_lengths = attention_mask.sum(-1)
        outputs = self.classifier(input_vectors[:, 0], labels=labels)
        return outputs



def data_collator(features):
    if not isinstance(features[0], dict):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.float32)

    return batch

def sigmoid(x):
    return (1 + np.exp(-x)) ** (-1)


def build_compute_metrics_fn():
    def compute_metrics_fn(p):
        preds = (p.predictions >= .5).astype(np.int64)
        micro_acc = sum((preds[i] == p.label_ids[i]).all() for i in range(len(preds))) / len(preds)
        return {'macro_acc': simple_accuracy(preds, p.label_ids), 'micro_acc': micro_acc}
    return compute_metrics_fn


def main(config):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((VectorDataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model = Speech2IntentModel(config=config)
    print(model)

    # Get datasets
    train_dataset = (
        VectorDataset(data_args) if training_args.do_train else None
    )

    test_dataset = (
        VectorDataset(data_args, mode="test")
        if training_args.do_predict
        else None
    )

    training_args.save_steps = 0
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=build_compute_metrics_fn(),
        # prediction_loss_only=True,
        data_collator=data_collator
    )

    # Training
    if training_args.do_train:
        trainer.train(
            # model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        # trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        # if trainer.is_world_master():
            # tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        test_datasets = [test_dataset]
        # if data_args.task_name == "mnli":
        #     mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
        #     eval_datasets.append(
        #         GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        #     )

        for test_dataset in test_datasets:
            # trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            eval_result = trainer.evaluate(eval_dataset=test_dataset)

            output_eval_file = Path(
                training_args.output_dir, f"eval_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results {} *****".format(test_dataset.args.task_name))
                    for key, value in eval_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))

            eval_results.update(eval_result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]

        for test_dataset in test_datasets:
            predictions = trainer.predict(test_dataset=test_dataset)
            probabilities = sigmoid(predictions.predictions)

            np.save(
                Path(training_args.output_dir, f"test_results_{test_dataset.args.task_name}.npy"), 
                probabilities
            )

            hard_preds = (probabilities > 0.5).astype(np.int64)

            output_test_file = Path(
                training_args.output_dir, f"test_results_{test_dataset.args.task_name}.txt"
            )
            if trainer.is_world_master():
                with open(output_test_file, "w") as output_test_file:
                    logger.info(f"***** Test results {test_dataset.args.task_name} *****")
                    output_test_file.write("index\tprediction\n")
                    for index, vector in enumerate(hard_preds):
                        item = " ".join([
                            test_dataset.get_labels()[i].split('_', maxsplit=1)[-1]
                            for i, item in enumerate(vector) if item
                        ])

                        output_test_file.write("%d\t%s\n" % (index, item))

            score_file = Path(training_args.output_dir, f"prediction_scores_{test_dataset.args.task_name}.txt")
            with open(score_file, 'w') as score_file:
                score_file.writelines([
                    f"{metric}\t{value}\n" for metric, value in build_compute_metrics_fn()(predictions).items()
                ])

            report_file = Path(training_args.output_dir, f"classification_report_{test_dataset.args.task_name}.txt")
            with open(report_file, 'w') as report_file:
                report_file.write(classification_report(
                    predictions.label_ids, 
                    hard_preds, 
                    target_names=test_dataset.get_labels(),
                    digits=5,
                    zero_division=0
                ))

    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main(Config())


if __name__ == "__main__":
    main(Config())