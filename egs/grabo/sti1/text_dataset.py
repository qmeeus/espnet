import os
import time
import numpy as np
import pandas as pd
import logging
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union
from filelock import FileLock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from transformers import PreTrainedTokenizer, DataProcessor
from transformers.file_utils import is_tf_available, is_torch_available
from data_utils import Split, TextExample, TextFeatures, GraboTarget


logger = logging.getLogger(__name__)


@dataclass
class TextDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    task_name: str = field(default="sti", metadata={"help": "The task name (sti)"})


class GraboProcessor(DataProcessor):

    def __init__(self, labels=None, examples=None, verbose=False):
        self.labels = GraboTarget.labels
        self.examples = [] if examples is None else examples
        self.verbose = verbose

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return STIProcessor(labels=self.labels, examples=self.examples[idx])
        return self.examples[idx]

    def extract_target(self, data):
        targets = pd.DataFrame(np.zeros((len(data), len(self.labels))), columns=self.labels)
        targets.update(pd.get_dummies(data[GraboTarget.label_columns]))
        labels = list(targets.columns)
        return targets.astype(np.int64).values.tolist()


class GraboProcessorTextInput(GraboProcessor):
        
    def get_train_examples(self, data_dir):
        """See base class."""
        return self.add_examples_from_csv(Path(data_dir, "train.csv"), Split.train)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.add_examples_from_csv(Path(data_dir, "valid.csv"), Split.dev)

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.add_examples_from_csv(Path(data_dir, "test.csv"), Split.test)

    def add_examples_from_csv(
        self,
        file_name,
        overwrite_examples=False,
        overwrite_labels=False
    ):

        data = (
            pd.read_csv(file_name, usecols=["text"] + GraboTarget.label_columns)
            .drop_duplicates(ignore_index=True)
            .reset_index(drop=True)
        )
        
        texts = data["text"]
        targets = self.extract_targets(data)
        ids = data.index

        return self.add_examples(
            texts, targets=targets, ids=ids, labels=labels, 
            overwrite_examples=overwrite_examples, 
            overwrite_labels=overwrite_labels
        )

    def add_examples(
        self, texts, 
        targets=None,
        labels=None,
        ids=None,
        overwrite_examples=False
    ):

        assert targets is None or len(texts) == len(targets)
        assert ids is None or len(texts) == len(ids)
        if ids is None:
            ids = [None] * len(texts)
        if targets is None:
            targets = [None] * len(texts)
        examples = []
        for text, target, guid in zip(texts, targets, ids):
            examples.append(TextExample(
                guid=guid, 
                input_text=text,
                label=target
            ))

        # Update examples
        if overwrite_examples:
            self.examples = examples
        else:
            self.examples.extend(examples)

        return self.examples

    def get_features(
        self,
        tokenizer,
        max_length=None,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        return_tensors=None,
    ):
        """
        Convert examples in a list of ``TextFeatures``
        Args:
            tokenizer: Instance of a tokenizer that will tokenize the examples
            max_length: Maximum example length
            pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
            pad_token: Padding token
            mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
                and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
                actual values)
        Returns:
            If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
            containing the task-specific features. If the input is a list of ``TextExamples``, will return
            a list of task-specific ``TextFeatures`` which can be fed to the model.
        """
        if max_length is None:
            max_length = tokenizer.max_len

        # label_map = {label: i for i, label in enumerate(self.labels)}

        all_input_ids = []
        for (ex_index, example) in enumerate(self.examples):
            if ex_index % 10000 == 0:
                logger.info("Tokenizing example %d", ex_index)

            input_ids = tokenizer.encode(
                example.input_text, add_special_tokens=True, max_length=min(max_length, tokenizer.max_len),
            )
            all_input_ids.append(input_ids)

        batch_length = max(len(input_ids) for input_ids in all_input_ids)

        features = []
        for (ex_index, (input_ids, example)) in enumerate(zip(all_input_ids, self.examples)):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len(self.examples)))
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = batch_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

            assert len(input_ids) == batch_length, "Error with input length {} vs {}".format(
                len(input_ids), batch_length
            )
            assert len(attention_mask) == batch_length, "Error with input length {} vs {}".format(
                len(attention_mask), batch_length
            )

            if ex_index < 5 and self.verbose:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info(f"target shape: {example.label.shape}")

            features.append(TextFeatures(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=example.label
            ))

        if return_tensors is None:
            return features
        elif return_tensors == "tf":
            if not is_tf_available():
                raise RuntimeError("return_tensors set to 'tf' but TensorFlow 2.0 can't be imported")
            import tensorflow as tf

            def gen():
                for ex in features:
                    yield ({"input_ids": ex.input_ids, "attention_mask": ex.attention_mask}, ex.label)

            dataset = tf.data.Dataset.from_generator(
                gen,
                ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
                ({"input_ids": tf.TensorShape([None]), "attention_mask": tf.TensorShape([None])}, tf.TensorShape([])),
            )
            return dataset
        elif return_tensors == "pt":
            if not is_torch_available():
                raise RuntimeError("return_tensors set to 'pt' but PyTorch can't be imported")
            import torch
            from torch.utils.data import TensorDataset

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

            dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
            return dataset
        else:
            raise ValueError("return_tensors should be one of 'tf' or 'pt'")


class TextDataset(Dataset):

    args: TextDataTrainingArguments
    output_mode: str
    features: List[TextFeatures]

    def __init__(
        self,
        args: TextDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = GraboProcessorTextInput()
        self.output_mode = "classification"
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name,
            ),
        )
        label_list = self.processor.labels
        if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
            RobertaTokenizer,
            RobertaTokenizerFast,
            XLMRobertaTokenizer,
            BartTokenizer,
            BartTokenizerFast,
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]

                self.features = self.processor.get_features(
                    tokenizer, 
                    max_length=args.max_seq_length
                )
                
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> TextFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


if __name__ == "__main__":
    from transformers import DistilBertTokenizer

    model_path = "/esat/spchdisk/scratch/qmeeus/repos/transformers/examples/language-modeling/output/distilbert/checkpoint-134000"
    filename = "/esat/spchdisk/scratch/qmeeus/repos/espnet/egs/grabo/sti1/data/grabo/target.csv"

    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    processor = GraboProcessorTextInput()
    processor.add_examples_from_csv(filename)
    train_set = processor.get_features(tokenizer, return_tensors="pt")
    import ipdb; ipdb.set_trace()
    print(processor)
