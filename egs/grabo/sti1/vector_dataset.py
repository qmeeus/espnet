import os
import time
import numpy as np
import pandas as pd
import logging
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from typing import List, Optional, Union
from filelock import FileLock
import h5py
from operator import attrgetter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from transformers import PreTrainedTokenizer, DataProcessor
from transformers.file_utils import is_tf_available, is_torch_available

logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "valid"
    test = "test"


@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        input_text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        target: list of int: a vector of 1s and 0s.
    """

    guid: str
    input_vector: List[float]
    label: List[int]

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_vectors: List[float]
    attention_mask: Optional[List[int]] = None
    labels: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass
class VectorDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .npy files (or other data files) for the task."}
    )

    feature_file: str = field(
        metadata={"help": "The path to the feature file"}
    )

    target_file: str = field(
        metadata={"help": "The path to the target file"}
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


@dataclass(frozen=True)
class Grabo:

    index_column = "uttid"
    label_columns = [
        "action", "throttle", "distance", "direction", "angle",
        "pos_x", "pos_y", "position", "state", "grabber"
    ]

    labels = [
        'action_approach', 'action_grab', 'action_lift', 'action_move_abs',
        'action_move_rel', 'action_pointer', 'action_turn_abs', 'action_turn_rel',
        'throttle_fast', 'throttle_slow', 'distance_alot', 'distance_little',
        'distance_normal', 'direction_backward', 'direction_forward',
        'angle_east', 'angle_north', 'angle_south', 'angle_west',
        'pos_x_centerx', 'pos_x_left', 'pos_x_right',
        'pos_y_centery', 'pos_y_down', 'pos_y_up',
        'position_down', 'position_up', 'state_off', 'state_on',
        'grabber_close', 'grabber_open'
    ]


class STIProcessor(DataProcessor):

    def __init__(self, labels=None, examples=None, verbose=False):
        self.labels = Grabo.labels
        self.examples = [] if examples is None else examples
        self.verbose = verbose

        for split in ('train', 'dev', 'test'):
            setattr(self, f'get_{split}_examples', lambda args: self.get_examples(args, split=split))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return STIProcessor(labels=self.labels, examples=self.examples[idx])
        return self.examples[idx]

    def get_examples(self, args, split='train'):
        ids_file = Path(args.data_dir, f"{split}.ids")
        return self.add_examples_from_csv(args.target_file, args.feature_file, ids_file)

    def add_examples_from_csv(
        self, target_file,
        features_file,
        uttids,
        overwrite_examples=False,
        overwrite_labels=False
    ):

        data = (
            pd.read_csv(target_file, usecols=["uttid"] + Grabo.label_columns)
            .set_index('uttid').loc[np.loadtxt(uttids, dtype='str')].copy()
        )

        targets = pd.DataFrame(np.zeros((len(data), len(self.labels))), index=data.index, columns=self.labels)
        targets.update(pd.get_dummies(data[Grabo.label_columns]))
        labels = list(targets.columns)
        targets = targets.astype(np.int64).values.tolist()
        ids = data.index

        with h5py.File(features_file, 'r') as h5f:
            features = [h5f[key][()] for key in ids]

        return self.add_examples(
            features, targets=targets, ids=ids, labels=labels,
            overwrite_examples=overwrite_examples,
            overwrite_labels=overwrite_labels
        )

    def add_examples(
        self, features,
        targets=None,
        labels=None,
        ids=None,
        overwrite_labels=False,
        overwrite_examples=False
    ):

        assert targets is None or len(features) == len(targets)
        assert ids is None or len(features) == len(ids)
        if ids is None:
            ids = [None] * len(features)
        if targets is None:
            targets = [None] * len(features)
        examples = []
        for feats, target, guid in zip(features, targets, ids):
            examples.append(InputExample(
                guid=guid,
                input_vector=feats,
                label=target
            ))

        # Update labels
        if labels and overwrite_labels:
            self.labels = labels
        else:
            self.labels = list(set(self.labels).union(labels))

        # Update examples
        if overwrite_examples:
            self.examples = examples
        else:
            self.examples.extend(examples)

        return self.examples

    def get_features(
        self,
        pad_token=0,
        mask_padding_with_zero=True,
        return_tensors=None,
    ):
        """
        Convert examples in a list of ``InputFeatures``
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
            containing the task-specific features. If the input is a list of ``InputExamples``, will return
            a list of task-specific ``InputFeatures`` which can be fed to the model.
        """

        batch_length = max(map(len, map(attrgetter('input_vector'), self.examples)))

        features = []
        for ex_index, example in enumerate(self.examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len(self.examples)))

            input_vector = example.input_vector
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(example.input_vector)

            # Zero-pad up to the sequence length.
            padding_length = batch_length - len(input_vector)
            padding = ((0, padding_length), (0, 0))
            input_vector = np.pad(input_vector, padding, mode='constant', constant_values=pad_token)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

            assert len(input_vector) == batch_length, "Error with input length {} vs {}".format(
                len(input_vector), batch_length
            )
            assert len(attention_mask) == batch_length, "Error with input length {} vs {}".format(
                len(attention_mask), batch_length
            )

            if ex_index < 5 and self.verbose:
                logger.info("*** Example ***")
                logger.info(f"guid: {example.guid}")
                logger.info(f"features shape: {input_vector.shape}")
                logger.info(f"attention_mask: {attention_mask}")
                logger.info(f"target shape: {example.label.shape}")

            features.append(InputFeatures(
                input_vectors=input_vector,
                attention_mask=attention_mask,
                labels=example.label
            ))

        return features


class VectorDataset(Dataset):

    args: VectorDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: VectorDataTrainingArguments,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = STIProcessor()
        self.output_mode = "classification"
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}".format(
                mode.value, str(args.max_seq_length), args.task_name,
            ),
        )
        label_list = self.processor.labels
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
                    examples = self.processor.get_dev_examples(args)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args)
                else:
                    examples = self.processor.get_train_examples(args)
                if limit_length is not None:
                    examples = examples[:limit_length]

                self.features = self.processor.get_features()

                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    f"Saving features into cached file {cached_features_file} [took {time.time() - start:.3f} s]"
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

