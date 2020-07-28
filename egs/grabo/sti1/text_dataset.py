import os
import time
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from typing import List, Optional, Union
from filelock import FileLock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

# from transformers import InputFeatures, 
from transformers import PreTrainedTokenizer, DataProcessor
from transformers.file_utils import is_tf_available, is_torch_available

logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        input_text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        action: string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        instruction: list of int: a vector of 1s and 0s.
    """

    guid: str
    input_text: str
    action: str
    instruction: List[int]

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
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    actions: Optional[int] = None
    instructions: Optional[List[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


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


class STIProcessor(DataProcessor):

    ACTIONS = [
        'approach', 'grab', 'lift', 'move_abs', 'move_rel', 
        'pointer', 'turn_abs', 'turn_rel'
    ]
    
    TEXT_COLUMN = "text"
    LABELS = [
        "action", "throttle", "distance", "direction", "angle", 
        "pos_x", "pos_y", "position", "state", "grabber"
    ]

    def __init__(self, labels=None, examples=None, verbose=False):
        self.labels = [] if labels is None else labels
        self.examples = [] if examples is None else examples
        self.verbose = verbose

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return STIProcessor(labels=self.labels, examples=self.examples[idx])
        return self.examples[idx]
        
    def get_train_examples(self, data_dir):
        """See base class."""
        return self.add_examples_from_csv(os.path.join(data_dir, "train.csv"), Split.train)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.add_examples_from_csv(os.path.join(data_dir, "valid.csv"), Split.dev)

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.add_examples_from_csv(os.path.join(data_dir, "test.csv"), Split.test)

    def add_examples_from_csv(
        self,
        file_name,
        overwrite_labels=False,
        overwrite_examples=False,
    ):

        data = (
            pd.read_csv(file_name, usecols=[self.TEXT_COLUMN] + self.LABELS)
            .drop_duplicates(ignore_index=True)
            .reset_index(drop=True)
        )
        
        texts = data[self.TEXT_COLUMN]
        labels = (data[self.LABELS[0]].to_list(),)
        labels += (pd.get_dummies(data[self.LABELS[1:]]).values.tolist(),)
        labels = list(zip(*labels))
        ids = data.index

        return self.add_examples(
            texts, labels, ids, overwrite_labels=overwrite_labels, overwrite_examples=overwrite_examples
        )

    def add_examples(
        self, texts, 
        labels=None, 
        ids=None, 
        overwrite_labels=False, 
        overwrite_examples=False
    ):

        assert labels is None or len(texts) == len(labels)
        assert ids is None or len(texts) == len(ids)
        if ids is None:
            ids = [None] * len(texts)
        if labels is None:
            labels = [None] * len(texts)
        examples = []
        added_labels = set()
        for (text, label, guid) in zip(texts, labels, ids):
            action, instruction = label
            added_labels.add(action)
            examples.append(InputExample(
                guid=guid, 
                input_text=text, 
                action=action, 
                instruction=instruction
            ))

        # Update examples
        if overwrite_examples:
            self.examples = examples
        else:
            self.examples.extend(examples)

        # Update labels
        if overwrite_labels:
            self.labels = list(added_labels)
        else:
            self.labels = list(set(self.labels).union(added_labels))

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
        if max_length is None:
            max_length = tokenizer.max_len

        label_map = {label: i for i, label in enumerate(self.labels)}

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

            action_id = label_map[example.action]

            if ex_index < 5 and self.verbose:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("label: %s (id = %d)" % (example.action, action_id))

            features.append(InputFeatures(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                actions=action_id,
                instructions=example.instruction
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
            all_labels = torch.tensor([f.label[0] for f in features], dtype=torch.long)
            all_instructions = torch.tensor([f.label[1] for f in features], dtype=torch.long)

            dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels, all_instructions)
            return dataset
        else:
            raise ValueError("return_tensors should be one of 'tf' or 'pt'")


class TextDataset(Dataset):

    args: TextDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: TextDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
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

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

    def collate(self, batch):
        import ipdb; ipdb.set_trace()
        return self[batch]


if __name__ == "__main__":
    from transformers import DistilBertTokenizer

    model_path = "/esat/spchdisk/scratch/qmeeus/repos/transformers/examples/language-modeling/output/distilbert/checkpoint-134000"
    filename = "/esat/spchdisk/scratch/qmeeus/repos/espnet/egs/grabo/sti1/data/grabo/target.csv"

    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    processor = STIProcessor()
    processor.add_examples_from_csv(filename)
    train_set = processor.get_features(tokenizer, return_tensors="pt")
    import ipdb; ipdb.set_trace()
    print(processor)


"""
class Dataset:

    TRAIN_SIZE = .7
    VALID_SIZE = .1
    TEST_SIZE = .2

    LABELS = {
        "throttle": ["fast", "slow"],
        "distance": ["alot", "litte", "normal"],
        "direction": ["backward", "forward"],
        "angle": ["north", "south", "west", "east"],
        "pos_x": ["center_x", "left", "right"],
        "pos_y": ["center_y", "up", "down"],
        "state": ["on", "off"],
        "grabber": ["open", "close"]
    }

    ACTIONS = [
        "move_rel",
        "turn_rel",
        "turn_abs",
        "move_abs",
        "approach",
        "lift",
        "pointer",
        "grab"
    ]

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument_group("Data")
        parser.add_argument("data", type=Path)
        parser.add_argument('--vocab', type=Path, required=True)
        parser.add_argument("--train-size", type=float, default=cls.TRAIN_SIZE)
        parser.add_argument("--valid-size", type=float, default=cls.VALID_SIZE)
        parser.add_argument("--test-size", type=float, default=cls.TEST_SIZE)
        return parser

    def __init__(self, data, vocab, train_size=TRAIN_SIZE, valid_size=VALID_SIZE, test_size=TEST_SIZE):
        data = pd.read_csv(data, usecols=range(2,14)).drop_duplicates(ignore_index=True)

        with open(vocab) as f:
            self.vocab = list(map(str.strip, f.readlines()))

        original_size = len(data)
        data = data.loc[
            data["text"].map(lambda s: all(token in self.vocab for token in s.split()))
        ].copy()

        self.n_samples = len(data)
        print(f"Dropped {self.n_samples - original_size} sentences with unknown words")
        self.x_dim = len(self.vocab)
        self.y1_dim = len(self.ACTIONS)
        self.y2_dim = sum(map(len, LABELS.values())) + len(LABELS)

        self.X = data["text"].map(self.encode_sentence).values
        self.y1 = data["action"].map(self.ACTIONS.index).values
        self.y2 = np.stack(
            data.iloc[:, 3:-2].apply(self.create_instruction_vector, axis=1).to_list(),
            axis=0
        )

        train_idx, self.test_idx = train_test_split(
            np.arange(self.n_samples), test_size=test_size
        )

        self.train_idx, self.valid_idx = train_test_split(
            train_idx, test_size=valid_size/train_size
        )

    def encode_sentence(self, sentence):
        tokens = sentence.split()
        return np.array([self.vocab.index(token) for token in tokens], dtype=np.int64)

    def create_instruction_vector(self, row):
        instructions = np.zeros((self.y2_dim,), np.int64)
        cur = 0
        for name, values in LABELS.items():
            instr = row[name]
            index = cur + (values.index(instr) + 1 if instr else 0)
            instruction[index] = 1
            cur += len(values) + 1
        return instructions
"""