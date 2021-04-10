import os
from typing import List
from typing import Tuple
import logging
from collections import defaultdict
from collections import Counter
import json
import random
import torch
import numpy as np

from GroundedScan.dataset import GroundedScan

logger = logging.getLogger(__name__)


class Vocabulary(object):
    """
    Object that maps words in string form to indices to be processed by numerical models.
    """

    def __init__(self, sos_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>"):
        """
        NB: <PAD> token is by construction idx 0.
        """
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self._idx_to_word = [pad_token, sos_token, eos_token]
        self._word_to_idx = defaultdict(lambda: self._idx_to_word.index(self.pad_token))
        self._word_to_idx[sos_token] = 1
        self._word_to_idx[eos_token] = 2
        self._word_frequencies = Counter()

    def word_to_idx(self, word: str) -> int:
        return self._word_to_idx[word]

    def idx_to_word(self, idx: int) -> str:
        return self._idx_to_word[idx]

    def add_sentence(self, sentence: List[str]):
        for word in sentence:
            if word not in self._word_to_idx:
                self._word_to_idx[word] = self.size
                self._idx_to_word.append(word)
            self._word_frequencies[word] += 1

    def most_common(self, n=10):
        return self._word_frequencies.most_common(n=n)

    @property
    def pad_idx(self):
        return self.word_to_idx(self.pad_token)

    @property
    def sos_idx(self):
        return self.word_to_idx(self.sos_token)

    @property
    def eos_idx(self):
        return self.word_to_idx(self.eos_token)

    @property
    def size(self):
        return len(self._idx_to_word)

    @classmethod
    def load(cls, path: str):
        assert os.path.exists(path), "Trying to load a vocabulary from a non-existing file {}".format(path)
        with open(path, 'r') as infile:
            all_data = json.load(infile)
            sos_token = all_data["sos_token"]
            eos_token = all_data["eos_token"]
            pad_token = all_data["pad_token"]
            vocab = cls(sos_token=sos_token, eos_token=eos_token, pad_token=pad_token)
            vocab._idx_to_word = all_data["idx_to_word"]
            vocab._word_to_idx = defaultdict(int)
            for word, idx in all_data["word_to_idx"].items():
                vocab._word_to_idx[word] = idx
            vocab._word_frequencies = Counter(all_data["word_frequencies"])
        return vocab

    def to_dict(self) -> dict:
        return {
            "sos_token": self.sos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "idx_to_word": self._idx_to_word,
            "word_to_idx": self._word_to_idx,
            "word_frequencies": self._word_frequencies
        }

    def save(self, path: str) -> str:
        with open(path, 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=4)
        return path


class GroundedScanDataset(object):
    """
    Loads a GroundedScan instance from a specified location.
    """

    def __init__(self, path_to_data: str, save_directory: str, k: int, split="train",
                 target_vocabulary_file="", generate_vocabulary=False, max_seq_length=np.inf):
        assert os.path.exists(path_to_data), \
            "Trying to read a gSCAN dataset from a non-existing file {}.".format(path_to_data)
        # only needs vocabulary for the output action sequences
        if not generate_vocabulary:
            assert os.path.exists(os.path.join(save_directory, target_vocabulary_file)), \
                "Trying to load vocabularies from non-existing files."
        if split == "test" and generate_vocabulary:
            logger.warning("WARNING: generating a vocabulary from the test set.")
        self.dataset = GroundedScan.load_dataset_from_file(path_to_data, save_directory=save_directory, k=k)
        if self.dataset._data_statistics.get("adverb_1"):
            logger.info("Verb-adverb combinations in training set: ")
            for adverb, items in self.dataset._data_statistics["train"]["verb_adverb_combinations"].items():
                logger.info("Verbs for adverb: {}".format(adverb))
                for key, count in items.items():
                    logger.info("   {}: {} occurrences.".format(key, count))
            logger.info("Verb-adverb combinations in dev set: ")
            for adverb, items in self.dataset._data_statistics["dev"]["verb_adverb_combinations"].items():
                logger.info("Verbs for adverb: {}".format(adverb))
                for key, count in items.items():
                    logger.info("   {}: {} occurrences.".format(key, count))
        self.image_dimensions = self.dataset.situation_image_dimension
        self.image_channels = 3
        self.split = split
        self.directory = save_directory
        self.max_seq_length = max_seq_length

        # Keeping track of data.
        # key: input command, value: list of examples
        self._input_commands = set()
        self._examples = defaultdict(list)
        self._target_lengths = defaultdict(list)
        if generate_vocabulary:
            logger.info("Generating vocabularies...")
            self.target_vocabulary = Vocabulary()
            self.read_vocabularies()
            logger.info("Done generating vocabularies.")
        else:
            logger.info("Loading vocabularies...")
            self.target_vocabulary = Vocabulary.load(os.path.join(save_directory, target_vocabulary_file))
            logger.info("Done loading vocabularies.")

    def read_vocabularies(self) -> {}:
        """
        Loop over all examples in the dataset and add the words in them to the vocabularies.
        """
        logger.info("Populating vocabulary...")
        for i, example in enumerate(self.dataset.get_examples_with_image(self.split)):
            self.target_vocabulary.add_sentence(example["target_command"])

    def save_vocabularies(self, target_vocabulary_file: str):
        self.target_vocabulary.save(os.path.join(self.directory, target_vocabulary_file))

    def get_vocabulary(self, vocabulary: str) -> Vocabulary:
        if vocabulary == "target":
            vocab = self.target_vocabulary
        else:
            raise ValueError("Specified unknown vocabulary in sentence_to_array: {}".format(vocabulary))
        return vocab

    def shuffle_data(self) -> {}:
        """
        Reorder the data examples and reorder the lengths of the input and target commands accordingly.
        """
        random.shuffle(self._input_commands)

    def get_data_iterator(self, batch_size=10, is_test=False) -> Tuple[torch.Tensor,
            List[int], torch.Tensor, List[dict], torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
        """
        Iterate over batches of example tensors, pad them to the max length in the batch and yield.
        :param batch_size: how many examples to put in each batch.
        :param is_test: running test or not.
        :return: tuple of input commands batch, situation image batch,
        list of corresponding situation representations, target commands batch and corresponding target lengths.
        """
        for input_text in self._input_commands:
            for example_i in range(0, len(self._examples[input_text]), batch_size):
                examples = self._examples[input_text][example_i:example_i + batch_size]
                target_lengths = self._target_lengths[input_text][example_i:example_i + batch_size]
                max_target_length = np.max(target_lengths)
                input_text_batch = []
                target_batch = []
                situation_batch = []
                situation_representation_batch = []
                derivation_representation_batch = []
                agent_positions_batch = []
                target_positions_batch = []
                for example in examples:
                    to_pad_target = max_target_length - example["target_tensor"].size(1)
                    padded_target = torch.cat([
                        example["target_tensor"],
                        torch.zeros(int(to_pad_target), dtype=torch.long).unsqueeze(0)], dim=1)
                    to_pad_situation = max_target_length - example["situation_tensor"].size(1)
                    if to_pad_situation > 0 and not is_test:
                        situation_size = example["situation_tensor"].shape
                        pad_size = (to_pad_situation, situation_size[2], situation_size[3], situation_size[4])
                        padded_situation = torch.cat([
                            example["situation_tensor"],
                            torch.zeros(pad_size).unsqueeze(0)], dim=1)
                    else:
                        padded_situation = example["situation_tensor"]
                    input_text_batch.append(input_text)
                    target_batch.append(padded_target)
                    situation_batch.append(padded_situation)
                    situation_representation_batch.append(example["situation_representation"])
                    derivation_representation_batch.append(example["derivation_representation"])
                    agent_positions_batch.append(example["agent_position"])
                    target_positions_batch.append(example["target_position"])

                yield (input_text_batch, derivation_representation_batch,
                       torch.cat(situation_batch, dim=0), situation_representation_batch, torch.cat(target_batch, dim=0),
                       target_lengths, torch.cat(agent_positions_batch, dim=0), torch.cat(target_positions_batch, dim=0))

    def read_dataset(self, max_examples=None, simple_situation_representation=True, is_test=False) -> {}:
        """
        Loop over the data examples in GroundedScan and convert them to tensors, also save the lengths
        for input and target sequences that are needed for padding.
        :param max_examples: how many examples to read maximally, read all if None.
        :param simple_situation_representation: whether to read the full situation image in RGB or the simplified
        smaller representation.
        :param is_test: running test or not.
        """
        logger.info("Converting dataset to tensors...")
        for i, example in enumerate(self.dataset.get_examples_with_image(self.split, simple_situation_representation)):
            if max_examples:
                if len(self._examples) > max_examples:
                    return
            empty_example = {}
            input_commands = example["input_command"]
            input_text = ' '.join(input_commands)
            target_commands = example["target_command"]
            situation_images = example["situation_images"]
            if i == 0:
                self.image_dimensions = situation_images[0].shape[0]
                self.image_channels = situation_images[0].shape[-1]
            situation_repr = example["situation_representation"]
            target_array = self.sentence_to_array(target_commands, vocabulary="target")
            if len(target_array) > self.max_seq_length+1:  # skip the last target token
                continue
            empty_example["target_tensor"] = torch.tensor(target_array, dtype=torch.long).unsqueeze(
                dim=0)
            if is_test:
                empty_example["situation_tensor"] = torch.tensor(situation_images[:1], dtype=torch.float).unsqueeze(dim=0)
            else:
                empty_example["situation_tensor"] = torch.tensor(situation_images, dtype=torch.float).unsqueeze(dim=0)
            empty_example["situation_representation"] = situation_repr
            empty_example["derivation_representation"] = example["derivation_representation"]
            empty_example["agent_position"] = torch.tensor(
                (int(situation_repr["agent_position"]["row"]) * int(situation_repr["grid_size"])) +
                int(situation_repr["agent_position"]["column"]), dtype=torch.long).unsqueeze(dim=0)
            empty_example["target_position"] = torch.tensor(
                (int(situation_repr["target_object"]["position"]["row"]) * int(situation_repr["grid_size"])) +
                int(situation_repr["target_object"]["position"]["column"]),
                dtype=torch.long).unsqueeze(dim=0)
            # update data
            self._input_commands.add(input_text)
            self._target_lengths[input_text].append(len(target_array))
            self._examples[input_text].append(empty_example)
        self._input_commands = list(self._input_commands)

    def sentence_to_array(self, sentence: List[str], vocabulary: str) -> List[int]:
        """
        Convert each string word in a sentence to the corresponding integer from the vocabulary and append
        a start-of-sequence and end-of-sequence token.
        :param sentence: the sentence in words (strings)
        :param vocabulary: whether to use the input or target vocabulary.
        :return: the sentence in integers.
        """
        vocab = self.get_vocabulary(vocabulary)
        sentence_array = []
        for word in sentence:
            sentence_array.append(vocab.word_to_idx(word))
        sentence_array.append(vocab.eos_idx)
        return sentence_array

    def array_to_sentence(self, sentence_array: List[int]) -> List[str]:
        """
        Translate each integer in a sentence array to the corresponding word.
        :param sentence_array: array with integers representing words from the vocabulary.
        :return: the sentence in words.
        """
        vocab = self.target_vocabulary
        return [vocab.idx_to_word(word_idx) for word_idx in sentence_array]

    @property
    def num_examples(self):
        n_examples = 0
        for text in self._examples.keys():
            n_examples += len(self._examples[text])
        return n_examples

    @property
    def target_vocabulary_size(self):
        return self.target_vocabulary.size
