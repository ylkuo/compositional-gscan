import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import json

from collections import defaultdict
from gSCAN_dataset import GroundedScanDataset, Vocabulary
from GroundedScan.world import Situation, World
from language.grammar import Grammar, split_str
from typing import Iterator, List, Tuple


logger = logging.getLogger(__name__)


def sequence_accuracy(prediction: List[int], target: List[int]) -> float:
    correct = 0
    total = 0
    prediction = prediction.copy()
    target = target.copy()
    if len(prediction) < len(target):
        difference = len(target) - len(prediction)
        prediction.extend([0] * difference)
    if len(target) < len(prediction):
        difference = len(prediction) - len(target)
        target.extend([-1] * difference)
    for i, target_int in enumerate(target):
        if i >= len(prediction):
            break
        prediction_int = prediction[i]
        if prediction_int == target_int:
            correct += 1
        total += 1
    if not total:
        return 0.
    return (correct / total) * 100


def predict_and_save(dataset: GroundedScanDataset, model: nn.Module, output_file_path: str,
                     max_steps: int, device=None, save=True, performance_by_length=False,
                     parser=None):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param max_steps: after how many steps to force quit decoding
    :param device: device to put the tensors
    :param save: save the predictions as json file or not
    :param performance_by_length: log performance by sequence length
    :param parser: the parser used for deriving the model
    """

    with open(output_file_path, mode='w') as outfile:
        output = []
        i = 0
        if performance_by_length:
            n_exact_match = defaultdict(int)
            n_examples = defaultdict(int)
        else:
            n_exact_match = 0
        with torch.no_grad():
            for (success, input_text, init_situ_spec, situations, output_sequence, target_sequence, att_maps) in predict(
                    dataset.get_data_iterator(batch_size=1, is_test=True),
                    world=dataset.dataset._world, model=model,
                    max_steps=max_steps, vocab=dataset.target_vocabulary,
                    pad_idx=dataset.target_vocabulary.pad_idx, sos_idx=dataset.target_vocabulary.sos_idx,
                    eos_idx=dataset.target_vocabulary.eos_idx, device=device, parser=parser):
                i += 1
                if not success:
                    if performance_by_length:
                        n_examples[len(target_str_sequence)] += 1
                    continue
                accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[:-1])
                target_str_sequence = dataset.array_to_sentence(target_sequence[0].tolist())
                target_str_sequence = target_str_sequence[:-1]  # Get rid of <SOS> and <EOS>
                output_str_sequence = dataset.array_to_sentence(output_sequence)
                exact_match = True if int(accuracy) == 100 else False
                if save:
                    output.append({"input": input_text, "prediction": output_str_sequence,
                                   "target": target_str_sequence, "situation": init_situ_spec,
                                   "attention_weights": att_maps,
                                   "accuracy": accuracy,
                                   "exact_match": exact_match})
                if exact_match:
                    if performance_by_length:
                        n_exact_match[len(target_str_sequence)] += 1
                    else:
                        n_exact_match += 1
                if performance_by_length:
                    n_examples[len(target_str_sequence)] += 1
        logger.info("Wrote predictions for {} examples.".format(i))
        if performance_by_length:
            logger.info("Percentage of exact match by target lengths:")
            for l in n_examples.keys():
                logger.info(" {}, {}, {}, {}".format(l, n_exact_match[l], n_examples[l],
                    n_exact_match[l]/float(n_examples[l])))
            logger.info("Percentage of exact match: {}".format(sum(n_exact_match.values())/float(sum(n_examples.values()))))
        else:
            logger.info("Num of exact match: {}, total: {}".format(n_exact_match, i))
            logger.info("Percentage of exact match: {}".format(n_exact_match/float(i)))
        if save:
            json.dump(output, outfile, indent=4)
    return output_file_path


def evaluate(dataset: GroundedScanDataset, data_iterator: Iterator, model: nn.Module,
             world: World, max_steps: int, vocab: Vocabulary, max_examples_to_evaluate=None,
             device=None, parser=None) -> Tuple[float, float]:
    accuracies = []
    n_exact_match = 0
    for (success, input_text, init_situ_spec, situations, output_sequence, target_sequence, att_maps) in predict(
            data_iterator, world=world, model=model, max_steps=max_steps, vocab=vocab,
            pad_idx=vocab.pad_idx, sos_idx=vocab.sos_idx, eos_idx=vocab.eos_idx,
            device=device, parser=parser):
        accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[:-1])
        output_str_sequence = dataset.array_to_sentence(output_sequence)
        exact_match = True if int(accuracy) == 100 else False
        if int(accuracy) == 100:
            n_exact_match += 1
        accuracies.append(accuracy)
    return (float(np.mean(np.array(accuracies))), (n_exact_match / len(accuracies)) * 100)


def initialize_world(world: World, situation: Situation) -> World:
    """
    Initializes the world with the passed situation.
    :param world: a simulated world for grounded SCAN
    :param situation: class describing the current situation in the world, fully determined by a grid size,
    agent position, agent direction, list of placed objects, an optional target object and optional carrying object.
    """
    objects = []
    for positioned_object in situation.placed_objects:
        objects.append((positioned_object.object, positioned_object.position))
    world.initialize(objects, agent_position=situation.agent_pos, agent_direction=situation.agent_direction,
                     target_object=situation.target_object, carrying=situation.carrying)
    return world


def predict(data_iterator: Iterator, world: World, model: nn.Module, max_steps: int,
            vocab: Vocabulary, pad_idx: int, sos_idx: int, eos_idx: int,
            max_examples_to_evaluate=None, device=None, parser=None) -> torch.Tensor:
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param world: a simulated world for grounded SCAN
    :param model: a trained model from model.py
    :param max_steps: after how many steps to abort decoding
    :param vocab: Vocabulary of the dataset
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    :param device: device to put the tensors
    :param parser: the parser used for deriving the model
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()
    grammar = Grammar()

    # Loop over the data.
    for i, data in enumerate(data_iterator):
        input_text, _, situation_sequence, situation_spec, \
                target_sequence, target_lengths, agent_positions, target_positions = data
        if max_examples_to_evaluate:
            if i+1 > max_examples_to_evaluate:
                break
        if parser is None:
            arg_tree = grammar.arg_tree(split_str(input_text[0]))
        else:
            arg_tree = parser.parse(input_text[0])

        model.update_words(arg_tree)
        # Prepare the initial env
        situation = Situation.from_representation(situation_spec[0])
        world.clear_situation()
        world = initialize_world(world, situation)
        out_spec = [world.get_current_situation()]
        feature = torch.tensor(world.get_current_situation_grid_repr(),
                               dtype=torch.float, device=device)
        feature = feature.unsqueeze(0)

        # Iteratively decode the output.
        # TODO: retrieve attention as well.
        output_sequence = []; out_att_maps = defaultdict(list)
        token = torch.tensor([sos_idx], dtype=torch.long, device=device)
        decoding_iteration = 0
        proposals = None
        hidden_states = None
        success = True
        while token != eos_idx and decoding_iteration <= max_steps:
            # Take one forward step
            try:
                proposals, hidden_states = model.forward(obs=feature, prev_hidden_states=hidden_states)
            except:
                success = False
                break
            output = F.log_softmax(proposals[len(model.current_words)-1], dim=-1)
            token = output.max(dim=-1)[1].data[0].item()
            decoding_iteration += 1
            if token != sos_idx and token != pad_idx:
                output_sequence.append(token)
            att_maps = model.att_maps
            for j, att_map in enumerate(att_maps):
                if att_map is None:
                    continue
                if type(att_map) is list:
                    other_word = 'small'
                    if model.current_words[j] == 'small':
                        other_word = 'big'
                    # current word
                    tmp_att_map = att_map[0]
                    att_size = int(np.sqrt(tmp_att_map.shape[1]))
                    out_att_maps[model.current_words[j]].append(tmp_att_map[0].view(att_size, att_size).cpu().data.numpy().tolist())
                    # word for comparison
                    tmp_att_map = att_map[1]
                    out_att_maps[other_word].append(tmp_att_map[0].view(att_size, att_size).cpu().data.numpy().tolist())
                else:
                    att_size = int(np.sqrt(att_map.shape[1]))
                    out_att_maps[model.current_words[j]].append(att_map[0].view(att_size, att_size).cpu().data.numpy().tolist())
            # Agent moves and update the input feature if not reaching the end
            if token not in [eos_idx, sos_idx, pad_idx]:
                target_command = vocab.idx_to_word(token)
                world.execute_command(target_command)
                out_spec.append(world.get_current_situation())
                feature = torch.tensor(world.get_current_situation_grid_repr(),
                                       dtype=torch.float, device=device)
                feature = feature.unsqueeze(0)

        if len(output_sequence) > 0 and output_sequence[-1] == eos_idx:
            output_sequence.pop()
        del situation
        del feature
        yield (success, input_text, situation_spec[0], out_spec, output_sequence, target_sequence, out_att_maps)

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))
