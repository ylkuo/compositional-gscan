import argparse
import logging
import os
import torch
import yaml

from easydict import EasyDict as edict
from gSCAN_dataset import GroundedScanDataset
from language.grammar import *
from language.parse import ConstituencyParser, StanfordDependencyParser
from logger import *
from model import *
from predict import predict_and_save


def test(args):
    splits = args.splits.split(',')
    parser = None
    if args.parse_type == 'default':
        word2narg = WORD2NARG
    else:
        if args.parse_type == 'constituency':
            parser = ConstituencyParser()
        elif args.parse_type == 'dependency':
            parser = StanfordDependencyParser()
        word2narg = parser.word2narg
    if args.compare_attention:
        compare_list = COMPARE_LIST
    else:
        compare_list = None
    for split in splits:
        args.logger.info("Loading {} dataset split...".format(split))
        test_set = GroundedScanDataset(args.data_path,
                                       args.data_directory + args.split + '/',
                                       split=split,
                                       target_vocabulary_file=args.target_vocabulary_file,
                                       k=args.k)
        test_set.read_dataset(max_examples=None,  # use all dataset
                              simple_situation_representation=args.simple_situation_representation,
                              is_test=True)
        args.logger.info("Done Loading {} dataset split.".format(split))
        args.logger.info("  Loaded {} examples.".format(test_set.num_examples))
        args.logger.info("  Output vocabulary size: {}".format(test_set.target_vocabulary_size))
        args.logger.info("  Most common target words: {}".format(test_set.target_vocabulary.most_common(5)))

        # set up model
        data_iter = test_set.get_data_iterator(batch_size=1, is_test=True)
        input_text_batch, _, situation_batch, situation_representation_batch, \
                target_batch, target_lengths, agent_positions, target_positions = next(data_iter)
        example_feature = situation_batch[0][0]  # first seq, first observation
        model = SentenceNetwork(words=word2narg,
                                cnn_kernel_size=args.cnn_kernel_size,
                                n_channels=args.cnn_num_channels,
                                example_feature=example_feature,
                                rnn_dim=args.rnn_dim,
                                rnn_depth=args.rnn_depth,
                                attention_dim=args.att_dim,
                                output_dim=args.output_dim,
                                device=args.device,
                                compare_list=compare_list,
                                normalize_size=args.normalize_size,
                                no_attention=args.no_attention,
                                parse_type=args.parse_type,
                                pass_state=args.pass_state)

        # Load model and vocabularies
        resume_file = args.model_prefix + args.resume_from_file
        assert os.path.isfile(resume_file), "No checkpoint found at {}".format(resume_file)
        args.logger.info("Loading checkpoint from file at '{}'".format(resume_file))
        model.load_state_dict(torch.load(resume_file)[0])
        model.to(args.device)

        # Run test
        performance_by_length = split == 'target_lengths'

        output_file_name = "_".join([split, args.output_file_name])
        output_file_path = os.path.join(args.output_directory, output_file_name)
        output_file = predict_and_save(dataset=test_set, model=model,
                                       output_file_path=output_file_path,
                                       max_steps=args.max_steps,
                                       device=args.device,
                                       save=args.save_prediction,
                                       performance_by_length=performance_by_length,
                                       parser=parser)
        args.logger.info("Saved predictions to {}".format(output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a compositional model for gSCAN.')
    parser.add_argument('--config', default='configs/config_test.yaml', type=str,
                        help='yaml config file')
    parser.add_argument('--model_prefix', default='', type=str,
                        help='location prefix to save model')
    parser.add_argument('--output_directory', default='', type=str,
                        help='folder to put the output files')
    config_args = parser.parse_args()
    args = edict(yaml.load(open(config_args.config, 'r'), Loader=yaml.FullLoader))
    if config_args.model_prefix != '':
        args.model_prefix = config_args.model_prefix
    if config_args.output_directory != '':
        args.output_directory = config_args.output_directory
    args.data_path = args.data_directory + args.split + '/dataset.txt'
    if args.use_cuda:
        torch.cuda.set_device(args.device_id)
        args.device = torch.device('cuda:'+str(args.device_id))
    else:
        args.device = torch.device('cpu')
    log_level = logging.DEBUG if args.is_debug else logging.INFO
    args.logger = get_logger('compositional_gscan_test', level=log_level)
    test(args)

