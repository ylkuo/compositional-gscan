import argparse
import logging
import os
import torch
import torch.optim as optim
import yaml

from datetime import datetime
from easydict import EasyDict as edict
from gSCAN_dataset import GroundedScanDataset
from language.grammar import *
from language.parse import ConstituencyParser, StanfordDependencyParser
from logger import *
from model import *
from predict import evaluate
from torch.utils.tensorboard import SummaryWriter


def setup_summary_writer(args):
    now = datetime.now()
    now_str = now.strftime("%Y%m%d-%H%M%S")
    lr_str = str(args.lr).split('.')[1]
    dir_name = 'runs/datetime=%s_split=%s_k=%d_lr=%s_rnndim=%d_rnndepth=%d_ours_parse=%s' \
        % (now_str, args.split, args.k, lr_str, args.rnn_dim, args.rnn_depth, args.parse_type)
    if args.no_attention:
        dir_name += '_noatt'
    if args.pass_state:
        dir_name += '_passstate'
    if args.split == 'target_length_split':
        dir_name += '_seqlen='+str(args.max_seq_length)
    args.writer = SummaryWriter(dir_name)
    return args


def train(args):
    if args.max_seq_length <= 0:
        args.max_seq_length = np.inf
    # load training data
    training_set = GroundedScanDataset(args.data_path,
                                       args.data_directory + args.split + '/',
                                       split="train",
                                       target_vocabulary_file=args.target_vocabulary_file,
                                       k=args.k,
                                       max_seq_length=args.max_seq_length)
    training_set.read_dataset(max_examples=None,  # use all dataset
                              simple_situation_representation=args.simple_situation_representation)
    training_set.shuffle_data()
    # load validation data
    validation_set = GroundedScanDataset(args.data_path,
                                         args.data_directory + args.split + '/',
                                         split="dev",
                                         target_vocabulary_file=args.target_vocabulary_file,
                                         k=args.k,
                                         max_seq_length=args.max_seq_length)
    validation_set.read_dataset(max_examples=None,  # use all dataset
                                simple_situation_representation=args.simple_situation_representation)
    validation_set.shuffle_data()
    parser = None
    if args.parse_type == 'default':
        grammar = Grammar()
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
    data_iter = training_set.get_data_iterator(batch_size=args.training_batch_size)
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
                            compare_weight=args.compare_weight,
                            normalize_size=args.normalize_size,
                            no_attention=args.no_attention,
                            parse_type=args.parse_type,
                            pass_state=args.pass_state)
    n_update = 0; n_validate = 0; n_checkpoint = 0; best_match = 0
    if args.resume_from_file != '':
        resume_file = args.model_prefix + args.resume_from_file
        assert os.path.isfile(resume_file), "No checkpoint found at {}".format(resume_file)
        args.logger.info("Loading checkpoint from file at '{}'".format(resume_file))
        model.load_state_dict(torch.load(resume_file)[0])
        n_checkpoint = args.resume_n_update
        n_update = args.checkpoint_range * n_checkpoint
        n_validate = n_update / args.validate_every 
    else:
        torch.save([model.state_dict()], args.model_prefix + '/model_0.pkl')
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.adam_beta_1, args.adam_beta_2))
    # training
    training_set.shuffle_data()
    for i in range(args.n_epochs):
        for j, data in enumerate(training_set.get_data_iterator(batch_size=args.training_batch_size)):
            model.train()
            input_text_batch, _, situation_batch, situation_representation_batch, \
                   target_batch, target_lengths, agent_positions, target_positions = data
            if args.parse_type == 'default':
                arg_tree = grammar.arg_tree(split_str(input_text_batch[0]))
            else:
                arg_tree = parser.parse(input_text_batch[0])
            args.logger.info('train {}, arg tree: {}'.format(input_text_batch[0], arg_tree))
            model.update_words(arg_tree)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=args.device)
            success, total_loss, word_losses = model.loss(situation_batch, target_batch, target_lengths, optimizer)
            if not success:
                continue
            args.logger.info('epoch {}, iter {}, train loss: {}'.format(i, j, float(total_loss)))
            # save checkpoints
            if n_update % args.checkpoint_range == 0:
                log_model_params(model, args.writer, 'comp_gscan', n_update)
                # log numbers, TODO: log loss per word
                args.writer.add_scalar('loss/train_total', float(total_loss), n_checkpoint)
                model_path = args.model_prefix + '/model_' + str(n_checkpoint) + '.pkl'
                torch.save([model.state_dict()], model_path)
                n_checkpoint += 1
            # validation
            if n_update % args.validate_every == 0:
                validation_set.shuffle_data()
                model.eval()
                # compute loss
                loss = 0; n_batch = 0
                for k, data in enumerate(validation_set.get_data_iterator(batch_size=args.training_batch_size)):
                    input_text_batch, _, situation_batch, situation_representation_batch, \
                           target_batch, target_lengths, agent_positions, target_positions = data
                    if args.parse_type == 'default':
                        arg_tree = grammar.arg_tree(split_str(input_text_batch[0]))
                    else:
                        arg_tree = parser.parse(input_text_batch[0])
                    model.update_words(arg_tree)
                    with torch.no_grad():
                        target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=args.device)
                        success, total_loss, word_losses = model.loss(situation_batch, target_batch, target_lengths)
                        loss += float(total_loss)
                    n_batch += 1
                loss = loss/n_batch
                args.logger.info('epoch {}, iter {}, val loss: {}'.format(i, j, float(loss)))
                args.writer.add_scalar('loss/val_total', float(loss), n_validate)
                # run evaluation
                accuracy, exact_match = evaluate(training_set,
                        validation_set.get_data_iterator(batch_size=1), model=model, world=validation_set.dataset._world,
                        max_steps=args.max_steps, vocab=validation_set.target_vocabulary,
                        max_examples_to_evaluate=args.max_testing_examples, device=args.device, parser=parser)
                args.logger.info("  Evaluation Accuracy: %5.2f Exact Match: %5.2f" % (accuracy, exact_match))
                args.writer.add_scalar('accuracy/val_total', accuracy, n_validate)
                args.writer.add_scalar('exact_match/val_total', exact_match, n_validate)
                # save the best model
                if exact_match > best_match:
                    model_path = args.model_prefix + '/model_best.pkl'
                    torch.save([model.state_dict(), n_update, exact_match], model_path)
                    best_match = exact_match
                    args.logger.info('save best model at n_update {}'.format(n_update))
                n_validate += 1
            n_update += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a compositional model for gSCAN.')
    parser.add_argument('--config', default='configs/config_train.yaml', type=str,
                        help='yaml config file')
    parser.add_argument('--k', default=0, type=int,
                        help='number of sentences with the new adverb')
    parser.add_argument('--model_prefix', default='', type=str,
                        help='location prefix to save model')
    parser.add_argument('--max_seq_length', default=-1, type=int,
                        help='max target sequence length in training')
    config_args = parser.parse_args()
    args = edict(yaml.load(open(config_args.config, 'r'), Loader=yaml.FullLoader))
    if config_args.k > 0:
        args.k = config_args.k
    if config_args.model_prefix != '':
        args.model_prefix = config_args.model_prefix
    if config_args.max_seq_length > 0:
        args.max_seq_length = config_args.max_seq_length
    args.data_path = args.data_directory + args.split + '/dataset.txt'
    if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
        args.device_id = os.environ['CUDA_VISIBLE_DEVICES']
    if args.use_cuda:
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            args.device = torch.device('cuda')
        else:
            torch.cuda.set_device(args.device_id)
            args.device = torch.device('cuda:'+str(args.device_id))
    else:
        args.device = torch.device('cpu')
    log_level = logging.DEBUG if args.is_debug else logging.INFO
    args.logger = get_logger('compositional_gscan_train', level=log_level)
    args = setup_summary_writer(args)
    train(args)
