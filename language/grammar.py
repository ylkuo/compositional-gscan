from collections import defaultdict
from nltk import CFG
from nltk import data as nltk_data
from nltk.parse.generate import generate
from nltk.parse import BottomUpLeftCornerChartParser

import os
import random


TERMINALS = ['VV_i', 'VV_t', 'RB', 'NN', 'JJ', 'DIR', 'DET']
WORDS = ['walk', 'push', 'pull', 'while spinning', 'while zigzagging',
         'hesitantly', 'cautiously', 'circle', 'square', 'cylinder',
         'red', 'green', 'blue', 'yellow', 'big', 'small']
WORD2NARG = {'walk': 1, 'push': 1, 'pull': 1, 'while spinning': 1, 'while zigzagging': 1,
             'hesitantly': 1, 'cautiously': 1, 'circle': 0, 'square': 0, 'cylinder': 0,
             'red': 1, 'green': 1, 'blue': 1, 'yellow': 1, 'big': 1, 'small': 1}
ADVERBS = ['while spinning', 'while zigzagging', 'hesitantly', 'cautiously']
SIZES = ['big', 'small']
COMPARE_LIST = [SIZES]


def to_arg_tree(parse, terminals):
    args = []
    for sub_tree in parse:
        if sub_tree.label() in terminals:
            value = sub_tree.leaves()[0]
            pos = sub_tree.label()
            if pos in ['DIR', 'DET']:
                continue
            args.append(value)
        else:
            new_args = to_arg_tree(sub_tree, terminals)
            args.append(new_args)
    return args


def reorg_arg_tree(arg_tree):
    if len(arg_tree) == 1:
        if type(arg_tree[0]) is list:
            return reorg_arg_tree(arg_tree[0])
        else:
            return [arg_tree[0]]
    elif len(arg_tree) == 2:
        if type(arg_tree[0]) is str:
            return [arg_tree[0], reorg_arg_tree(arg_tree[1])]
        else:
            return [arg_tree[1], reorg_arg_tree(arg_tree[0])]
    return arg_tree


def split_str(text):
    """Special split to make sure `while verb-ing` are combined together"""
    parts = text.split(' ')
    out = []; skip = False
    for i, part in enumerate(parts):
        if skip:
            skip = False
            continue
        if part == 'while':
            out.append(part + ' ' + parts[i+1])
            skip = True
        else:
            out.append(part)
    return out


class Grammar(object):
    def __init__(self, cfg_filename='', terminals=TERMINALS):
        if cfg_filename == '':
            dir_path = os.path.dirname(os.path.realpath(__file__))
            cfg_filename = dir_path + '/grammar.cfg'
        self.grammar = self._load_from_file(cfg_filename)
        self.parser = BottomUpLeftCornerChartParser(self.grammar)
        self.terminals = terminals

    def sample_sentence(self, cfactor=0.5):
        pcount = defaultdict(int)

        def weighted_choice(weights):
            rnd = random.random() * sum(weights)
            for i, w in enumerate(weights):
                rnd -= w
                if rnd < 0:
                    return i

        def generate_sample(grammar, prod, frags):
            if prod in grammar._lhs_index:  # derivation
                derivations = grammar._lhs_index[prod]
                weights = []
                for prod in derivations:
                    if prod in pcount:
                       weights.append(cfactor ** (pcount[prod]))
                    else:
                        weights.append(1.0)
                # tend to not sample the already expanded productions
                derivation = derivations[weighted_choice(weights)]
                pcount[derivation] += 1
                for d in derivation._rhs:
                    generate_sample(grammar, d, frags)
                pcount[derivation] -= 1
            elif prod in grammar._rhs_index:  # terminal
                prod = str(prod)
                frags.append(prod)
        frags = []
        generate_sample(self.grammar, self.grammar.start(), frags)
        parse = self.parser.parse(frags)
        return frags

    def arg_tree(self, sentence):
        for p in self.parser.parse(sentence):
            parse = p
            break
        arg_tree = to_arg_tree(parse, self.terminals)
        arg_tree = reorg_arg_tree(arg_tree)
        return arg_tree

    def _load_from_file(self, filename):
        url = 'file:%s' % filename
        url.replace('\\', '/')
        return nltk_data.load(url)


if __name__ == '__main__':
    grammar = Grammar()
    for i in range(10):
        sentence = grammar.sample_sentence()
        print(i, sentence, grammar.arg_tree(sentence))

