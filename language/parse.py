import stanza

from allennlp.predictors.predictor import Predictor
from language.grammar import reorg_arg_tree, WORDS, WORD2NARG


class ArgNode(object):
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parent = None

    @property
    def narg(self):
        return len(self.children)

    def __str__(self):
        if self.narg > 0:
            children_str = ', '.join([c.__str__() for c in self.children])
            return self.name + ' [ ' + children_str + ' ]'
        else:
            return self.name

def to_arg_tree(parse, parent=None):
    node = None
    value = parse['word']
    pos = parse['attributes'][0]
    if pos not in ['S', 'ADP', 'DET', 'SCONJ']:
        if value in ['zigzagging', 'spinning']:
            node = ArgNode('while ' + value)
            node.parent = parent
            if parent: parent.children.append(node)
        elif len(value.split(' ')) > 1:  # intermediate nodes
            node = ArgNode(pos)
            node.parent = parent
            if parent: parent.children.append(node)
            for sub_tree in parse['children']:
                value = sub_tree['word']
                pos = sub_tree['attributes'][0]
                if pos in ['IN', 'DT']:  # constituency parse
                    continue
                if pos == 'SBAR' or pos == 'ADVP':
                    node.children.append(ArgNode(value))
                    node.children[-1].parent = node
                    continue
                if pos == 'PP':
                    to_arg_tree(sub_tree['children'][1], node)
                    continue
                to_arg_tree(sub_tree, node)
        else:
            node = ArgNode(value)
            node.parent = parent
            if parent: parent.children.append(node)
            if 'children' in parse.keys():
                for sub_tree in parse['children']:
                    to_arg_tree(sub_tree, node)
    else:
        if 'children' in parse.keys():
            for sub_tree in parse['children']:
                node = to_arg_tree(sub_tree, parent)
    return node


def relabel_narg(node):
    node.name = node.name + '_' + str(node.narg)
    for sub_tree in node.children:
        relabel_narg(sub_tree)


class Parser(object):
    def __init__(self):
        self.model = None
        self.words = []  # including leaves and intermediate nodes
        self.word2narg = {}  # number of args supported for each node

    def parse(self, text):
        out = self.model.predict(text)
        root = out['hierplane_tree']['root']
        arg_tree = to_arg_tree(root)
        relabel_narg(arg_tree)
        return arg_tree


class ConstituencyParser(Parser):
    def __init__(self):
        super().__init__()
        self.model = predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
        self.words = WORDS
        self.words.extend(['VP', 'NP'])
        self.word2narg = WORD2NARG
        for key in self.word2narg.keys():
            self.word2narg[key] = 0
        self.word2narg['VP'] = [2, 3]
        self.word2narg['NP'] = [1, 2, 3]
        self.word2narg['cautiously'] = [0, 1]


class StanfordDependencyParser(Parser):
    def __init__(self):
        self.model = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')
        self.words = WORDS
        self.word2narg = WORD2NARG
        for key in self.word2narg.keys():
            self.word2narg[key] = 0
        self.word2narg['walk'] = [1, 2]
        self.word2narg['push'] = [1, 2]
        self.word2narg['pull'] = [1, 2]
        self.word2narg['circle'] = [0, 1, 2]
        self.word2narg['square'] = [0, 1, 2]
        self.word2narg['cylinder'] = [0, 1, 2]

    def parse(self, text):
        doc = self.model(text)
        sub_tree = {}
        for head_idx in range(len(doc.sentences[0].words)+1):
            for word in doc.sentences[0].words:
                if word.head != head_idx:
                    continue
                if word.text in ['to', 'a', 'while', 'the']:
                    continue
                if word.text in ['zigzagging', 'spinning']:
                    node = ArgNode('while ' + word.text)
                else:
                    node = ArgNode(word.text)
                if word.head > 0:
                    node.parent = sub_tree[word.head]
                    sub_tree[word.head].children.append(node)
                sub_tree[word.id] = node
        relabel_narg(sub_tree[1])
        return sub_tree[1]


if __name__ == '__main__':
    texts = ['walk to a green big square while spinning',
             'walk to a red circle while spinning',
             'walk to a circle while zigzagging',
             'walk to a red big circle cautiously',
             'walk to a big circle cautiously',
             'walk to a circle cautiously',
             'walk to a red big circle',
             'walk to a red circle',
             'walk to a circle',
             'push a red big circle while spinning',
             'push a big circle while spinning',
             'push a circle while spinning',
             'push a square while spinning',
             'push a red big circle cautiously',
             'push a big circle cautiously',
             'push a circle cautiously',
             'push a green big square',
             'push a big square',
             'push a square']
    c_parser = ConstituencyParser()
    sd_parser = StanfordDependencyParser()
    for text in texts:
        print(text)
        print(' C: ', c_parser.parse(text))
        print(' D: ', d_parser.parse(text))
        print(' S: ', sd_parser.parse(text))
