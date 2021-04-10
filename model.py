import copy
import numpy as np
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNet(nn.Module):
    """Simple conv net. Convolves the input channels but retains input image width."""
    def __init__(self, num_channels: int, cnn_kernel_size: int, num_conv_channels: int, dropout_probability: float,
                 stride=1, input_shape=None):
        super(ConvolutionalNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=1,
                                padding=0, stride=stride)
        self.conv_2 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=5,
                                stride=stride, padding=2)
        self.conv_3 = nn.Conv2d(in_channels=num_channels, out_channels=num_conv_channels, kernel_size=cnn_kernel_size,
                                stride=stride, padding=cnn_kernel_size // 2)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU()
        layers = [self.relu, self.dropout]
        self.layers = nn.Sequential(*layers)
        self.cnn_output_shape = self.forward(torch.zeros(input_shape).unsqueeze(0)).shape

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        """
        :param input_images: [batch_size, image_width, image_width, image_channels]
        :return: [batch_size, image_width * image_width, num_conv_channels]
        """
        batch_size = input_images.size(0)
        input_images = input_images.transpose(1, 3)
        conved_1 = self.conv_1(input_images)
        conved_2 = self.conv_2(input_images)
        conved_3 = self.conv_3(input_images)
        images_features = torch.cat([conved_1, conved_2, conved_3], dim=1)
        _, num_channels, _, image_dimension = images_features.size()
        images_features = images_features.transpose(1, 3)
        images_features = self.layers(images_features)
        return images_features.reshape(batch_size, image_dimension * image_dimension, num_channels)


class AttentionArgN(nn.Module):
    def __init__(self, smp_dim, hidden_dim, attention_dim, n=1):
        super(AttentionArgN, self).__init__()
        self.hidden_dim = hidden_dim
        self.smp_att = []
        n = 1 if n == 0 else n
        for i in range(n):
            self.smp_att.append(nn.Linear(smp_dim, attention_dim))
            self.add_module('smp_att_{}_'.format(i), self.smp_att[i])
        self.init_hidden_att = nn.Linear(hidden_dim, attention_dim)
        self.dim = attention_dim*(n+1)
        self.full_att = nn.Linear(self.dim, 1)

    def forward(self, smp_out, rnn_hidden, in_atts):
        att_smp = []
        for i, in_att in enumerate(in_atts):
            att_smp.append(self.smp_att[i](smp_out*in_att.unsqueeze(2)))  # (batch_size, num_pixels, attention_dim)
        if len(in_atts) == 0:
            att_smp.append(self.smp_att[0](smp_out))
        att2 = self.init_hidden_att(rnn_hidden)  # (batch_size, attention_dim)
        att_smp.append(att2.unsqueeze(1).repeat(1, smp_out.shape[1], 1))
        emb = torch.cat(att_smp, dim=2)
        att = self.full_att(torch.relu(emb))
        alpha = F.softmax(att, dim=1)
        return alpha[:,:,0]


class ComponentNetwork(nn.Module):
    def __init__(self, nargs=0, smp_emb_dim=64, attention_dim=16,
                 rnn_dim=256, rnn_depth=1, output_dim=9, device=None,
                 related_words=None, normalize_att=False,
                 no_attention=False, pass_state=False):
        super(ComponentNetwork, self).__init__()

        self.device = device
        self.nargs = nargs
        self.related_words = related_words
        self.normalize_att = normalize_att
        self.no_attention = no_attention
        self.pass_state = pass_state
        # set up layers
        self.smp_emb_dim = smp_emb_dim
        self.output_dim = output_dim
        if not self.no_attention:
            if related_words is None:
                self.attention = AttentionArgN(smp_emb_dim, rnn_dim*rnn_depth, attention_dim, n=nargs)
            else:
                self.attention = {}
                for word in related_words:
                    self.attention[word] = AttentionArgN(smp_emb_dim, rnn_dim*rnn_depth,
                                                         attention_dim, n=nargs)
                    self.add_module('attention_{}_{}'.format(word, nargs), self.attention[word])
        if pass_state and self.nargs > 0:
            self.combine_state = nn.Linear(self.nargs*rnn_dim*rnn_depth, rnn_dim)
        self._set_rnn(rnn_dim, rnn_depth)
        self.hidden_to_output = nn.Linear(rnn_dim, output_dim)

    def get_parameter_str(self):
        ret = ''
        for p in self.parameters():
            ret = ret + '{0} {1}\n'.format(type(p.data), list(p.size()))
        return ret

    def _set_rnn(self, rnn_dim, rnn_depth):
        self.rnn_dim = rnn_dim
        self.rnn_depth = rnn_depth
        if self.pass_state and self.nargs > 0:
            self.rnn_input_dim = self.smp_emb_dim + rnn_dim
        else:
            self.rnn_input_dim = self.smp_emb_dim
        self.rnn = nn.GRU(self.rnn_input_dim, rnn_dim, rnn_depth)
        self.rnn.reset_parameters()
        # orthogonal initialization of recurrent weights
        for _, hh, _, _ in self.rnn.all_weights:
            for i in range(0, hh.size(0), self.rnn.hidden_size):
                nn.init.orthogonal_(hh[i:i + self.rnn.hidden_size])

    def forward(self, smp_emb=None, prev_hidden_state=None,
                in_att_maps=[], gen_proposal=False,
                in_states=[], in_actions=[], current_word=None):
        batch_size = smp_emb.shape[0]
        if prev_hidden_state is None:
            prev_hidden_state = torch.zeros(self.rnn_depth, batch_size, self.rnn_dim, device=self.device)
        # compute attention map
        att_in = smp_emb.view(batch_size, -1, self.smp_emb_dim)  # (batch_size, num_pixels, smp_emb_dim)
        prev_hidden_state = prev_hidden_state.permute(1,0,2)  # swap to batch first for operations
        prev_hidden_state = prev_hidden_state.contiguous().view(batch_size, -1)
        if self.no_attention:
            rnn_input = att_in.sum(dim=1)
            out_alpha = None
        else:
            if self.related_words is not None:
                other_word = list(set(self.related_words).difference(set([current_word])))[0]
                alpha = self.attention[current_word](att_in, prev_hidden_state, in_att_maps)
                related_alpha = self.attention[other_word](att_in, prev_hidden_state, in_att_maps)
                if self.normalize_att:
                    out_alpha = []
                    for k, a in enumerate([alpha, related_alpha]):
                        a = a - a.min(1, keepdim=True)[0]
                        maxes = a.max(1, keepdim=True)[0]
                        for b in range(maxes.shape[0]):
                            if maxes[b][0] == 0:
                                maxes[b][0] = 1
                        a = a / maxes
                        const = torch.ones(a.shape).to(self.device) * (10 ** -5)
                        a = a + const
                        a = torch.clamp(a, min=0., max=1.)
                        if k == 0:
                            alpha = a
                        out_alpha.append(a)
                else:
                    out_alpha = [alpha, related_alpha]
            else:
                alpha = self.attention(att_in, prev_hidden_state, in_att_maps)
                out_alpha = alpha
            rnn_input = (att_in * alpha.unsqueeze(2)).sum(dim=1)
        # combine states if pass_state is True
        if self.pass_state:
            for i in range(len(in_states)):
                in_states[i] = in_states[i].permute(1,0,2).contiguous().view(batch_size, -1) # switch to batch first
            if len(in_states) > 0:
                in_states = torch.cat(in_states, dim=1)
                in_states = self.combine_state(in_states)
                rnn_input = torch.cat([rnn_input, in_states], dim=1)
        # prepare rnn input
        prev_hidden_state = prev_hidden_state.view(batch_size, self.rnn_depth, self.rnn_dim)
        prev_hidden_state = prev_hidden_state.permute(1,0,2).contiguous()  # swap back
        rnn_input = rnn_input.unsqueeze(0)
        rnn_output, hidden_state = self.rnn(rnn_input, prev_hidden_state)
        # generate proposal if needed
        if gen_proposal:
            proposal_input = rnn_output[0]  # only has one time step, gets that one
            proposals = self.hidden_to_output(proposal_input)
            return proposals, hidden_state, out_alpha
        else:
            return None, hidden_state, out_alpha


class SentenceNetwork(nn.Module):
    def __init__(self, words, cnn_kernel_size=7, n_channels=50, attention_dim=16,
                 example_feature=None, rnn_dim=256, rnn_depth=1, output_dim=9,
                 device=None, compare_list=None, compare_weight=1, normalize_size=False,
                 no_attention=False, parse_type='default', pass_state=False):
        super(SentenceNetwork, self).__init__()
        self.rnn_depth = rnn_depth
        self.rnn_dim = rnn_dim
        self.device = device
        self.output_dim = output_dim
        self.compare_list = compare_list
        self.compare_weight = compare_weight
        self.normalize_size = normalize_size
        self.parse_type = parse_type

        self.current_words = []
        self.current_tree = []
        self.models = dict()
        self.hidden_states = []
        self.set_sample_embedding(n_channels, cnn_kernel_size, example_feature)
        for word, nargs in words.items():
            if compare_list is not None:
                skip = False
                for compare_words in compare_list:
                    if word in compare_words:
                        skip = True
                if skip:
                    continue
            if parse_type != 'default':
                if type(nargs) is not list:
                    nargs = [nargs]
                for narg in nargs:
                    model_idx = word+'_'+str(narg)
                    self.models[model_idx] = \
                            ComponentNetwork(rnn_dim=rnn_dim, rnn_depth=rnn_depth,
                                             smp_emb_dim=self.smp_emb_dim,
                                             attention_dim=attention_dim,
                                             nargs=narg,
                                             output_dim=output_dim,
                                             device=device,
                                             no_attention=no_attention,
                                             pass_state=pass_state)
                    self.add_module('ComponentNetwork_{}_'.format(model_idx), self.models[model_idx])
            else:
                self.models[word] = ComponentNetwork(rnn_dim=rnn_dim, rnn_depth=rnn_depth,
                                                     smp_emb_dim=self.smp_emb_dim,
                                                     attention_dim=attention_dim,
                                                     nargs=nargs,
                                                     output_dim=output_dim,
                                                     device=device,
                                                     no_attention=no_attention,
                                                     pass_state=pass_state)
                self.add_module('ComponentNetwork_{}_'.format(word), self.models[word])
        if compare_list is not None:
            # share weights for related words
            for compare_words in compare_list:
                if type(words[compare_words[0]]) is list:
                    nargs = words[compare_words[0]]
                else:
                    nargs = [words[compare_words[0]]]
                for narg in nargs:
                    model = ComponentNetwork(rnn_dim=rnn_dim, rnn_depth=rnn_depth,
                                             smp_emb_dim=self.smp_emb_dim,
                                             attention_dim=attention_dim,
                                             nargs=narg,
                                             output_dim=output_dim,
                                             device=device,
                                             related_words=compare_words,
                                             normalize_att=self.normalize_size,
                                             pass_state=pass_state)
                    for word in compare_words:
                        if parse_type == 'default':
                            self.models[word] = model
                        else:
                            self.models[word+'_'+str(narg)] = model
                    word_str = '_'.join(compare_words)
                    self.add_module('ComponentNetwork_{}_{}'.format(word_str, narg), model)
        # set up loss criterion for training
        self.loss_criterion = nn.NLLLoss(ignore_index=0, reduction='none')  # ignore the padding index

    def to(self, device):
        self.device = device
        super(SentenceNetwork, self).to(device)

    def set_sample_embedding(self, n_channels, cnn_kernel_size, example_feature):
        num_in_channels = example_feature.shape[-1]
        self.sample_layer = ConvolutionalNet(num_channels=num_in_channels,
                cnn_kernel_size=cnn_kernel_size, num_conv_channels=n_channels,
                dropout_probability=0.1, stride=1, input_shape=example_feature.shape)
        self.smp_emb_dim = self.sample_layer.cnn_output_shape[2]

    def update_words(self, tree):
        self.current_words = []
        def update_word_order(word, args):
            if len(args) > 0:
                for arg in args:
                    update_word_order(arg[0], arg[1:])
            self.current_words.append(word)
        def update_word_order_parse(node):
            for sub_tree in node.children:
                update_word_order_parse(sub_tree)
            self.current_words.append(node.name)
        self.current_tree = tree
        if self.parse_type == 'default':
            update_word_order(self.current_tree[0], self.current_tree[1:])
        else:
            update_word_order_parse(self.current_tree)
        self.prev_hidden_states = [None for _ in self.current_words]
        self.hidden_states = [None for _ in self.current_words]
        # store attention map or list of attention maps if it is comparative word
        self.att_maps = [None for _ in self.current_words]
        self.proposals = [None for _ in self.current_words]

    # TODO: remove this after using forward_child_parse for semantic parser as well
    def forward_child(self, smp_emb, word, args, gen_proposal=False, idx=0):
        # forward args
        alphas = []; states = []; actions = []
        new_idx = idx; old_idx = idx
        if len(args) > 0:
            for arg in args:
                new_word = arg[0]
                new_args = arg[1:]
                new_idx = self.forward_child(smp_emb, new_word, new_args, gen_proposal=gen_proposal, idx=new_idx)
                if type(self.att_maps[new_idx-1]) is list:
                    alphas.append(self.att_maps[new_idx-1][0])
                else:
                    alphas.append(self.att_maps[new_idx-1])
                states.append(self.hidden_states[new_idx-1])
                actions.append(self.proposals[new_idx-1])
        # forward current word
        prev_hidden_state = self.prev_hidden_states[new_idx]
        out_hidden_states = None
        self.proposals[new_idx], out_hidden_states, self.att_maps[new_idx] = \
                self.models[word].forward(smp_emb=smp_emb,
                                          prev_hidden_state=prev_hidden_state,
                                          in_att_maps=alphas, gen_proposal=gen_proposal,
                                          in_states=states, in_actions=actions, current_word=word)
        self.hidden_states[new_idx] = out_hidden_states
        new_idx += 1
        return new_idx

    def forward_child_parse(self, smp_emb, node, gen_proposal=False, idx=0):
        # forward args
        alphas = []; states = []; actions = []
        new_idx = idx; old_idx = idx
        for arg in node.children:
            new_idx = self.forward_child_parse(smp_emb, arg, gen_proposal=gen_proposal, idx=new_idx)
            if type(self.att_maps[new_idx-1]) is list:
                alphas.append(self.att_maps[new_idx-1][0])
            else:
                alphas.append(self.att_maps[new_idx-1])
            states.append(self.hidden_states[new_idx-1])
            actions.append(self.proposals[new_idx-1])
        # forward current word
        prev_hidden_state = self.prev_hidden_states[new_idx]
        self.proposals[new_idx], self.hidden_states[new_idx], self.att_maps[new_idx] = \
                self.models[node.name].forward(smp_emb=smp_emb,
                                              prev_hidden_state=prev_hidden_state,
                                              in_att_maps=alphas, gen_proposal=gen_proposal,
                                              in_states=states,
                                              in_actions=actions,
                                              current_word=node.name.split('_')[0])
        new_idx += 1
        return new_idx

    def forward(self, obs, prev_hidden_states=None):
        # get low level embeddings
        smp_emb = self.sample_layer(obs)
        if len(smp_emb.shape) == 4:
            smp_emb = smp_emb.permute(0, 2, 3, 1)
        # get words
        if prev_hidden_states is not None:
            self.prev_hidden_states = prev_hidden_states
        # make prediction for each word following the parse tree
        if self.parse_type == 'default':
            word = self.current_tree[0]
            args = self.current_tree[1:]
            self.forward_child(smp_emb, word, args, gen_proposal=True)
        else:
            self.forward_child_parse(smp_emb, self.current_tree, gen_proposal=True)
        return self.proposals, self.hidden_states

    def loss(self, situation_batch, target_batch, target_lengths, optimizer=None):
        batch_size = target_lengths.shape[0]
        words = [self.current_words[-1]] # only compute loss from root of the tree
        hidden_states = None
        losses = dict(); log_probs = dict()
        for word in words:
            log_probs[word] = torch.zeros(batch_size, device=self.device)
        # switch to seq-first instead of batch-first
        features = situation_batch.permute(1, 0, 2, 3, 4).to(self.device)
        target_batch = target_batch.permute(1, 0).to(self.device)
        target_lengths = target_lengths.to(self.device)
        # start trainging each time step
        for t in range(torch.max(target_lengths)):
            proposals, hidden_states = \
                self.forward(obs=features[t], prev_hidden_states=hidden_states)
            for i, word in enumerate(words):
                target_scores_2d = F.log_softmax(proposals[self.current_words.index(word)], dim=-1)
                log_probs[word] += self.loss_criterion(target_scores_2d, target_batch[t])
                if self.compare_list is not None:
                    # compute difference for comparison attention maps
                    for att_map in self.att_maps:
                        if type(att_map) is list:  # TODO: only works for two words now, make it generic
                            att_loss = torch.mul(att_map[0], att_map[1]).sum(dim=1) / \
                                    att_map[0].shape[1]
                            # add loss
                            log_probs[word] += att_loss * self.compare_weight
        # aggregate loss
        total_loss = 0
        for word in words:
            loss = log_probs[word] / target_lengths
            losses[word] = torch.sum(loss) / batch_size
            total_loss = total_loss + losses[word]
        # backpropagate if loss != 0
        n_words = len(np.nonzero([v for v in losses.values()])[0])
        if n_words > 0:
            total_loss /= float(n_words)
            if optimizer is not None:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        return True, total_loss, losses
