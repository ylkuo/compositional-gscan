import tqdm
import sys
import logging
import time
import torchvision


class DummyTqdmFile(object):
    """ Dummy file-like that will write to tqdm
    https://github.com/tqdm/tqdm/issues/313
    """
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.tqdm.write(x, file=self.file, end='')

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()


def get_logger(name, level=logging.INFO):
    # temp fix to avoid PIL logging polution
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    # our logger
    logging.basicConfig(level=level, stream=DummyTqdmFile(sys.stderr))
    log = logging.getLogger(name)
    return log


def log_model_params(model, writer, name, n_update):
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/').replace(' ', '_')
        writer.add_histogram(name + '_' + tag,
                             value.data.cpu().numpy(), n_update)
        if value.grad is not None:
            writer.add_histogram(name + '_' + tag + '/grad',
                                 value.grad.data.cpu().numpy(), n_update)


def make_filter_image(layer, use_color=True, scale_each=True):
    """Build an image of the weights of the filters in a given convolutional layer."""
    weights = layer.weight.data.to("cpu")
    if not use_color:
        n_input_channels = weights.size()[1]
        weights = weights.view([weights.size()[0], 1, weights.size()[1]*weights.size()[2], weights.size()[3]])
    img = torchvision.utils.make_grid(weights, normalize=True, scale_each=scale_each)
    return img


def log_conv_filters(model, writer, n_update):
    writer.add_image('image/conv1', make_filter_image(model.sample_layer.conv_1, use_color=False), n_update)
    writer.add_image('image/conv2', make_filter_image(model.sample_layer.conv_2, use_color=False), n_update)
    writer.add_image('image/conv3', make_filter_image(model.sample_layer.conv_3, use_color=False), n_update)


if __name__ == "__main__":
    log = get_logger(__name__, level=logging.DEBUG)
    log.info("loop")

    for x in tqdm.trange(1, 16):
        time.sleep(.2)
        if not x % 5:
            log.debug("in loop %d\n" % x)

    log.info("done\n")

