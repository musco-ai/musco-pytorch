import copy
import numpy as np
import os
import torch
import torchvision

import dataloaders

from absl import flags
from absl import app

from model_utils import get_layer_names
from tensor_compression import get_compressed_model, finetune
from tensor_compression.train_validate import validate, rm_checkpoints

FLAGS = flags.FLAGS

flags.DEFINE_string("model", None, "What model to load. Available vgg16 or resnet50.")
flags.DEFINE_string("model_weights", None, "Path to VGG pre-trained weights.")

flags.DEFINE_string("dataset", None, "What dataset to use. Available imagnet, mnist, cifar10.")
flags.DEFINE_string("data_dir", None, "Where data is located.")
flags.DEFINE_boolean("simple_normalize", False, "Normalize dataset or not.")
flags.DEFINE_integer("batch_size", 32, "Set a batch size.")
flags.DEFINE_integer("data_loader_workers", 4, "How many threads read data.")

flags.DEFINE_integer("compress_iters", 4, "How many times compress a neural network.")
flags.DEFINE_integer("ft_epochs", 1, "Number of epochs for fine-tuning agter each compression.")
flags.DEFINE_integer("conv_split", 3, "")
flags.DEFINE_integer("fc_split", 1, "")
flags.DEFINE_integer("x_factor", 0, "")
flags.DEFINE_float("weaken_factor", 0.9, "Weaken factor.")
flags.DEFINE_string("rank_selection_alg", "vbmf", "Algorithm for rang calculation.")
flags.DEFINE_string("decompos_conv_alg", "tucker2", "Decomposition algorithm. Available tucker2 and cp3")
flags.DEFINE_string("decompos_svd_alg", "svd", "Decomposition algorithm. Available tucker2 and cp3")
flags.DEFINE_boolean("resnet_split", False, "Resnet split")
flags.DEFINE_boolean("compress_fc", False, "Compress fc or nor")
flags.DEFINE_boolean("adv_finetune", False, "Use default fine-tunning or advanced")

flags.DEFINE_integer("batches_per_train", 10 ** 7, "")
flags.DEFINE_integer("batches_per_val", 10 ** 7, "")
flags.DEFINE_integer("patience", 3, "")
flags.DEFINE_boolean("validate_before_ft", False, "Evaluate or not validation before a fine tunning.")

flags.DEFINE_string("gpu_number", "-1", "GPU's id. Bu default, GPU computation is disabled.")
flags.DEFINE_string("save_dir", None, "Where to store the result.")


def load_model(model_name, model_weights_path=None):
    if model_name == "vgg16":
        model = torchvision.models.vgg16()
        if model_weights_path is not None:
            print("Loading weights from", model_weights_path)
            model.load_state_dict(torch.load(model_weights_path))
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
    return model


def get_ranks(model_name, layer_names, fc_layer_mask, conv_layer_mask, x_factor):
    if model_name == 'vgg16':
        ranks_conv = [None] + [-x_factor] * (len(layer_names[conv_layer_mask]) - 1)
        ranks_fc = [-x_factor] * (len(layer_names[fc_layer_mask]))
    elif model_name.startswith('resnet'):
        ranks_conv = [None if not name.endswith('conv2') else -x_factor
                      for name in layer_names[conv_layer_mask]]
        ranks_fc = [-x_factor] * (len(layer_names[fc_layer_mask]))

    ranks = np.array([None] * len(layer_names))
    ranks[conv_layer_mask] = ranks_conv
    if FLAGS.compress_fc:
        ranks[fc_layer_mask] = ranks_fc

    return ranks


def get_decompositions(decomposition_fc, decomposition_conv, layer_names, fc_layer_mask, conv_layer_mask):
    decompositions = np.array([None] * len(layer_names))
    decompositions[conv_layer_mask] = decomposition_conv
    decompositions[fc_layer_mask] = decomposition_fc
    return decompositions


def make_catalog(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_save_and_log_dirs():
    if FLAGS.rank_selection_alg == 'vbmf':
        rank_selection_suffix = "/wf:{}".format(FLAGS.weaken_factor)
    elif FLAGS.rank_selection_alg == 'nx':
        rank_selection_suffix = "/{}x".format(FLAGS.x_factor)
    save_dir = "{}/models_finetuned/{}/{}/rank_selection:{}{}/layer_groups:{}".format(FLAGS.save_dir,
                                                                                      FLAGS.model,
                                                                                      FLAGS.decompos_conv_alg,
                                                                                      FLAGS.rank_selection_alg,
                                                                                      rank_selection_suffix,
                                                                                      FLAGS.fc_split + FLAGS.conv_split)
    loggs_dir = "{}/loggs".format(save_dir)

    make_catalog(save_dir)
    make_catalog(loggs_dir)

    return save_dir, loggs_dir


def split_resnet_layers_by_blocks(lnames):
    starts = ['conv1'] + ['layer{}'.format(i) for i in range(1, 5)]

    start_idx = 0
    blocks_idxs = []
    layer_names_by_blocks = []

    for s in starts:
        curr_block = [l for l in lnames if l.startswith(s)]
        layer_names_by_blocks.append(curr_block)

        blocks_idxs.append(np.arange(start_idx, start_idx + len(curr_block)))
        start_idx += len(curr_block)

    return blocks_idxs


def get_split(layer_names, conv_layer_mask, fc_layer_mask, fc_split, conv_split, resnet_split):
    n_layers = len(layer_names)
    if FLAGS.model.startswith('resnet') and resnet_split:
        split_tuples = split_resnet_layers_by_blocks(layer_names[conv_layer_mask])[::-1]
    else:
        split_tuples = np.array_split(np.arange(n_layers)[conv_layer_mask], conv_split)[::-1]

    split_tuples.append(np.array_split(np.arange(n_layers)[fc_layer_mask], fc_split))

    return split_tuples


def main(_):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_number

    if FLAGS.gpu_number == "-1":
        device = "cpu"
    else:
        device = "cuda"

    save_dir, loggs_dir = make_save_and_log_dirs()
    model = load_model(FLAGS.model, FLAGS.model_weights)
    loaders = dataloaders.get_loader(FLAGS.batch_size,
                                     FLAGS.dataset,
                                     FLAGS.data_dir,
                                     num_workers=FLAGS.data_loader_workers,
                                     simple_normalize=FLAGS.simple_normalize)

    layer_names, conv_layer_mask = get_layer_names(model)
    fc_layer_mask = (1 - conv_layer_mask).astype(bool)

    ranks = get_ranks(FLAGS.model, layer_names, fc_layer_mask, conv_layer_mask, FLAGS.x_factor)
    decom = get_decompositions(FLAGS.decompos_svd_alg, FLAGS.decompos_conv_alg,
                               layer_names, fc_layer_mask, conv_layer_mask)

    split_tuples = get_split(layer_names, conv_layer_mask, fc_layer_mask,
                             FLAGS.fc_split, FLAGS.conv_split, FLAGS.resnet_split)
    compressed_model = copy.deepcopy(model)

    for glob_iter in range(FLAGS.compress_iters):
        for local_iter, tupl in enumerate(split_tuples):

            lname, rank, decomposition = layer_names[tupl], ranks[tupl], decom[tupl]
            if isinstance(tupl[0], np.ndarray):
                print("Uncompressed layer:", lname)
                continue
            compressed_model = get_compressed_model(compressed_model,
                                                    ranks=rank,
                                                    layer_names=lname,
                                                    decompositions=decomposition,
                                                    vbmf_weaken_factor=FLAGS.weaken_factor
                                                    if FLAGS.rank_selection_alg == 'vbmf' else None)

            suffix = "_iter:{}-{}".format(glob_iter, local_iter)
            if FLAGS.validate_before_ft and not FLAGS.adv_finetune:
                validate(loaders['val'],
                         compressed_model,
                         device=device,
                         iters=FLAGS.batches_per_val,
                         suffix=suffix,
                         prefix="Before_fine",
                         loggs_dir=loggs_dir)

            if FLAGS.adv_finetune:
                finetune.fine_tune_adv(compressed_model,
                                                           loaders,
                                                           device=device,
                                                           save_dir=save_dir,
                                                           train_iters=FLAGS.batches_per_train,
                                                           val_iters=FLAGS.batches_per_val,
                                                           ft_epochs=FLAGS.ft_epochs,
                                                           suffix=suffix,
                                                           loggs_dir=loggs_dir)
            else:
                finetune.fine_tune(compressed_model,
                                   loaders,
                                   device=device,
                                   save_dir=save_dir,
                                   batches_per_train=FLAGS.batches_per_train,
                                   batches_per_val=FLAGS.batches_per_val,
                                   ft_epochs=FLAGS.ft_epochs,
                                   suffix=suffix,
                                   loggs_dir=loggs_dir,
                                   patience=FLAGS.patience)
            try:
                compressed_model = torch.load('{}/model_best{}.pth.tar'.format(save_dir, suffix))
            except:
                rm_checkpoints(save_dir)
                print('Stop compression')
                return

    rm_checkpoints(save_dir)


if __name__ == "__main__":
    app.run(main)
