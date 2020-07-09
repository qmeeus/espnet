"""Finetuning methods."""
import logging
import os
import torch
from collections import OrderedDict

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.nets.asr_interface import ASRInterface
from espnet.utils.dynamic_import import dynamic_import


def transfer_verification(model_state_dict, partial_state_dict, modules):
    """Verify tuples (key, shape) for input model modules match specified modules.

    Args:
        model_state_dict (OrderedDict): the initial model state_dict
        partial_state_dict (OrderedDict): the trained model state_dict
        modules (list): specified module list for transfer

    Return:
        (boolean): allow transfer

    """
    partial_modules = []
    for key_p, value_p in partial_state_dict.items():
        if any(key_p.startswith(m) for m in modules):
            if value_p.shape == model_state_dict[key_p].shape:
                partial_modules += [(key_p, value_p.shape)]
    return len(partial_modules) > 0


def get_partial_asr_mt_state_dict(model_state_dict, modules):
    """Create state_dict with specified modules matching input model modules.

    Args:
        model_state_dict (OrderedDict): trained model state_dict
        modules (list): specified module list for transfer

    Return:
        new_state_dict (OrderedDict): the updated state_dict

    """
    new_state_dict = OrderedDict()

    for key, value in model_state_dict.items():
        if any(key.startswith(m) for m in modules):
            new_state_dict[key] = value

    return new_state_dict


def filter_modules(model_state_dict, modules):
    """Filter non-matched modules in module_state_dict.

    Args:
        model_state_dict (OrderedDict): trained model state_dict
        modules (list): specified module list for transfer

    Return:
        new_mods (list): the update module list

    """
    new_mods = []
    incorrect_mods = []

    mods_model = list(model_state_dict.keys())
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]
        else:
            incorrect_mods += [mod]

    if incorrect_mods:
        logging.warning("module(s) %s don\'t match or (partially match) "
                        "available modules in model.", incorrect_mods)
        logging.warning('for information, the existing modules in model are:')
        logging.warning('%s', mods_model)

    return new_mods


def load_trained_model(model_path):
    """Load the trained model for recognition.

    Args:
        model_path (str): Path to model.***.best

    """
    idim, odim, train_args = get_model_conf(
        model_path, os.path.join(os.path.dirname(model_path), 'model.json'))

    logging.warning('reading model parameters from ' + model_path)

    if hasattr(train_args, "model_class"):
        model_class = train_args.model_class
    else:
        model_class = "espnet.nets.pytorch_backend.e2e_asr:E2E"
    model_class = dynamic_import(model_class)
    model = model_class(idim, odim, train_args)

    torch_load(model_path, model)

    return model, train_args


def get_trained_model_state_dict(model_path):
    """Extract the trained model state dict for pre-initialization.

    Args:
        model_path (str): Path to model.***.best

    Return:
        model.state_dict() (OrderedDict): the loaded model state_dict
        (str): Type of model. Either ASR/MT or LM.

    """

    conf_path = os.path.join(os.path.dirname(model_path), 'model.json')
    if 'rnnlm' in model_path:
        logging.warning('reading model parameters from %s', model_path)

        return torch.load(model_path), 'lm'

    idim, odim, args = get_model_conf(model_path, conf_path)

    logging.warning('reading model parameters from ' + model_path)

    if hasattr(args, "model_class"):
        model_class = args.model_class
    else:
        model_class = "espnet.nets.pytorch_backend.e2e_asr:E2E"

    model_class = dynamic_import(model_class)
    model = model_class(idim, odim, args)
    torch_load(model_path, model)
    # assert isinstance(model, ASRInterface)

    return model.state_dict(), 'asr-mt'


def load_trained_modules(idim, odim, args, interface=ASRInterface):
    """Load model encoder or/and decoder modules with ESPNET pre-trained model(s).

    Args:
        idim (int): initial input dimension.
        odim (int): initial output dimension.
        args (Namespace): The initial model arguments.
        interface (Interface): ASRInterface or STInterface

    Return:
        model (torch.nn.Module): The model with pretrained modules.

    """
    enc_model_path = args.enc_init
    dec_model_path = args.dec_init
    enc_modules = args.enc_init_mods
    dec_modules = args.dec_init_mods

    Model = dynamic_import(args.model_class)
    model = Model(idim, odim, args)

    state_dict = model.state_dict()

    logging.warning('model(s) found for pre-initialization')
    for model_path, modules in [(enc_model_path, enc_modules),
                                (dec_model_path, dec_modules)]:
        if model_path is not None:
            if os.path.isfile(model_path):
                model_state_dict, mode = get_trained_model_state_dict(model_path)

                modules = filter_modules(model_state_dict, modules)

                partial_state_dict = get_partial_asr_mt_state_dict(model_state_dict, modules)

                if partial_state_dict:
                    if transfer_verification(state_dict, partial_state_dict, modules):
                        logging.warning('loading %s from model: %s', modules, model_path)
                        for k in partial_state_dict.keys():
                            logging.warning('override %s' % k)
                        state_dict.update(partial_state_dict)
                    else:
                        logging.warning('modules %s in model %s don\'t match your training config',
                                        modules, model_path)
            else:
                logging.warning('model was not found : %s', model_path)

    model.load_state_dict(state_dict)

    return model
