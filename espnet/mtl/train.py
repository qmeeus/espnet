import os
import torch
import logging
import warnings

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from espnet.mtl.model import ASRModel
from espnet.data.dataset import ASRDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.nets.pytorch_backend_.e2e_asr import E2E


warnings.filterwarnings("ignore", category=RuntimeWarning) 


def train(args):

    # SETUP AND PRELIMINARY OPERATIONS
    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    training_set, validation_set = (
        ASRDataset.from_json(getattr(args, json_file))
        for json_file in ('train_json', 'valid_json')
    )

    # specify attention, CTC, hybrid mode
    if args.mtlalpha == 1.0:
        args.mtl_mode = 'ctc'
        logging.info('Pure CTC mode')
    elif args.mtlalpha == 0.0:
        args.mtl_mode = 'att'
        logging.info('Pure attention mode')
    else:
        args.mtl_mode = 'mtl'
        logging.info('Multitask learning mode')

    datasets = {"train": training_set, "validation": validation_set}
    input_dim = training_set.input_dim[0]
    output_dim = training_set.output_dim[0]
    
    # HACK
    args.shuffle = True
    args.batch_size = 64
    args.num_workers = 10
    args.optimizer_name = "adadelta"
    args.optimizer_config = {
        "rho": .95, 
        "eps": args.eps,
        "weight_decay": args.weight_decay
    }

    network = E2E(input_dim, output_dim, args)
    model = ASRModel(network, datasets, args)

    for line in str(model).split("\n"):
        logging.info(line) 
    
    logger = TensorBoardLogger(save_dir=args.tensorboard_dir)

    trainer = Trainer(
        max_epochs=1,
        gpus=args.gpus, 
        log_gpu_memory='all', 
        # overfit_pct=.01, 
        profiler=True,
        logger=logger
    )

    # import ipdb
    # with ipdb.launch_ipdb_on_exception():
    trainer.fit(model)
