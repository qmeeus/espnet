import numpy as np
import logging
from chainer.training.triggers import EarlyStoppingTrigger


def check_early_stop(trainer, epochs):
    """Checks if the training was stopped by an early stopping trigger and warns the user if it's the case

    :param trainer: The trainer used for training
    :param epochs: The maximum number of epochs
    """
    end_epoch = trainer.updater.get_iterator('main').epoch
    if end_epoch < (epochs - 1):
        logging.warning("Hit early stop at epoch " + str(
            end_epoch) + "\nYou can change the patience or set it to 0 to run all epochs")


def set_early_stop(trainer, args, is_lm=False):
    """Sets the early stop trigger given the program arguments

    :param trainer: The trainer used for training
    :param args: The program arguments
    :param is_lm: If the trainer is for a LM (epoch instead of epochs)
    """
    patience = args.patience
    criterion = args.early_stop_criterion
    epochs = args.epoch if is_lm else args.epochs
    mode = 'max' if 'acc' in criterion else 'min'
    if patience > 0:
        trainer.stop_trigger = EarlyStoppingTrigger(
            monitor=criterion,
            mode=mode,
            patients=patience,
            max_trigger=(epochs, 'epoch')
        )


def make_logistic_scheduler(x0=0.1, xN=0.9, rho=20, delay=10, N=15):
    """Create a logistic scheduler as a function of the current step

    :param x0: initial value
    :param xN: end value
    :param rho: the rate of growth/decay
    :param N: number of expected steps
    """
    def scheduler(n):
        return x0 + ((xN - x0) / ((1 + delay * np.exp(rho * (.5 - n / N)))))

    return scheduler
