import logging
import torch
import operator
from itertools import product
from chainer import training
from chainer.training import extensions, triggers
from tensorboardX import SummaryWriter

from espnet.asr.pytorch_backend.evaluator import CustomEvaluator
from espnet.asr.pytorch_backend.updater import CustomUpdater
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import adadelta_eps_decay, alpha_scheduler
from espnet.utils.training.tensorboard_logger import TensorboardLogger


class CustomTrainer:


    def __init__(self, args, model, optimizer, training_set, validation_set,
                 converter, device, valid_json=None, load_cv=None):

        self.epochs = args.epochs
        self.outdir = args.outdir
        self.resume = args.resume
        self.device = device

        self.model = model
        self.optimizer = optimizer
        self.training_set = training_set
        self.validation_set = validation_set

        self.converter = converter

        self.updater = CustomUpdater(
            model, args.grad_clip, training_set, self.optimizer,
            device, args.ngpu, args.grad_noise, args.accum_grad,
            use_apex=args.use_apex
        )

        self.evaluator = CustomEvaluator(
            model, validation_set, model.reporter, device, args.ngpu
        )

        self.trainer = training.Trainer(
            self.updater, (args.epochs, 'epoch'), out=args.outdir
        )

        self.configure(args, valid_json, load_cv)

        # Resume from a snapshot
        if args.resume:
            logging.info('resumed from %s' % args.resume)
            torch_resume(args.resume, self.trainer)

        set_early_stop(self.trainer, args)

    def add_extension(self, extension, trigger=None):
        self.trainer.extend(extension, trigger=trigger)

    def run(self):
        self.trainer.run()
        check_early_stop(self.trainer, self.epochs)

    def configure(self, args, valid_json=None, load_cv=None):
        if args.use_sortagrad:
            self.add_extension(
                ShufflingEnabler([self.training_set]),
                trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, 'epoch')
            )

        self.add_extension(self.evaluator, trigger=(
            (args.save_interval_iters, 'iteration')
            if args.save_interval_iters > 0
            else None
        ))

        # Save attention weight each epoch
        if args.num_save_attention > 0 and args.mtlalpha != 1.0:
            assert valid_json is not None
            assert load_cv is not None

            data = sorted(
                list(valid_json.items())[:args.num_save_attention],
                key=lambda x: int(x[1]['input'][0]['shape'][1]),
                reverse=True
            )

            if hasattr(self.model, "module"):
                att_vis_fn = self.model.module.calculate_all_attentions
                plot_class = self.model.module.attention_plot_class

            else:
                att_vis_fn = self.model.calculate_all_attentions
                plot_class = self.model.attention_plot_class

            att_reporter = plot_class(
                att_vis_fn, data, args.outdir + "/att_ws",
                converter=self.converter, transform=load_cv, device=self.device
            )

            self.add_extension(att_reporter, trigger=(1, 'epoch'))

        else:
            att_reporter = None

        format_metric = "/".join

        prefixes = ['main', 'validation/main']

        losses, metrics = [
            list(map("/".join, product(prefixes, names)))
            for names in (self.model.LOSS_NAMES, self.model.METRIC_NAMES)
        ]

        report_keys = ['epoch', 'iteration', *losses, *metrics, 'elasped_time']

        # report_keys = [
        #         'epoch', 'iteration', 'main/loss', 'main/loss_ctc', 'main/loss_att',
        #         'validation/main/loss', 'validation/main/loss_ctc', 'validation/main/loss_att',
        #         'main/accuracy', 'validation/main/accuracy', 'main/cer_ctc', 'validation/main/cer_ctc',
        #         'elapsed_time'
        #     ]


        loss_plot_keys = losses
        # loss_plot_keys = [
        #     'main/loss', 'validation/main/loss', 'main/loss_ctc', 'validation/main/loss_ctc',
        #     'main/loss_att', 'validation/main/loss_att'
        # ]

        acc_plot_keys, cer_plot_keys = (
            [metric for metric in metrics if mname in metric]
            for mname in ("accuracy", "cer_ctc")
        )

        # acc_plot_keys, cer_plot_keys = (
        #     [f'main/{metric}', f'validation/main/{metric}']
        #     for metric in ('accuracy', 'cer_ctc')
        # )

        # Resume from a snapshot
        # if args.resume:
        #     logging.info('resumed from %s' % args.resume)
        #     torch_resume(args.resume, trainer)
        #     report_keys_loss_ctc = (
        #         ['main/loss_ctc{}'.format(i + 1) for i in range(self.model.num_encs)]
        #         + ['validation/main/loss_ctc{}'.format(i + 1) for i in range(self.model.num_encs)]
        #     )

        #     report_keys_cer_ctc = (
        #         ['main/cer_ctc{}'.format(i + 1) for i in range(self.model.num_encs)]
        #         + ['validation/main/cer_ctc{}'.format(i + 1) for i in range(self.model.num_encs)]
        #     )

        #     report_keys.extend(report_keys_cer_ctc + report_keys_loss_ctc)
        #     loss_plot_keys.extend(report_keys_loss_ctc)
        #     cer_plot_keys.extend(report_keys_cer_ctc)

        # Make a plot for training and validation values
        for keys, metric in zip([loss_plot_keys, acc_plot_keys, cer_plot_keys], ['loss', 'accuracy', 'cer']):
            if keys:
                self.add_extension(
                    extensions.PlotReport(keys, 'epoch', file_name=f'{metric}.png')
                )

        self.add_extension(
            extensions.PlotReport(
                ['grad_norm'],
                trigger=(args.report_interval_iters, 'iteration'),
                file_name=f'grad_norm.png'
            )
        )

        # Write a log of evaluation statistics for each epoch
        self.add_extension(
            extensions.LogReport(trigger=(args.report_interval_iters, 'iteration'))
        )

        # Model checkpoints
        self.add_extension(
            snapshot_object(self.model, 'model.loss.best'),
            trigger=triggers.MinValueTrigger('validation/main/loss')
        )


        if args.mtl_mode != 'ctc' and acc_plot_keys:
            self.add_extension(
                snapshot_object(self.model, 'model.acc.best'),
                trigger=triggers.MaxValueTrigger('validation/main/accuracy')
            )

        # save snapshot which contains model and optimizer states
        if args.save_interval_iters > 0:
            self.add_extension(
                torch_snapshot(filename='snapshot.iter.{.updater.iteration}'),
                trigger=(args.save_interval_iters, 'iteration')
            )
        else:
            self.add_extension(torch_snapshot(), trigger=(1, 'epoch'))


        # epsilon decay in the optimizer
        if args.opt == 'adadelta':

            if args.criterion == 'accuracy' and args.mtl_mode != 'ctc':
                trigger_best = CompareValueTrigger('validation/main/accuracy', operator.gt)
                snapshot_name = 'model.acc.best'

            else:  # args.criterion == 'loss'
                trigger_best = CompareValueTrigger('validation/main/loss', operator.lt)
                snapshot_name = 'model.loss.best'

            self.add_extension(
                restore_snapshot(self.model, f"{args.outdir}/{snapshot_name}", load_fn=torch_load),
                trigger=trigger_best
            )

            self.add_extension(
                adadelta_eps_decay(args.eps_decay),
                trigger=trigger_best
            )

            def get_eps_value(trainer):
                return trainer.updater.get_optimizer('main').param_groups[0]["eps"]

            self.add_extension(
                extensions.observe_value('eps', get_eps_value),
                trigger=(args.report_interval_iters, 'iteration')
            )

            report_keys.append('eps')

        if getattr(args, "alpha_scheduler", False):
            # alpha_scheduler = (initial_value, end_value, epsilon, max_steps)
            self.add_extension(
                alpha_scheduler(*args.alpha_scheduler),
                trigger=(1, 'epoch')
            )

            def get_alpha_value(trainer):
                return self.model.mtlalpha

            self.add_extension(
                extensions.observe_value('alpha', get_alpha_value),
                trigger=(1, 'epoch')
            )

            report_keys.append('alpha')

        if args.report_cer:
            report_keys.append('validation/main/cer')

        if args.report_wer:
            report_keys.append('validation/main/wer')

        self.add_extension(
            extensions.PrintReport(report_keys),
            trigger=(args.report_interval_iters, 'iteration')
        )

        self.add_extension(
            extensions.ProgressBar(update_interval=args.report_interval_iters),
            trigger=(args.report_interval_iters, 'iteration')
        )

        if args.tensorboard_dir:
            self.add_extension(
                TensorboardLogger(SummaryWriter(args.tensorboard_dir), att_reporter),
                trigger=(args.report_interval_iters, "iteration")
            )
