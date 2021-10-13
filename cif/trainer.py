import os
from contextlib import suppress
from collections import Counter
import sys

import numpy as np

import torch
import torch.nn.utils

from ignite.engine import Events, Engine
from ignite.exceptions import NotComputableError
from ignite.metrics import RunningAverage, Metric, Loss
from ignite.handlers import TerminateOnNan
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, GradsScalarHandler


class AverageMetric(Metric):
    # XXX: This is not ideal, since we are overriding a protected attribute in Metric.
    # However, as of ignite v0.3.0, this is necessary to allow us to return a
    # map from the Engines we attach this to. (In particular, note that e.g.
    # `Trainer._train_batch` should return a map of the form `{"metrics": METRICS_MAP}`.)
    _required_output_keys = ["metrics"]

    def reset(self):
        self._sums = Counter()
        self._num_examples = Counter()

    def update(self, output):
        metrics, = output
        for k, v in metrics.items():
            self._sums[k] += torch.sum(v)
            self._num_examples[k] += torch.numel(v)

    def compute(self):
        return {k: v / self._num_examples[k] for k, v in self._sums.items()}

    def completed(self, engine):
        engine.state.metrics = {**engine.state.metrics, **self.compute()}

    def attach(self, engine):
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed)


class Trainer:
    _STEPS_PER_LOSS_WRITE = 10
    _STEPS_PER_GRAD_WRITE = 10
    _STEPS_PER_LR_WRITE = 10

    def __init__(
            self,

            module,
            device,

            train_metrics,
            train_loader,
            optimizers,
            lr_schedulers,
            max_epochs,
            max_grad_norm,

            test_metrics,
            test_loader,
            epochs_per_test,

            early_stopping,
            early_stopping_start_epoch,
            valid_loss,
            valid_loader,
            max_bad_valid_epochs,
            valid_frequency,

            visualizer,

            writer,
            should_checkpoint_latest,
            should_checkpoint_best_valid,

            non_square,
            likelihood_introduction_epoch,

            fid_function,

            only_testing,

            ood_test,
            ood_with_train,
            other_dataset
    ):
        self._module = module

        self._device = device

        self._train_metrics = train_metrics
        self._train_loader = train_loader
        self._optimizers = optimizers
        self._lr_schedulers = lr_schedulers
        self._num_optimizers = len(self._optimizers)
        assert self._num_optimizers == len(self._lr_schedulers)

        self._max_epochs = max_epochs
        self._max_grad_norm = max_grad_norm

        self._test_metrics = test_metrics
        self._test_loader = test_loader
        self._epochs_per_test = epochs_per_test

        self._valid_loss = valid_loss
        self._valid_loader = valid_loader
        self._max_bad_valid_epochs = max_bad_valid_epochs
        self._best_valid_loss = float("inf")
        self._num_bad_valid_epochs = 0
        self._valid_frequency = valid_frequency

        self._visualizer = visualizer

        self._writer = writer
        self._should_checkpoint_best_valid = should_checkpoint_best_valid

        self._non_square = non_square
        self._likelihood_introduction_epoch = likelihood_introduction_epoch
        self._fid_function = fid_function

        self._ood_test = ood_test
        self._ood_with_train = ood_with_train
        self._other_dataset = other_dataset

        ### Training

        self._trainer = Engine(self._train_batch)

        AverageMetric().attach(self._trainer)
        ProgressBar(persist=True).attach(self._trainer, ["loss"])

        self._trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        self._trainer.add_event_handler(Events.ITERATION_COMPLETED, self._log_training_info)

        ### Validation

        if early_stopping:
            self._early_stopping_start_epoch = early_stopping_start_epoch
            self._validator = Engine(self._validate_batch)

            AverageMetric().attach(self._validator)
            ProgressBar(persist=False, desc="Validating").attach(self._validator)

            if non_square:
                validate_fn = self._validate
            else:
                validate_fn = torch.no_grad()(self._validate)
            self._trainer.add_event_handler(Events.EPOCH_COMPLETED, validate_fn)

        ### Testing

        self._tester = Engine(self._test_batch)

        AverageMetric().attach(self._tester)
        ProgressBar(persist=False, desc="Testing").attach(self._tester)

        if non_square:
            test_and_log_fn = self._test_and_log
        else:
            test_and_log_fn = torch.no_grad()(self._test_and_log)
        self._trainer.add_event_handler(Events.EPOCH_COMPLETED, test_and_log_fn)

        ### Checkpointing

        if should_checkpoint_latest:
            self._trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: self._save_checkpoint("latest"))

        checkpoint_1 = "best_valid" if only_testing else "latest"
        checkpoint_2 = "latest" if only_testing else "best_valid"

        try:
            self._load_checkpoint(checkpoint_1)
        except FileNotFoundError:
            print(f"Did not find `{checkpoint_1}' checkpoint.", file=sys.stderr)

            try:
                self._load_checkpoint(checkpoint_2)
            except FileNotFoundError:
                print(f"Did not find `{checkpoint_2}' checkpoint.", file=sys.stderr)

    def train(self):
        self._trainer.run(data=self._train_loader, max_epochs=self._max_epochs)

    def _train_batch(self, engine, batch):
        # HACK: If we are before likelihood introduction epoch *and* have more
        #       than one objective, we just skip one of the objectives.
        #       This is done to incorporate likelihood warmup into the alternating
        #       optimization.
        if (
            engine.state.epoch < self._likelihood_introduction_epoch
            and
            not engine.state.epoch % self._num_optimizers == 0
        ):
            return {"metrics": {"loss": torch.tensor(0.)}}

        self._module.train()

        x, _ = batch # TODO: Potentially pass y also for genericity
        x = x.to(self._device)

        opt = self._optimizers[engine.state.epoch % self._num_optimizers]
        lr_scheduler = self._lr_schedulers[engine.state.epoch % self._num_optimizers]

        opt.zero_grad()

        train_metrics = self._train_metrics(self._module, x, engine.state.epoch)

        loss = train_metrics["loss"]
        loss.backward()

        if self._max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._module.parameters(), self._max_grad_norm)

        opt.step()
        lr_scheduler.step()

        return {"metrics": train_metrics}

    def test(self):
        self._module.eval()

        if self._ood_test:
            loader = self._train_loader if self._ood_with_train else self._test_loader
            self._buffer_dict = {}
            self._buffer_ind = 0
            self._buffer_length = loader.dataset.x.shape[0]

            metrics = self._tester.run(data=loader).metrics

            if "likelihood" in self._buffer_dict:
                likelihoods = self._buffer_dict["likelihood"]
                reconstruction_errors = self._buffer_dict["reconstruction-error"]
            else:
                likelihoods = self._buffer_dict["log-prob"]
                reconstruction_errors = torch.zeros_like(likelihoods)

            ood_info_array = torch.cat((likelihoods, reconstruction_errors), dim=1)

            train_test = "train" if self._ood_with_train else "test"
            in_or_out_of_sample = "out" if self._other_dataset else "in"
            write_str = f"ood_metrics_{train_test}_{in_or_out_of_sample}"

            self._writer.write_numpy(write_str, ood_info_array.detach().cpu().numpy())

            return metrics

        if self._fid_function:
            fid = torch.tensor(self._fid_function(self._module)).to(self._device)
            return {
                **self._tester.run(data=self._test_loader).metrics,
                "fid": fid
            }

        return self._tester.run(data=self._test_loader).metrics

    def _test_and_log(self, engine):
        epoch = engine.state.epoch
        if (epoch - 1) % self._epochs_per_test == 0: # Test after first epoch
            for k, v in self.test().items():
                self._writer.write_scalar(f"test/{k}", v, global_step=engine.state.epoch)

                if not torch.isfinite(v):
                    self._save_checkpoint(tag="nan_during_test")

            self._visualizer.visualize(self._module, epoch)

    def _test_batch(self, engine, batch):
        x, _ = batch
        x = x.to(self._device)
        metrics = self._test_metrics(self._module, x)

        if not self._ood_test:
            return {"metrics": metrics}

        for k, v in metrics.items():
            if k not in self._buffer_dict:
                self._buffer_dict[k] = torch.empty((self._buffer_length, 1))

            end_ind = self._buffer_ind + v.shape[0]
            self._buffer_dict[k][self._buffer_ind:end_ind] = v.detach()

        self._buffer_ind = end_ind
        return {"metrics": metrics}

    def _validate(self, engine):
        if engine.state.epoch < self._early_stopping_start_epoch:
            return
        if not engine.state.epoch % self._valid_frequency == 0:
            return

        self._module.eval()

        # HACK: Don't run self._validator if there's an FID function
        if self._fid_function:
            valid_loss = torch.tensor(self._fid_function(self._module)).to(self._device)
        else:
            state = self._validator.run(data=self._valid_loader)
            valid_loss = state.metrics["loss"]

        if valid_loss < self._best_valid_loss:
            print(f"Best validation loss {valid_loss} after epoch {engine.state.epoch}")
            self._num_bad_valid_epochs = 0
            self._best_valid_loss = valid_loss

            if self._should_checkpoint_best_valid:
                self._save_checkpoint(tag="best_valid")

        else:
            if not torch.isfinite(valid_loss):
                self._save_checkpoint(tag="nan_during_validation")

            self._num_bad_valid_epochs += 1

            # We do this manually (i.e. don't use Ignite's early stopping) to permit
            # saving/resuming more easily
            if self._num_bad_valid_epochs > self._max_bad_valid_epochs:
                print(
                    f"No validation improvement after {self._num_bad_valid_epochs} epochs. Terminating."
                )
                self._trainer.terminate()

    def _validate_batch(self, engine, batch):
        x, _ = batch
        x = x.to(self._device)
        return {"metrics": {"loss": self._valid_loss(self._module, x)}}

    def _log_training_info(self, engine):
        i = engine.state.iteration

        if i % self._STEPS_PER_LOSS_WRITE == 0:
            for k, v in engine.state.output["metrics"].items():
                self._writer.write_scalar("train/" + k, v, global_step=i)

        # TODO: Inefficient to recompute this if we are doing gradient clipping
        if i % self._STEPS_PER_GRAD_WRITE == 0:
            self._writer.write_scalar("train/grad-norm", self._get_grad_norm(), global_step=i)

        # TODO: We should do this _before_ calling self._lr_scheduler.step(), since
        # we will not correspond to the learning rate used at iteration i otherwise
        if i % self._STEPS_PER_LR_WRITE == 0:
            self._writer.write_scalar("train/lr", self._get_lr(), global_step=i)

    def _get_grad_norm(self):
        norm = 0
        for param in self._module.parameters():
            if param.grad is not None:
                norm += param.grad.norm().item()**2
        return np.sqrt(norm)

    def _get_lr(self):
        # HACK: Just looking at the first optimizer isn't correct
        param_group, = self._optimizers[0].param_groups
        return param_group["lr"]

    def _save_checkpoint(self, tag):
        # We do this manually (i.e. don't use Ignite's checkpointing) because
        # Ignite only allows saving objects, not scalars (e.g. the current epoch) 
        checkpoint = {
            "epoch": self._trainer.state.epoch,
            "iteration": self._trainer.state.iteration,
            "module_state_dict": self._module.state_dict(),
            "opt_state_dicts": [opt.state_dict() for opt in self._optimizers],
            "best_valid_loss": self._best_valid_loss,
            "num_bad_valid_epochs": self._num_bad_valid_epochs,
            "lr_scheduler_state_dicts": [lr_sched.state_dict() for lr_sched in self._lr_schedulers]
        }

        self._writer.write_checkpoint(tag, checkpoint)

    def _load_checkpoint(self, tag):
        checkpoint = self._writer.load_checkpoint(tag, device=self._device)

        @self._trainer.on(Events.STARTED)
        def resume_trainer_state(engine):
            engine.state.epoch = checkpoint["epoch"]
            engine.state.iteration = checkpoint["iteration"]

        self._module.load_state_dict(checkpoint["module_state_dict"])
        self._best_valid_loss = checkpoint["best_valid_loss"]
        self._num_bad_valid_epochs = checkpoint["num_bad_valid_epochs"]

        def load_state_dict_list(key, old_key, object_list):
            try:
                state_dicts = checkpoint[key]
                for i, obj in enumerate(object_list):
                    obj.load_state_dict(state_dicts[i])
            except KeyError: # HACK: Accommodate old behaviour
                object_list[0].load_state_dict(checkpoint[old_key])

        load_state_dict_list("opt_state_dicts", "opt_state_dict", self._optimizers)
        load_state_dict_list("lr_scheduler_state_dicts", "lr_scheduler_state_dict", self._lr_schedulers)

        print(f"Loaded checkpoint `{tag}' after epoch {checkpoint['epoch']}", file=sys.stderr)
