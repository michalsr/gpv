import torch.optim 
import math 
import weakref
from functools import wraps
class LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.verbose = verbose

        #self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(group, lr))
            else:
                print('Epoch {:5d}: adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(epoch, group, lr))


    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        #if self._step_count == 1:
            # if not hasattr(self.optimizer.step, "_with_counter"):
            #     warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
            #                   "initialization. Please, make sure to call `optimizer.step()` before "
            #                   "`lr_scheduler.step()`. See more details at "
            #                   "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            # elif self.optimizer._step_count < 1:
            #     warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
            #                   "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
            #                   "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
            #                   "will result in PyTorch skipping the first value of the learning rate schedule. "
            #                   "See more details at "
            #                   "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
        
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
class StepLR(LRScheduler):
     def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch, verbose)
     def get_closed_form_lr(self,epoch):
        return [base_lr * self.gamma ** (epoch // self.step_size)
                for base_lr in self.base_lrs]
class ReduceLROnPlateau(object):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                    threshold=1e-4, threshold_mode='rel', cooldown=0,
                    min_lr=0, eps=1e-8, verbose=False):

            if factor >= 1.0:
                raise ValueError('Factor should be < 1.0.')
            self.factor = factor

            # Attach optimizer
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TypeError('{} is not an Optimizer'.format(
                    type(optimizer).__name__))
            self.optimizer = optimizer

            if isinstance(min_lr, list) or isinstance(min_lr, tuple):
                if len(min_lr) != len(optimizer.param_groups):
                    raise ValueError("expected {} min_lrs, got {}".format(
                        len(optimizer.param_groups), len(min_lr)))
                self.min_lrs = list(min_lr)
            else:
                self.min_lrs = [min_lr] * len(optimizer.param_groups)

            self.patience = patience
            self.verbose = verbose
            self.cooldown = cooldown
            self.cooldown_counter = 0
            self.mode = mode
            self.threshold = threshold
            self.threshold_mode = threshold_mode
            self.best = None
            self.num_bad_epochs = None
            self.mode_worse = None  # the worse value for the chosen mode
            self.eps = eps
            self.last_epoch = 0
            self.last_best_val_score = -1
            self._init_is_better(mode=mode, threshold=threshold,
                                threshold_mode=threshold_mode)
            self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self,actual_optimizer, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = self.last_best_val_score
        print(current,'current')
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            #new_lr = self._reduce_lr(epoch)
            for i, param_group in enumerate(actual_optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = max(old_lr * self.factor, self.min_lrs[i])
                if old_lr - new_lr > self.eps:
                    param_group['lr'] = new_lr
                    if self.verbose:
                        print('Epoch {:5d}: reducing learning rate'
                            ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        self._last_lr = [group['lr'] for group in actual_optimizer.param_groups]
        return actual_optimizer
        

    def _reduce_lr(self, actual_optimizer, epoch):
        for i, param_group in enumerate(actual_optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                        ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
        return new_lr 

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = math.inf
        else:  # mode == 'max':
            self.mode_worse = -math.inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)
