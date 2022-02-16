import torch.optim 
import math 
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
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
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
