from .hooks import Hook


class EMAHook(Hook):
    def __init__(self,
                 momentum=0.9999,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.9,
                 evaluate_on_ema=True,
                 evaluate_on_nonema=False,
                 full_params_ema=False,
                 update_interval=1,
                 **kwargs):
        assert isinstance(update_interval, int) and update_interval > 0
        assert momentum > 0 and momentum < 1
        self.momentum = momentum
        self.regular_momentum = momentum
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up!')
            assert warmup_iters > 0 and 0 < warmup_ratio <= 1.0
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.update_interval = update_interval

        if not evaluate_on_ema and not evaluate_on_nonema:
            evaluate_on_nonema = True
        self.evaluate_on_ema = evaluate_on_ema
        self.evaluate_on_nonema = evaluate_on_nonema
        self.full_params_ema = full_params_ema

    def get_warmup_momentum(self, cur_iters):
        if self.warmup == 'constant':
            warmup_m = self.warmup_ratio * self.momentum
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_m = (1 - k) * self.momentum
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_m = k * self.momentum
        return warmup_m

    def before_run(self, runner):
        model = runner.method.model
        if runner._dist:
            model = model.module
        self.param_ema_buffer = {}
        if self.full_params_ema:
            self.model_parameters = dict(model.state_dict())
        else:
            self.model_parameters = dict(model.named_parameters(recurse=True))
        for name, value in self.model_parameters.items():
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            curr_iter = runner._iter
            if self.warmup is None or curr_iter > self.warmup_iters:
                self.regular_momentum = self.momentum
            else:
                self.regular_momentum = self.get_warmup_momentum(curr_iter)
            for name, parameter in self.model_parameters.items():
                buffer_name = self.param_ema_buffer[name]
                buffer_parameter = self.model_buffers[buffer_name]
                buffer_parameter.mul_(self.regular_momentum).add_(
                    parameter.data, alpha=1. - self.regular_momentum)

    def after_train_epoch(self, runner):
        self._swap_ema_parameters()

    def before_train_epoch(self, runner):
        self._swap_ema_parameters()

    def before_val_epoch(self, runner):
        if self.evaluate_on_ema:
            print('switch to EMA params')
            self._swap_ema_parameters()

    def after_val_epoch(self, runner):
        if self.evaluate_on_ema:
            print('switch back to ori params')
            self._swap_ema_parameters()

    def _swap_ema_parameters(self):
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)


class SwitchEMAHook(Hook):
    def __init__(self,
                 momentum=0.9999,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.9,
                 switch_params=False,
                 switch_by_iter=False,
                 switch_start=0,
                 switch_end=None,
                 switch_interval=1,
                 full_params_ema=False,
                 update_interval=1,
                 **kwargs):
        assert isinstance(update_interval, int) and update_interval > 0
        assert momentum > 0 and momentum < 1
        self.momentum = momentum
        self.regular_momentum = momentum
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up!')
            assert warmup_iters > 0 and 0 < warmup_ratio <= 1.0
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.update_interval = update_interval

        self.switch_params = switch_params
        self.switch_by_iter = switch_by_iter
        self.switch_start = switch_start
        self.switch_end = switch_end \
            if isinstance(switch_end, int) and self.switch_params else 1e100
        self.switch_interval = switch_interval
        self.full_params_ema = full_params_ema

    def get_warmup_momentum(self, cur_iters):
        if self.warmup == 'constant':
            warmup_m = self.warmup_ratio * self.momentum
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_m = (1 - k) * self.momentum
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_m = k * self.momentum
        return warmup_m

    def before_run(self, runner):
        model = runner.method.model
        if runner._dist:
            model = model.module
        self.param_ema_buffer = {}
        if self.full_params_ema:
            self.model_parameters = dict(model.state_dict())
        else:
            self.model_parameters = dict(model.named_parameters(recurse=True))
        for name, value in self.model_parameters.items():
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers(recurse=True))

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            curr_iter = runner._iter
            if self.warmup is None or curr_iter > self.warmup_iters:
                self.regular_momentum = self.momentum
            else:
                self.regular_momentum = self.get_warmup_momentum(curr_iter)
            for name, parameter in self.model_parameters.items():
                buffer_name = self.param_ema_buffer[name]
                buffer_parameter = self.model_buffers[buffer_name]
                buffer_parameter.mul_(self.regular_momentum).add_(
                    parameter.data, alpha=1. - self.regular_momentum)
        
        if self.switch_params and self.switch_by_iter:
            curr_iter = runner._iter
            if self.switch_start < curr_iter < self.switch_end:
                if not self.every_n_iters(runner, self.switch_interval):
                    self._switch_ema_parameters()

    def after_train_epoch(self, runner):
        if self.switch_end < runner._epoch:
            return
        self._swap_ema_parameters()

    def before_train_epoch(self, runner):
        if self.switch_end < runner._epoch:
            return
        self._swap_ema_parameters()
        if self.switch_params and not self.switch_by_iter:
            if self.switch_start < runner._epoch:
                if not self.every_n_epochs(runner, self.switch_interval):
                    self._switch_ema_parameters()

    def before_val_epoch(self, runner):
        self._swap_ema_parameters()

    def after_val_epoch(self, runner):
        self._swap_ema_parameters()

    def _swap_ema_parameters(self):
        for name, value in self.model_parameters.items():
            temp = value.data.clone()
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)
            ema_buffer.data.copy_(temp)

    def _switch_ema_parameters(self):
        for name, value in self.model_parameters.items():
            ema_buffer = self.model_buffers[self.param_ema_buffer[name]]
            value.data.copy_(ema_buffer.data)