from typing import Dict, List, Union
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from contextlib import suppress
from timm.utils import NativeScaler
from timm.utils.agc import adaptive_clip_grad

from openstl.core import metric
from openstl.core.optim_scheduler import get_optim_scheduler
from openstl.utils import gather_tensors_batch, get_dist_info, ProgressBar
from openstl.datasets.dataset_constant import dataset_parameters

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


class Base_method(object):
    def __init__(self, args, device, steps_per_epoch):
        super(Base_method, self).__init__()
        self.args = args
        self.dist = args.dist
        self.device = device
        self.config = args.__dict__
        self.criterion = None
        self.model_optim = None
        self.scheduler = None
        if self.dist:
            self.rank, self.world_size = get_dist_info()
            assert self.rank == int(device.split(':')[-1])
        else:
            self.rank, self.world_size = 0, 1
        self.clip_value = self.args.clip_grad
        self.clip_mode = self.args.clip_mode if self.clip_value is not None else None
        self.amp_autocast = suppress
        self.loss_scaler = None
        self.metric_list = dataset_parameters[args.dataname]['metrics']
        self.spatial_norm = False
        if 'weather' in self.args.dataname:
            self.spatial_norm = True

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def _init_optimizer(self, steps_per_epoch):
        return get_optim_scheduler(
            self.args, self.args.epoch, self.model, steps_per_epoch)

    def _init_distributed(self):
        if self.args.fp16 and has_native_amp:
            self.amp_autocast = torch.cuda.amp.autocast
            self.loss_scaler = NativeScaler()
            if self.rank == 0:
                print('Using native PyTorch AMP. Training in mixed precision (fp16).')
        else:
            print('AMP not enabled. Training in float32.')
        self.model = NativeDDP(self.model, device_ids=[self.rank],
                               broadcast_buffers=self.args.broadcast_buffers,
                               find_unused_parameters=self.args.find_unused_parameters)

    def train_one_epoch(self, runner, train_loader, **kwargs):
        raise NotImplementedError

    def _predict(self, batch_x, batch_y, **kwargs):
        raise NotImplementedError

    def _dist_forward_collect(self, data_loader, length=None, gather_data=False):
        results = []
        length = len(data_loader.dataset) if length is None else length
        if self.rank == 0:
            prog_bar = ProgressBar(len(data_loader))

        for idx, (batch_x, batch_y) in enumerate(data_loader):
            if idx == 0:
                part_size = batch_x.shape[0]
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self._predict(batch_x, batch_y)

            if gather_data:
                results.append(dict(zip(['inputs', 'preds', 'trues'],
                                        [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy()])))
            else:
                eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                     data_loader.dataset.mean, data_loader.dataset.std,
                                     metrics=self.metric_list, spatial_norm=self.spatial_norm, return_log=False)
                eval_res['loss'] = self.criterion(pred_y, batch_y).cpu().numpy()
                for k in eval_res.keys():
                    if isinstance(eval_res[k], (int, float)):
                        eval_res[k] = np.array([eval_res[k]])
                    else:
                        eval_res[k] = eval_res[k].reshape(1)
                results.append(eval_res)

            if self.args.empty_cache:
                torch.cuda.empty_cache()
            if self.rank == 0:
                prog_bar.update()

        results_all = {}
        for k in results[0].keys():
            results_cat = np.concatenate([batch[k] for batch in results], axis=0)
            results_gathered = gather_tensors_batch(results_cat, part_size=min(part_size*8, 16))
            results_strip = np.concatenate(results_gathered, axis=0)[:length]
            results_all[k] = results_strip
        return results_all

    def _nondist_forward_collect(self, data_loader, length=None, gather_data=False):
        results = []
        prog_bar = ProgressBar(len(data_loader))
        length = len(data_loader.dataset) if length is None else length

        for idx, (batch_x, batch_y) in enumerate(data_loader):
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self._predict(batch_x, batch_y)

            if gather_data:
                results.append(dict(zip(['inputs', 'preds', 'trues'],
                                        [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy()])))
            else:
                eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                     data_loader.dataset.mean, data_loader.dataset.std,
                                     metrics=self.metric_list, spatial_norm=self.spatial_norm, return_log=False)
                eval_res['loss'] = self.criterion(pred_y, batch_y).cpu().numpy()

                for k in eval_res.keys():
                    if isinstance(eval_res[k], (int, float)):
                        eval_res[k] = np.array([eval_res[k]])
                    else:
                        eval_res[k] = eval_res[k].reshape(1)
                results.append(eval_res)

            prog_bar.update()
            if self.args.empty_cache:
                torch.cuda.empty_cache()

        results_all = {}
        for k in results[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in results], axis=0)
        return results_all

    def vali_one_epoch(self, runner, vali_loader, **kwargs):
        self.model.eval()
        if self.dist and self.world_size > 1:
            results = self._dist_forward_collect(vali_loader, len(vali_loader.dataset), gather_data=False)
        else:
            results = self._nondist_forward_collect(vali_loader, len(vali_loader.dataset), gather_data=False)

        eval_log = ""
        for k, v in results.items():
            v = v.mean()
            if k != "loss":
                eval_str = f"{k}:{v:.6f}" if len(eval_log) == 0 else f", {k}:{v:.6f}"
                eval_log += eval_str

        return results, eval_log

    def test_one_epoch(self, runner, test_loader, **kwargs):
        self.model.eval()
        if self.dist and self.world_size > 1:
            results = self._dist_forward_collect(test_loader, gather_data=True)
        else:
            results = self._nondist_forward_collect(test_loader, gather_data=True)

        return results

    def current_lr(self) -> Union[List[float], Dict[str, List[float]]]:
        lr: Union[List[float], Dict[str, List[float]]]
        if isinstance(self.model_optim, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.model_optim.param_groups]
        elif isinstance(self.model_optim, dict):
            lr = dict()
            for name, optim in self.model_optim.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def clip_grads(self, params, norm_type: float = 2.0):
        if self.clip_mode is None:
            return
        if self.clip_mode == 'norm':
            torch.nn.utils.clip_grad_norm_(params, self.clip_value, norm_type=norm_type)
        elif self.clip_mode == 'value':
            torch.nn.utils.clip_grad_value_(params, self.clip_value)
        elif self.clip_mode == 'agc':
            adaptive_clip_grad(params, self.clip_value, norm_type=norm_type)
        else:
            assert False, f"Unknown clip mode ({self.clip_mode})."