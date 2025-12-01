import cv2
import numpy as np
import torch

try:
    import lpips
    from skimage.metrics import structural_similarity as cal_ssim
except:
    lpips = None
    cal_ssim = None


def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1


def MAE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean(np.abs(pred - true), axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean(np.abs(pred - true) / norm, axis=(0, 1)).sum()


def MSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.mean((pred - true)**2, axis=(0, 1)).sum()
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.mean((pred - true)**2 / norm, axis=(0, 1)).sum()


def RMSE(pred, true, spatial_norm=False):
    if not spatial_norm:
        return np.sqrt(np.mean((pred - true)**2, axis=(0, 1)).sum())
    else:
        norm = pred.shape[-1] * pred.shape[-2] * pred.shape[-3]
        return np.sqrt(np.mean((pred - true)**2 / norm, axis=(0, 1)).sum())


def PSNR(pred, true, min_max_norm=True):
    mse = np.mean((pred.astype(np.float32) - true.astype(np.float32))**2)
    if mse == 0:
        return float('inf')
    else:
        if min_max_norm:
            return 20. * np.log10(1. / np.sqrt(mse))
        else:
            return 20. * np.log10(255. / np.sqrt(mse))


def SNR(pred, true):
    signal = ((true)**2).mean()
    noise = ((true - pred)**2).mean()
    return 10. * np.log10(signal / noise)


class LPIPS(torch.nn.Module):
    def __init__(self, net='alex', use_gpu=True):
        super().__init__()
        assert net in ['alex', 'squeeze', 'vgg']
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.loss_fn = lpips.LPIPS(net=net)
        if self.use_gpu:
            self.loss_fn.cuda()

    def forward(self, img1, img2):
        img1 = lpips.im2tensor(img1 * 255)
        img2 = lpips.im2tensor(img2 * 255)
        if self.use_gpu:
            img1, img2 = img1.cuda(), img2.cuda()
        return self.loss_fn.forward(img1, img2).squeeze().detach().cpu().numpy()


def metric(pred, true, mean=None, std=None, metrics=['mae', 'mse'],
             clip_range=[0, 1], channel_names=None,
             spatial_norm=False, return_log=True):
    if mean is not None and std is not None:
        pred = pred * std + mean
        true = true * std + mean
    eval_res = {}
    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'psnr', 'snr', 'lpips']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')
    if isinstance(channel_names, list):
        assert pred.shape[2] % len(channel_names) == 0 and len(channel_names) > 1
        c_group = len(channel_names)
        c_width = pred.shape[2] // c_group
    else:
        channel_names, c_group, c_width = None, None, None

    if 'mse' in metrics:
        if channel_names is None:
            eval_res['mse'] = MSE(pred, true, spatial_norm)
        else:
            mse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mse_{str(c_name)}'] = MSE(pred[:, :, i * c_width: (i + 1) * c_width, ...],
                                                     true[:, :, i * c_width: (i + 1) * c_width, ...], spatial_norm)
                mse_sum += eval_res[f'mse_{str(c_name)}']
            eval_res['mse'] = mse_sum / c_group

    if 'mae' in metrics:
        if channel_names is None:
            eval_res['mae'] = MAE(pred, true, spatial_norm)
        else:
            mae_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'mae_{str(c_name)}'] = MAE(pred[:, :, i * c_width: (i + 1) * c_width, ...],
                                                     true[:, :, i * c_width: (i + 1) * c_width, ...], spatial_norm)
                mae_sum += eval_res[f'mae_{str(c_name)}']
            eval_res['mae'] = mae_sum / c_group

    if 'rmse' in metrics:
        if channel_names is None:
            eval_res['rmse'] = RMSE(pred, true, spatial_norm)
        else:
            rmse_sum = 0.
            for i, c_name in enumerate(channel_names):
                eval_res[f'rmse_{str(c_name)}'] = RMSE(pred[:, :, i * c_width: (i + 1) * c_width, ...],
                                                       true[:, :, i * c_width: (i + 1) * c_width, ...], spatial_norm)
                rmse_sum += eval_res[f'rmse_{str(c_name)}']
            eval_res['rmse'] = rmse_sum / c_group

    pred = np.maximum(pred, clip_range[0])
    pred = np.minimum(pred, clip_range[1])

    if 'ssim' in metrics:
        if cal_ssim is None:
            eval_res['ssim'] = 0.0
        else:
            ssim = 0.0
            num_frames = pred.shape[0] * pred.shape[1]
            for b in range(pred.shape[0]):
                for f in range(pred.shape[1]):
                    img_pred_slice = pred[b, f]
                    img_true_slice = true[b, f]
                    if img_pred_slice.ndim == 3:
                        if img_pred_slice.shape[0] == 1:
                            img_pred = img_pred_slice[0]
                            img_true = img_true_slice[0]
                            h, w = img_pred.shape
                            min_side = min(h, w)
                            ws = 7 if min_side >= 7 else (min_side // 2 * 2 + 1)
                            if ws < 1:
                                ws = 1
                            ssim_val = cal_ssim(img_pred, img_true, multichannel=False, channel_axis=None, win_size=ws, data_range=1.0)
                        else:
                            ssim_val = 0.0
                            for c in range(img_pred_slice.shape[0]):
                                p_c = img_pred_slice[c]
                                t_c = img_true_slice[c]
                                h, w = p_c.shape
                                min_side = min(h, w)
                                ws = 7 if min_side >= 7 else (min_side // 2 * 2 + 1)
                                if ws < 1:
                                    ws = 1
                                ssim_val += cal_ssim(p_c, t_c, multichannel=False, channel_axis=None, win_size=ws, data_range=1.0)
                            ssim_val /= img_pred_slice.shape[0]
                    else:
                        h, w = img_pred_slice.shape
                        min_side = min(h, w)
                        ws = 7 if min_side >= 7 else (min_side // 2 * 2 + 1)
                        if ws < 1:
                            ws = 1
                        ssim_val = cal_ssim(img_pred_slice, img_true_slice, multichannel=False, channel_axis=None, win_size=ws, data_range=1.0)
                    ssim += ssim_val
            eval_res['ssim'] = ssim / num_frames

    if 'psnr' in metrics:
        psnr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                psnr += PSNR(pred[b, f], true[b, f])
        eval_res['psnr'] = psnr / (pred.shape[0] * pred.shape[1])

    if 'snr' in metrics:
        snr = 0
        for b in range(pred.shape[0]):
            for f in range(pred.shape[1]):
                snr += SNR(pred[b, f], true[b, f])
        eval_res['snr'] = snr / (pred.shape[0] * pred.shape[1])

    if 'lpips' in metrics:
        if lpips is None:
            eval_res['lpips'] = 0.0
        else:
            cal_lpips = LPIPS(net='alex', use_gpu=torch.cuda.is_available())
            pred_for_lpips = pred.copy()
            true_for_lpips = true.copy()
            if len(pred.shape) == 4:
                pred_for_lpips = np.repeat(pred_for_lpips[..., np.newaxis], 3, axis=-1)
                true_for_lpips = np.repeat(true_for_lpips[..., np.newaxis], 3, axis=-1)
            else:
                pred_for_lpips = np.transpose(pred_for_lpips, (0, 1, 3, 4, 2))
                true_for_lpips = np.transpose(true_for_lpips, (0, 1, 3, 4, 2))
            lpips_val = 0.0
            num_frames = pred.shape[0] * pred.shape[1]
            for b in range(pred.shape[0]):
                for f in range(pred.shape[1]):
                    if len(pred.shape) == 4:
                        p_frame = pred_for_lpips[b, f]
                        t_frame = true_for_lpips[b, f]
                    else:
                        p_frame = pred_for_lpips[b, f]
                        t_frame = true_for_lpips[b, f]
                    lpips_val += cal_lpips(p_frame, t_frame)
            eval_res['lpips'] = lpips_val / num_frames

    if return_log:
        for k, v in eval_res.items():
            eval_str = f"{k}:{v}" if len(eval_log) == 0 else f", {k}:{v}"
            eval_log += eval_str

    return eval_res, eval_log