import os
import sys
import time
import yaml
import csv
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from fvcore.nn import FlopCountAnalysis

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
BASE_DIR = REPO_ROOT
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

def resolve_path(p):
    if os.path.isabs(p):
        return p
    p_root = os.path.join(REPO_ROOT, p)
    if os.path.exists(p_root):
        return p_root
    p_base = os.path.join(BASE_DIR, p)
    if os.path.exists(p_base):
        return p_base
    return p_root

from dataset_lol import LOLDataset
from scheduler import LinearWarmupCosineAnnealingLR
from metrics import PSNR, SSIM
from vgg_loss import VGGLoss
from loss import YUVLoss

from models.encoder_decoder import EncoderDecoderEnhancement
from models.nafnet import NAFNet
from models.nafnet_channel_attention import NAFNetChannelAttn
from models.nafnet_adaptive_gating import NAFNetAdaptiveGating

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_model(cfg):
    arch = cfg['model']['arch']
    m_cfg = cfg['model']
    if arch == 'encoder_decoder':
        return EncoderDecoderEnhancement(
            base_channels=m_cfg.get('base_channels', 32),
            num_res_blocks=m_cfg.get('num_res_blocks', 2)
        )
    elif arch == 'nafnet':
        return NAFNet(
            width=m_cfg.get('width', 32),
            middle_blk_num=m_cfg.get('middle_blk_num', 1),
            enc_blk_nums=m_cfg.get('enc_blk_nums', [2, 2, 4, 8]),
            dec_blk_nums=m_cfg.get('dec_blk_nums', [2, 2, 2, 2])
        )
    elif arch == 'nafnet_channel':
        return NAFNetChannelAttn(
            width=m_cfg.get('width', 32),
            middle_blk_num=m_cfg.get('middle_blk_num', 1),
            enc_blk_nums=m_cfg.get('enc_blk_nums', [2, 2, 4, 8]),
            dec_blk_nums=m_cfg.get('dec_blk_nums', [2, 2, 2, 2])
        )
    elif arch == 'nafnet_gating':
        return NAFNetAdaptiveGating(
            width=m_cfg.get('width', 32),
            middle_blk_num=m_cfg.get('middle_blk_num', 1),
            enc_blk_nums=m_cfg.get('enc_blk_nums', [2, 2, 4, 8]),
            dec_blk_nums=m_cfg.get('dec_blk_nums', [2, 2, 2, 2])
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

class CompositeLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.1, w_vgg=0.05, w_yuv=0.1, device='cuda'):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_vgg = w_vgg
        self.w_yuv = w_yuv
        self.l1 = nn.L1Loss().to(device)
        self.ssim = SSIM(channels=3).to(device)
        self.yuv = YUVLoss().to(device)
        self.vgg = VGGLoss().to(device)

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_ssim = self.ssim(pred, target, as_loss=True)
        loss_yuv = self.yuv(pred, target)
        loss_vgg = self.vgg(pred, target)
        total = (self.w_l1 * loss_l1 +
                 self.w_ssim * loss_ssim +
                 self.w_vgg * loss_vgg +
                 self.w_yuv * loss_yuv)
        return total, {
            'l1': loss_l1.item(),
            'ssim': loss_ssim.item(),
            'yuv': loss_yuv.item(),
            'vgg': loss_vgg.item()
        }

def save_image_tensor(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = tensor.detach().cpu().clamp(0, 1).numpy()
    t = (t.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    Image.fromarray(t).save(path)

def evaluate_dataset(model, loader, criterion, device, save_images_dir=None):
    model.eval()
    psnr_fn = PSNR()
    ssim_fn = SSIM(channels=3)

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_time = 0.0
    results = []

    with torch.no_grad():
        for low_img, high_img, filenames in loader:
            low_img = low_img.to(device)
            high_img = high_img.to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                enhanced = model(low_img)
            enhanced_clamped = torch.clamp(enhanced.float(), 0.0, 1.0)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.time()

            total_time += (t1 - t0)

            if criterion is not None:
                batch_loss, _ = criterion(enhanced_clamped, high_img.float())
                total_loss += batch_loss.item() * low_img.size(0)

            for b in range(low_img.size(0)):
                pred_b = enhanced_clamped[b:b+1]
                gt_b = high_img[b:b+1]
                fname = filenames[b]

                cur_psnr = psnr_fn(pred_b, gt_b).item()
                cur_ssim = ssim_fn(pred_b, gt_b, as_loss=False).mean().item()

                total_psnr += cur_psnr
                total_ssim += cur_ssim

                results.append({
                    "image": fname,
                    "psnr": round(cur_psnr, 4),
                    "ssim": round(cur_ssim, 4)
                })

                if save_images_dir:
                    save_path = os.path.join(save_images_dir, fname)
                    save_image_tensor(enhanced_clamped[b], save_path)

    n_samples = len(results)
    mean_loss = total_loss / n_samples if n_samples > 0 else 0.0
    mean_psnr = total_psnr / n_samples if n_samples > 0 else 0.0
    mean_ssim = total_ssim / n_samples if n_samples > 0 else 0.0
    avg_time = total_time / n_samples if n_samples > 0 else 0.0

    return mean_loss, mean_psnr, mean_ssim, avg_time, results

def measure_complexity(model, device, input_res=(1, 3, 256, 256)):
    dummy = torch.randn(*input_res, device=device)
    try:
        analyzer = FlopCountAnalysis(model, dummy)
        gflops = analyzer.total() / 1e9
    except Exception as e:
        print(f"[Note] FLOP computation notice: {e}")
        gflops = 0.0
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = total_params / 1e6
    return round(gflops, 3), round(params_m, 2), total_params

def benchmark_latency(model, device, num_runs=50, input_res=(1, 3, 256, 256)):
    model.eval()
    dummy = torch.randn(*input_res, device=device)
    with torch.no_grad():
        for _ in range(10):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = model(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.time()
            latencies.append((t1 - t0) * 1000)

    avg_ms = round(sum(latencies) / len(latencies), 2)
    std_ms = round((sum((x - avg_ms)**2 for x in latencies) / len(latencies)) ** 0.5, 2)
    return avg_ms, std_ms

def train_final(config_path, resume_path=None):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    experiment_name = cfg['experiment_name']
    arch = cfg['model']['arch']
    train_cfg = cfg['training']
    data_cfg = cfg['data']
    out_cfg = cfg['output']

    set_seed(train_cfg.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Error: CUDA with BF16 support is required for final benchmark."

    print("=" * 80)
    print(f"FINAL CORRECTED BENCHMARK — {experiment_name}")
    print(f"  Architecture: {arch}")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"  Precision: Uniform Native BF16 Mixed Precision (Loss in FP32)")
    print(f"  Fixed Budget: {train_cfg['epochs']} Epochs (No dynamic extensions)")
    print(f"  Batch Size: {train_cfg['batch_size']} | LR: {train_cfg['lr']}")
    print(f"  Random Seed: {train_cfg.get('seed', 42)}")
    print("=" * 80, flush=True)

    for d in [out_cfg['checkpoint_dir'], out_cfg['results_dir'], out_cfg['log_dir']]:
        os.makedirs(resolve_path(d), exist_ok=True)

    # 1. Dataset splits
    train_low = resolve_path(data_cfg['train_low'])
    train_high = resolve_path(data_cfg['train_high'])
    val_low = resolve_path(data_cfg['val_low'])
    val_high = resolve_path(data_cfg['val_high'])
    test_low = resolve_path(data_cfg['test_low'])
    test_high = resolve_path(data_cfg['test_high'])

    train_dataset = LOLDataset(train_low, train_high, patch_size=data_cfg.get('patch_size', 256), is_train=True)
    val_dataset = LOLDataset(val_low, val_high, patch_size=None, is_train=False)
    test_dataset = LOLDataset(test_low, test_high, patch_size=None, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 0),
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 2. Build Model
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Built model: {arch} | Trainable parameters: {total_params:,}", flush=True)

    # 3. Criterion, Optimizer, Scheduler
    lw = train_cfg.get('loss_weights', {'l1': 1.0, 'ssim': 0.1, 'vgg': 0.05, 'yuv': 0.1})
    criterion = CompositeLoss(
        w_l1=lw.get('l1', 1.0),
        w_ssim=lw.get('ssim', 0.1),
        w_vgg=lw.get('vgg', 0.05),
        w_yuv=lw.get('yuv', 0.1),
        device=device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg.get('weight_decay', 1e-4),
        betas=(0.9, 0.999)
    )

    total_epochs = train_cfg['epochs'] # Exactly 150
    warmup_epochs = 5
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=total_epochs,
        warmup_start_lr=1e-6,
        eta_min=1e-6
    )

    start_epoch = 1
    best_val_psnr = 0.0
    best_val_ssim = 0.0
    training_history = []

    # Auto-resume capability
    if resume_path is None:
        auto_resume = os.path.join(BASE_DIR, out_cfg['checkpoint_dir'], 'latest_model.pth')
        if os.path.isfile(auto_resume):
            resume_path = auto_resume

    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming from checkpoint: {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_psnr = ckpt.get('best_val_psnr', 0.0)
        best_val_ssim = ckpt.get('best_val_ssim', 0.0)
        training_history = ckpt.get('training_history', [])
        print(f"  Resumed at epoch {start_epoch} | Best Val PSNR: {best_val_psnr:.2f} dB, SSIM: {best_val_ssim:.4f}", flush=True)

    grad_clip = train_cfg.get('grad_clip', 1.0)
    total_start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_l1 = 0.0
        running_ssim = 0.0
        running_vgg = 0.0
        running_yuv = 0.0
        running_grad_norm = 0.0
        has_nan_epoch = False
        n_batches = 0

        for step, (low_img, high_img, _) in enumerate(train_loader):
            low_img = low_img.to(device, non_blocking=True)
            high_img = high_img.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                enhanced = model(low_img)
            
            enhanced_clamped = torch.clamp(enhanced.float(), 0.0, 1.0)
            loss, loss_dict = criterion(enhanced_clamped, high_img.float())

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[CRITICAL WARNING] NaN/Inf detected in loss at Ep {epoch} Step {step}!", flush=True)
                has_nan_epoch = True

            loss.backward()

            if grad_clip > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if torch.isnan(gnorm) or torch.isinf(gnorm):
                    has_nan_epoch = True
                running_grad_norm += gnorm.item() if not (torch.isnan(gnorm) or torch.isinf(gnorm)) else 0.0

            optimizer.step()

            running_loss += loss.item()
            running_l1 += loss_dict['l1']
            running_ssim += loss_dict['ssim']
            running_vgg += loss_dict['vgg']
            running_yuv += loss_dict['yuv']
            n_batches += 1

            if (step + 1) % train_cfg.get('log_interval', 20) == 0 or (step + 1) == len(train_loader):
                print(f"  [{arch}] Ep {epoch:03d}/{total_epochs:03d} "
                      f"Step [{step+1:03d}/{len(train_loader):03d}] "
                      f"Loss: {loss.item():.4f} (L1: {loss_dict['l1']:.4f}, SSIM: {loss_dict['ssim']:.4f}, VGG: {loss_dict['vgg']:.4f})", flush=True)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        train_loss = running_loss / max(1, n_batches)
        avg_l1 = running_l1 / max(1, n_batches)
        avg_ssim_l = running_ssim / max(1, n_batches)
        avg_vgg = running_vgg / max(1, n_batches)
        avg_yuv = running_yuv / max(1, n_batches)
        avg_grad_norm = running_grad_norm / max(1, n_batches)

        # Validation evaluation every epoch
        val_loss, val_psnr, val_ssim, val_time, _ = evaluate_dataset(model, val_loader, criterion, device)

        print(f"\n>> [{arch}] Epoch {epoch:03d}/{total_epochs:03d} | {epoch_time:.1f}s | LR: {current_lr:.6f} "
              f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f} | Grad Norm: {avg_grad_norm:.3f}", flush=True)

        is_best = False
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_val_ssim = val_ssim
            is_best = True
            best_path = os.path.join(BASE_DIR, out_cfg['checkpoint_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_psnr': best_val_psnr,
                'best_val_ssim': best_val_ssim,
                'config': cfg,
            }, best_path)
            print(f"  [BEST CHECKPOINT] Val PSNR={best_val_psnr:.2f} dB, SSIM={best_val_ssim:.4f} -> {best_path}\n", flush=True)

        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.type == 'cuda' else 0.0

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_psnr": round(val_psnr, 4),
            "val_ssim": round(val_ssim, 6),
            "l1": round(avg_l1, 6),
            "ssim_loss": round(avg_ssim_l, 6),
            "vgg_loss": round(avg_vgg, 6),
            "yuv_loss": round(avg_yuv, 6),
            "grad_norm": round(avg_grad_norm, 4),
            "has_nan": has_nan_epoch,
            "lr": round(current_lr, 8),
            "epoch_time_sec": round(epoch_time, 2),
            "val_inference_ms": round(val_time * 1000, 1),
            "peak_gpu_mem_gb": round(peak_mem_gb, 3)
        }
        training_history.append(epoch_record)

        latest_path = os.path.join(BASE_DIR, out_cfg['checkpoint_dir'], 'latest_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_psnr': best_val_psnr,
            'best_val_ssim': best_val_ssim,
            'training_history': training_history,
            'config': cfg,
        }, latest_path)

        log_csv = os.path.join(BASE_DIR, out_cfg['log_dir'], "training_log.csv")
        csv_fields = ["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "l1", "ssim_loss", "vgg_loss", "yuv_loss", "grad_norm", "has_nan", "lr", "epoch_time_sec", "val_inference_ms", "peak_gpu_mem_gb"]
        with open(log_csv, 'w', newline='') as lf:
            writer = csv.DictWriter(lf, fieldnames=csv_fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(training_history)

    total_training_time_min = (time.time() - total_start_time) / 60
    print("\n" + "=" * 80)
    print(f"TRAINING PHASE CONCLUDED FOR {arch.upper()} in {total_training_time_min:.1f} minutes")
    print(f"Best Validation PSNR: {best_val_psnr:.2f} dB | Best Validation SSIM: {best_val_ssim:.4f}")
    print("=" * 80, flush=True)

    # 5. FINAL TEST EVALUATION ON UNTOUCHED EVAL15
    print(f"\nInitiating final evaluation on untouched 15-image official test set using best validation checkpoint...", flush=True)
    best_path = os.path.join(BASE_DIR, out_cfg['checkpoint_dir'], 'best_model.pth')
    if os.path.isfile(best_path):
        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        print(f"Loaded best checkpoint from epoch {best_ckpt.get('epoch', 'N/A')}", flush=True)

    test_enhanced_dir = os.path.join(BASE_DIR, out_cfg['results_dir'], "enhanced")
    _, test_psnr, test_ssim, test_latency, per_image = evaluate_dataset(
        model, test_loader, None, device, save_images_dir=test_enhanced_dir
    )

    gflops, params_m, total_params = measure_complexity(model, device)
    avg_latency_ms, std_latency_ms = benchmark_latency(model, device)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == 'cuda' else 0.0

    print("\n" + "=" * 70)
    print(f"FINAL OFFICIAL TEST SET RESULTS (eval15) — {arch.upper()}")
    print("=" * 70)
    print(f"{'Image':<15} | {'PSNR (dB)':<12} | {'SSIM':<10}")
    print("-" * 70)
    for r in per_image:
        print(f"{r['image']:<15} | {r['psnr']:<12.4f} | {r['ssim']:<10.4f}")
    print("-" * 70)
    print(f"{'MEAN':<15} | {test_psnr:<12.4f} | {test_ssim:<10.4f}")
    print(f"{'Inference Latency':<15} | {avg_latency_ms:.2f} ± {std_latency_ms:.2f} ms")
    print(f"{'Computational':<15} | {gflops:.3f} GFLOPs ({params_m:.2f}M params)")
    print(f"{'Peak GPU Memory':<15} | {peak_vram_mb:.1f} MB")
    print("=" * 70, flush=True)

    per_img_csv = os.path.join(BASE_DIR, out_cfg['results_dir'], "per_image_metrics.csv")
    with open(per_img_csv, 'w', newline='') as pf:
        writer = csv.DictWriter(pf, fieldnames=["image", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(per_image)

    final_report = {
        "experiment_name": experiment_name,
        "arch_key": arch,
        "config_path": config_path,
        "total_params": total_params,
        "params_m": params_m,
        "gflops": gflops,
        "avg_latency_ms": avg_latency_ms,
        "std_latency_ms": std_latency_ms,
        "peak_gpu_mb": round(peak_vram_mb, 1),
        "best_epoch": best_ckpt.get('epoch', 'N/A') if 'best_ckpt' in locals() else 'N/A',
        "best_val_psnr": round(best_val_psnr, 4),
        "best_val_ssim": round(best_val_ssim, 6),
        "test_mean_psnr": round(test_psnr, 4),
        "test_mean_ssim": round(test_ssim, 6),
        "training_time_min": round(total_training_time_min, 2),
        "per_image_results": per_image,
        "config": cfg,
    }
    report_json = os.path.join(BASE_DIR, out_cfg['results_dir'], "experiment_report.json")
    with open(report_json, 'w') as rf:
        json.dump(final_report, rf, indent=2)

    print(f"Results recorded to: {report_json}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Final Benchmark Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to final YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume")
    args = parser.parse_args()

    train_final(args.config, args.resume)

if __name__ == "__main__":
    main()

