"""
v5 training: EXACT v3 recipe + three deltas:
  (1) checkpoints selected by val macro-F1 (proper sklearn metric) in addition to instance acc
  (2) SWA: equal-weight average of epoch weights from --swa_start to end, BN recalibrated
  (3) same-class CutMix-R synthesis for rare classes (fresh geometric hybrids per draw,
      replacing v4's pure-copy oversampling)

Fixed (v3) recipe: CBFocalLoss sqrt weights clip10 gamma=1.0 ls=0.1, PointCutMix p=0.2 a=0.5
(unweighted soft-CE branch), cosine LR + 5 warmup, 250 epochs, bs 16, Adam 1e-3 wd 1e-4.
"""
import os, sys, csv, copy, random, argparse, datetime, logging, importlib, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import provider
from data_utils.ProteinToPointCloudProcessor import DataProcessor
from train_improved import CBFocalLoss as _CBFocalLoss, inplace_relu, count_parameters


class CBFocalLoss(_CBFocalLoss):
    """Copy of v3 CBFocalLoss with device-aware weights (parent hardcodes .cuda())."""
    def __init__(self, samples_per_class, num_classes=97, beta=0.9999, gamma=1.0,
                 label_smoothing=0.1, weight_mode='sqrt', weight_clip=10.0):
        torch.nn.Module.__init__(self)
        if weight_mode == 'sqrt':
            n = np.maximum(samples_per_class.astype(np.float64), 1.0)
            weights = 1.0 / np.sqrt(n)
            weights = weights / weights.min()
            weights = np.clip(weights, 1.0, weight_clip)
            weights = weights / weights.mean()
        else:
            effective_num = 1.0 - np.power(beta, samples_per_class)
            weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
            weights = weights / np.sum(weights) * num_classes
        self.class_weights = torch.FloatTensor(weights).to(DEVICE)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes


class RareSynthDataset(torch.utils.data.Dataset):
    """Wraps the real DataProcessor. Extra virtual items beyond n_real are
    same-class PointCutMix-R hybrids of two members of a rare class.
    Every draw is freshly randomized (no memorized copies)."""

    def __init__(self, base_dataset, labels, rare_thresh=10, n_syn=4, alpha=0.5):
        self.base = base_dataset
        self.n_real = len(base_dataset)
        labels = np.asarray(labels)
        self.cls_idx = {c: np.where(labels == c)[0] for c in np.unique(labels)}
        rare = [c for c, v in self.cls_idx.items() if len(v) <= rare_thresh]
        # per epoch: n_syn fresh hybrids per rare class
        self.syn_slots = [(c, j) for c in rare for j in range(n_syn)]
        self.alpha = alpha
        self.rare = rare

    def __len__(self):
        return self.n_real + len(self.syn_slots)

    def _mix(self, c):
        pool = self.cls_idx[c]
        i, j = np.random.choice(pool, 2, replace=(len(pool) == 1))
        a = self.base[int(i)][0].copy()
        b = self.base[int(j)][0].copy()
        ratio = np.random.beta(self.alpha, self.alpha)
        ratio = max(ratio, 1.0 - ratio)  # fraction kept from a
        mask = np.random.rand(a.shape[0]) < ratio
        out = a.copy()
        out[~mask] = b[~mask]
        return out.astype(np.float32), int(c)

    def __getitem__(self, k):
        if k < self.n_real:
            return self.base[k]
        c, _ = self.syn_slots[k - self.n_real]
        return self._mix(c)


def evaluate(model, loader, max_batches=0):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for bi, (points, target) in enumerate(tqdm(loader, total=len(loader), desc="val", leave=False)):
            points, target = points.to(DEVICE), target.to(DEVICE)
            pred, _ = model(points)
            if pred.dim() == 3:
                pred = pred.mean(dim=1)
            preds.append(pred.argmax(1).cpu().numpy()); ys.append(target.cpu().numpy())
            if max_batches and bi + 1 >= max_batches:
                break
    y, p = np.concatenate(ys), np.concatenate(preds)
    return (accuracy_score(y, p), balanced_accuracy_score(y, p),
            f1_score(y, p, average="macro", zero_division=0))


def recalibrate_bn(model, loader, n_batches=30):
    """Reset BN stats and recompute running mean/var over train batches (cumulative)."""
    keeps = []
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            keeps.append((m, m.momentum))
            m.reset_running_stats()
            m.momentum = None  # cumulative average
    model.train()
    with torch.no_grad():
        for i, (points, _) in enumerate(loader):
            if i >= n_batches:
                break
            model(points.to(DEVICE))
    for m, mom in keeps:
        m.momentum = mom


def moving_average(avg_sd, sd, n):
    return {k: avg_sd[k] + (sd[k].float() - avg_sd[k]) / n for k in avg_sd}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', default='0'); p.add_argument('--seed', type=int, default=1)
    p.add_argument('--log_dir', default='riconv_large_v5_s1')
    p.add_argument('--epoch', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--decay_rate', type=float, default=1e-4)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--focal_gamma', type=float, default=1.0)
    p.add_argument('--cutmix_alpha', type=float, default=0.5)
    p.add_argument('--cutmix_prob', type=float, default=0.2)
    p.add_argument('--weight_mode', default='sqrt'); p.add_argument('--weight_clip', type=float, default=10.0)
    # v5 deltas
    p.add_argument('--swa_start', type=int, default=200)
    p.add_argument('--rare_thresh', type=int, default=10)
    p.add_argument('--n_syn', type=int, default=4)
    p.add_argument('--no_synth', action='store_true', help='ablation: SWA+macro-selection only')
    p.add_argument('--max_train_batches', type=int, default=0, help='debug cap (0=all)')
    p.add_argument('--max_val_batches', type=int, default=0, help='debug cap (0=all)')
    return p.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    exp_dir = Path('./log/classification_shrec2025') / args.log_dir
    (exp_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'logs').mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("v5-" + args.log_dir); logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(exp_dir / 'logs' / 'train_v5.txt'); fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(fh)
    def log(s):
        logger.info(s); print(s, flush=True)
    log(str(args))
    shutil.copy('models/riconv2_cls_v2_features2_largest.py', str(exp_dir))

    root_path = "./new_data/"
    data = pd.read_csv("./csv/train_set_2.csv")
    filenames, labels = data["protein_id"], data["class_id"]
    tr_files, tr_labels = filenames[:7428], labels[:7428].values
    va_files, va_labels = filenames[7428:], labels[7428:].values

    base_train = DataProcessor(root_path, tr_files, tr_labels, pc_folder="/txt8/")
    if args.no_synth:
        train_dataset = base_train
    else:
        train_dataset = RareSynthDataset(base_train, tr_labels,
                                         rare_thresh=args.rare_thresh, n_syn=args.n_syn,
                                         alpha=args.cutmix_alpha)
        log("rare classes (<=%d): %d, synthetic items/epoch=%d, epoch size=%d"
            % (args.rare_thresh, len(train_dataset.rare), len(train_dataset.syn_slots), len(train_dataset)))
    val_dataset = DataProcessor(root_path, va_files, va_labels, pc_folder="/txt8/")
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=10, drop_last=True)
    valLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=10)

    model_mod = importlib.import_module('riconv2_cls_v2_features2_largest')
    num_class = 97
    classifier = model_mod.get_model(num_class, 2, normal_channel=True)
    classifier.apply(inplace_relu).to(DEVICE)
    samples_per_class = np.array([max((tr_labels == c).sum(), 1) for c in range(num_class)])
    criterion = CBFocalLoss(samples_per_class, num_classes=num_class, beta=0.9999,
                            gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
                            weight_mode=args.weight_mode, weight_clip=args.weight_clip)
    criterion.class_weights = criterion.class_weights.to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=args.decay_rate)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epoch - args.warmup_epochs, eta_min=1e-5)
    log("params=%d" % count_parameters(classifier))

    best_acc, best_f1, swa_sd, swa_n = 0.0, 0.0, None, 0
    metrics_csv = exp_dir / 'logs' / 'val_metrics.csv'
    with open(metrics_csv, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['epoch', 'lr', 'train_acc', 'val_acc', 'val_bal', 'val_macf1',
                         'swa_acc', 'swa_bal', 'swa_macf1'])

    for epoch in range(args.epoch):
        classifier.train()
        if epoch < args.warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = args.learning_rate * (epoch + 1) / args.warmup_epochs
        else:
            cosine.step()
        correct, total = 0, 0
        for batch_id, (points, target) in enumerate(tqdm(trainDataLoader, total=len(trainDataLoader),
                                   desc="ep%d" % (epoch + 1), smoothing=0.9, leave=False)):
            optimizer.zero_grad()
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points).to(DEVICE)
            target = target.to(DEVICE)

            if np.random.rand() < args.cutmix_prob:
                perm = torch.randperm(points.shape[0])
                ratio = np.random.beta(args.cutmix_alpha, args.cutmix_alpha)
                ratio = max(ratio, 1.0 - ratio)
                mask = torch.rand(points.shape[0], points.shape[1], 1, device=points.device) > ratio
                points_mix = torch.where(mask, points[perm], points)
                t_oh = F.one_hot(target.long(), num_class).float()
                tp_oh = F.one_hot(target[perm].long(), num_class).float()
                mixed_target = t_oh * ratio + tp_oh * (1 - ratio)
                pred, _ = classifier(points_mix)
                if pred.dim() == 3:
                    pred = pred.mean(dim=1)
                loss = -(mixed_target * F.log_softmax(pred, 1)).sum(1).mean()
            else:
                pred, _ = classifier(points)
                if pred.dim() == 3:
                    target_2 = target.unsqueeze(-1).repeat(1, pred.shape[1]).view(-1)
                    loss = criterion(pred.contiguous().view(-1, num_class), target_2.long())
                    pred = pred.mean(dim=1)
                else:
                    loss = criterion(pred, target.long())
            correct += pred.argmax(1).eq(target.long()).sum().item()
            total += points.shape[0]
            loss.backward(); optimizer.step()
            if args.max_train_batches and batch_id + 1 >= args.max_train_batches:
                break
        train_acc = correct / total

        acc, bal, mf1 = evaluate(classifier, valLoader, args.max_val_batches)
        row = [epoch + 1, optimizer.param_groups[0]['lr'], train_acc, acc, bal, mf1, '', '', '']

        if acc >= best_acc:
            best_acc = acc
            torch.save({'epoch': epoch + 1, 'instance_acc': acc, 'class_acc': bal,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       exp_dir / 'checkpoints/best_model.pth')
        if mf1 >= best_f1:
            best_f1 = mf1
            torch.save({'epoch': epoch + 1, 'instance_acc': acc, 'macrof1': mf1,
                        'model_state_dict': classifier.state_dict()},
                       exp_dir / 'checkpoints/best_macrof1.pth')

        swa_vals = ('', '', '')
        if epoch + 1 >= args.swa_start:
            sd = {k: v.detach().cpu().float() for k, v in classifier.state_dict().items()}
            if swa_sd is None:
                swa_sd, swa_n = sd, 1
            else:
                swa_n += 1
                swa_sd = moving_average(swa_sd, sd, swa_n)
            swa_model = model_mod.get_model(num_class, 2, normal_channel=True)
            swa_model.load_state_dict(swa_sd, strict=True)
            swa_model.apply(inplace_relu).to(DEVICE)
            recalibrate_bn(swa_model, trainDataLoader,
                           n_batches=(min(30, args.max_train_batches) if args.max_train_batches else 30))
            swa_vals = evaluate(swa_model, valLoader, args.max_val_batches)
            torch.save({'epoch': epoch + 1, 'n_averaged': swa_n,
                        'model_state_dict': swa_model.state_dict()},
                       exp_dir / 'checkpoints/swa_model.pth')
            del swa_model; torch.cuda.empty_cache()
            row[6:] = swa_vals

        with open(metrics_csv, 'a', newline='') as fcsv:
            csv.writer(fcsv).writerow(row)
        log("ep%d lr=%.6f train=%.4f val acc=%.4f bal=%.4f macF1=%.4f | best acc=%.4f bestF1=%.4f%s"
            % (epoch + 1, optimizer.param_groups[0]['lr'], train_acc, acc, bal, mf1,
               best_acc, best_f1,
               (" | SWA acc=%.4f bal=%.4f macF1=%.4f" % swa_vals) if swa_vals[0] != '' else ''))

    log("done. best acc=%.4f best macroF1=%.4f" % (best_acc, best_f1))


if __name__ == '__main__':
    main(parse_args())
