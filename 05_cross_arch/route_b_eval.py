"""
Route B: cross-architecture seed-sensitivity replication on ModelNet40.

Runs K clean single FPS draws and one K-mixture pass through two public
PointNet++ architectures (SSG without normals, MSG with normals), using the
authors' released checkpoints.  Input point selection (first 1024 rows of the
normal_resampled files) is fixed, so all variation comes from the random
initial point of in-network farthest point sampling.

Metrics: per-draw acc/macro-F1 mean+-std, label-flip rate, single vs mixture
ECE, AURC, BALD mutual information.  numpy + torch only.
Run with: python route_b_eval.py --data_root ... --repo_root ...
"""
import argparse, json, os, sys
import numpy as np
import torch


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0, keepdims=True)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc[:, :3] ** 2, axis=1)))
    return (pc / m).astype(np.float32)


def macro_f1(pred, y, C):
    f1s = np.zeros(C)
    for c in range(C):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fp + fn == 0:
            continue
        f1s[c] = 2 * tp / (2 * tp + fp + fn)
    return float(np.mean(f1s[f1s > 0]))


def ece_score(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i > 0 else \
            (conf >= edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def aurc_score(p, y):
    conf, pred = p.max(1), p.argmax(1)
    order = np.argsort(-conf)
    correct = (pred == y).astype(np.float64)[order]
    risks = 1 - np.cumsum(correct) / (np.arange(len(y)) + 1)
    coverages = (np.arange(len(y)) + 1) / len(y)
    return float(np.trapezoid(risks, coverages))


def load_testset(root, use_normals):
    names = [l.rstrip() for l in open(os.path.join(root, 'modelnet40_shape_names.txt'))]
    classes = dict(zip(names, range(len(names))))
    ids = [l.rstrip() for l in open(os.path.join(root, 'modelnet40_test.txt'))]
    X, Y = [], []
    for k, sid in enumerate(ids):
        cat = '_'.join(sid.split('_')[:-1])
        pc = np.loadtxt(os.path.join(root, cat, sid + '.txt'), delimiter=',').astype(np.float32)
        pc = pc[:1024]                       # fixed input subset (authors' default)
        pc = pc_normalize(pc)
        X.append(pc[:, :6] if use_normals else pc[:, :3])
        Y.append(classes[cat])
        if (k + 1) % 250 == 0:
            print('  loaded %d/%d' % (k + 1, len(ids)))
    return np.stack(X), np.array(Y, dtype=np.int64)


def run_arch(tag, model, X, y, device, K):
    N, C = y.shape[0], X.shape[2]
    logp = np.zeros((K, N, 40), dtype=np.float16)
    bs = 64
    model.eval()
    with torch.no_grad():
        for k in range(K):
            torch.manual_seed(1000 + k)     # distinct, reproducible FPS draw
            for i0 in range(0, N, bs):
                i1 = i0 + min(bs, N - i0)
                xb = torch.from_numpy(X[i0:i1]).to(device).transpose(2, 1).contiguous()
                out, _ = model(xb)
                logp[k, i0:i1] = torch.log_softmax(out, 1).cpu().numpy().astype(np.float16)
            if (k + 1) % 5 == 0:
                print('  %s draw %d/%d' % (tag, k + 1, K))
    p = np.exp(logp.astype(np.float64))
    pk = p.mean(0)
    per_acc, per_f1, per_ece, per_aurc = [], [], [], []
    preds = np.zeros((K, N), dtype=np.int64)
    for k in range(K):
        pr = p[k].argmax(1)
        preds[k] = pr
        per_acc.append(float((pr == y).mean()))
        per_f1.append(macro_f1(pr, y, 40))
        per_ece.append(ece_score(p[k], y))
        per_aurc.append(aurc_score(p[k], y))
    flip = float(np.mean((preds != preds[0]).any(0)))
    mix_pred = pk.argmax(1)
    H = lambda pp: float(np.mean(-np.sum(pp * np.log(np.clip(pp, 1e-12, 1)), 1)))
    mi = H(pk) - float(np.mean([H(p[k]) for k in range(K)]))
    res = {
        'arch': tag, 'K': K,
        'single_acc_mean': float(np.mean(per_acc)), 'single_acc_std': float(np.std(per_acc)),
        'single_macrof1_mean': float(np.mean(per_f1)), 'single_macrof1_std': float(np.std(per_f1)),
        'mixture_acc': float((mix_pred == y).mean()),
        'mixture_macrof1': macro_f1(mix_pred, y, 40),
        'label_flip_rate': flip,
        'single_ece_mean': float(np.mean(per_ece)), 'single_ece_std': float(np.std(per_ece)),
        'mixture_ece': ece_score(pk, y),
        'single_aurc_mean': float(np.mean(per_aurc)), 'single_aurc_std': float(np.std(per_aurc)),
        'mixture_aurc': aurc_score(pk, y),
        'bald_mi_nats': mi,
    }
    return res, logp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--repo_root', required=True)
    ap.add_argument('--out_dir', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--K', type=int, default=25)
    args = ap.parse_args()
    assert np.__version__.split('.')[0] == '2', 'numpy>=2 required'
    device = torch.device('cuda')
    sys.path.insert(0, os.path.join(args.repo_root, 'models'))

    configs = [
        ('pointnet2_ssg', 'pointnet2_cls_ssg', False,
         'log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'),
        ('pointnet2_msg', 'pointnet2_cls_msg', True,
         'log/classification/pointnet2_msg_normals/checkpoints/best_model.pth'),
    ]
    summaries = []
    for tag, modname, use_normals, ckpt_rel in configs:
        print('== %s: loading data (use_normals=%s)' % (tag, use_normals))
        X, y = load_testset(args.data_root, use_normals)
        import importlib
        mod = importlib.import_module(modname)
        model = mod.get_model(40, normal_channel=use_normals).to(device)
        ck = torch.load(os.path.join(args.repo_root, ckpt_rel),
                        map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        print('== %s: running %d draws' % (tag, args.K))
        res, logp = run_arch(tag, model, X, y, device, args.K)
        summaries.append(res)
        np.savez(os.path.join(args.out_dir, 'routeb_%s_cache.npz' % tag),
                 logp=logp, y=y)
        print(json.dumps(res, indent=2))

    with open(os.path.join(args.out_dir, 'routeb_summary.json'), 'w') as f:
        json.dump(summaries, f, indent=2)
    print('saved routeb_summary.json')


if __name__ == '__main__':
    main()
