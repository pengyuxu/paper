# Pretrained checkpoint downloads

## v3 best_model.pth (deep signal baseline / MC-FPS reference)

**Already committed** to this package at:
`replication/log/classification_shrec2025/riconv_large_shrec_run2_8k_largest/checkpoints/best_model.pth`

The dump scripts (`dump_mcfps.py`, `dump_clean.py`) locate it via
`BASE / "log/classification_shrec2025" / "riconv_large_shrec_run2_8k_largest" / "checkpoints" / "best_model.pth"`.

If you need to re-download: https://drive.contact.de/s/liDlJQmn4mcxeAD

## v5 seed2 swa_model.pth (final deep signal)

This checkpoint is produced by training v5 with seed 2 and is **not** hosted as
a separate download — it is written by the trainer into:
```
log/classification_shrec2025/riconv_large_v5_s2/checkpoints/swa_model.pth
```

Training command:
```bash
python 01_a_layer/train_v5.py --seed 2 --log_dir riconv_large_v5_s2
```

After training completes, copy (or symlink) the SWA checkpoint here:
```bash
cp log/classification_shrec2025/riconv_large_v5_s2/checkpoints/swa_model.pth \
   01_a_layer/checkpoints/swa_model.pth
```

The dump scripts (`dump_mcfps.py`, `dump_v5.py`) locate checkpoints via
`BASE / "log/classification_shrec2025" / <log_dir> / "checkpoints" / <ckpt>`,
so either keep the checkpoint in the trainer's output path or adjust the
`--log_dir` / `--ckpt` flags accordingly.
