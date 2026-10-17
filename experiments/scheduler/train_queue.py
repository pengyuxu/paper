"""Persistent dual-GPU training queue for the 12 deep runs (6 per version).

Runs as ONE long-lived background process (via the host's background-task
facility, NOT setsid — sandboxed sessions lose /dev/nvidia* on daemonize).
Child python processes inherit GPU access from this process.

State file: experiments/results/queue_state.json (atomic rewrite).
Each run: max 2 attempts; the 2nd attempt adds --resume (train_v5.py resumes
from best_model.pth). Logs: logs_setup/queue/<version>__<name>.log.

Usage:
  python3 -m experiments.scheduler.train_queue --only paper,v5_s1   # filter
  python3 -m experiments.scheduler.train_queue --status              # print state
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STATE = REPO / "experiments" / "results" / "queue_state.json"
QLOG = REPO / "logs_setup" / "queue"
RUN_ENV = REPO / "experiments" / "run_env.sh"
NEWCODE = REPO / "newcode"

# (version, run_name, extra train_v5.py args)
# Barisin rows use a separate driver (experiments.methods.barisin_run).
RUNS = [
    ("paper",   "v5_s1",          ["--seed", "1"]),
    ("paper",   "v5_s2",          ["--seed", "2"]),
    ("paper",   "v3_noSynth_s2",  ["--seed", "2", "--no_synth"]),
    ("paper",   "abl_focalg2_s2", ["--seed", "2", "--focal_gamma", "2.0"]),
    ("paper",   "abl_noCutMix_s2",["--seed", "2", "--cutmix_prob", "0"]),
    ("paper",   "abl_cbbeta_s2",  ["--seed", "2", "--weight_mode", "beta"]),
    ("updated", "v5_s1",          ["--seed", "1"]),
    ("updated", "v5_s2",          ["--seed", "2"]),
    ("updated", "v3_noSynth_s2",  ["--seed", "2", "--no_synth"]),
    ("updated", "abl_focalg2_s2", ["--seed", "2", "--focal_gamma", "2.0"]),
    ("updated", "abl_noCutMix_s2",["--seed", "2", "--cutmix_prob", "0"]),
    ("updated", "abl_cbbeta_s2",  ["--seed", "2", "--weight_mode", "beta"]),
    # Barisin et al., updated version, official recipe (seed pinned, see docstring)
    ("updated", "barisin_run1",   ["__barisin__", "1"]),
    ("updated", "barisin_run2",   ["__barisin__", "2"]),
]


def default_state():
    return {"runs": [{"version": v, "name": n, "status": "pending",
                      "attempts": 0, "gpu": None, "pid": None,
                      "started": None, "ended": None, "rc": None,
                      "note": ""}
                     for v, n, _ in RUNS]}


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return default_state()


def save_state(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False))
    tmp.rename(STATE)


def build_cmd(run, gpu, attempt):
    v, n, extra = run
    if extra and extra[0] == "__barisin__":
        # wrapper auto-resumes from its own best_model.pth; cwd = REPO
        return [str(RUN_ENV), "python3", "-m", "experiments.methods.barisin_run",
                "--mode", "train", "--run", extra[1],
                "--version", v, "--gpu", str(gpu), "--log_dir", n]
    cmd = [str(RUN_ENV), "python3", "train_v5.py",
           "--version", v, "--gpu", str(gpu), "--log_dir", n] + extra
    ckpt = REPO / "work" / v / "log" / "classification_shrec2025" / n / "checkpoints" / "best_model.pth"
    if attempt >= 2 and ckpt.exists():
        cmd.append("--resume")
    return cmd


def cmd_cwd(run):
    return REPO if (run[2] and run[2][0] == "__barisin__") else NEWCODE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list version,name (run both if one token)")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    st = load_state()
    if args.status:
        for r in st["runs"]:
            print(f"{r['version']:8s} {r['name']:18s} {r['status']:8s} "
                  f"att={r['attempts']} rc={r['rc']} {r['note']}")
        return

    only = set(x.strip() for x in args.only.split(",") if x.strip())
    gpus = [int(g) for g in args.gpus.split(",")]
    QLOG.mkdir(parents=True, exist_ok=True)

    # Migrate: add records for RUNS introduced after the state file was written.
    have = {(r["version"], r["name"]) for r in st["runs"]}
    for v, n, _ in RUNS:
        if (v, n) not in have:
            st["runs"].append({"version": v, "name": n, "status": "pending",
                               "attempts": 0, "gpu": None, "pid": None,
                               "started": None, "ended": None, "rc": None,
                               "note": "added by migration"})
            print(f"[{datetime.now():%H:%M:%S}] migrated new run {v}/{n}",
                  flush=True)

    # On queue startup no child can exist: any leftover "running" record is a
    # stale lock from a previous queue process -> requeue (attempts preserved,
    # attempt 2 will pass --resume).
    requeued = [r for r in st["runs"] if r["status"] == "running"]
    for r in requeued:
        r["status"] = "pending"; r["gpu"] = None; r["pid"] = None
        r["note"] = "requeued after queue restart"
    if requeued:
        save_state(st)
        print(f"[{datetime.now():%H:%M:%S}] requeued %d stale run(s)"
              % len(requeued), flush=True)

    # index active records by (v,n)
    active = {}   # gpu -> (record, popen, logfh)
    while True:
        # reap
        for g in list(active):
            rec, proc, fh = active[g]
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            rec["rc"] = rc; rec["ended"] = datetime.now().isoformat(timespec="seconds")
            rec["pid"] = None; rec["gpu"] = None
            if rc == 0:
                rec["status"] = "done"; rec["note"] = ""
            elif rec["attempts"] >= 2:
                rec["status"] = "failed"; rec["note"] = "2 attempts exhausted"
            else:
                rec["status"] = "pending"; rec["note"] = "retry queued"
            active.pop(g)
            save_state(st)
            print(f"[{datetime.now():%H:%M:%S}] gpu{g} {rec['version']}/{rec['name']} "
                  f"rc={rc} -> {rec['status']}", flush=True)

        free = [g for g in gpus if g not in active]
        for g in free:
            cand = None
            for run, rec in zip(RUNS, st["runs"]):
                v, n, _ = run
                if rec["status"] != "pending":
                    continue
                if only and not (v in only or n in only):
                    continue
                # don't run same run_name on both GPUs simultaneously
                if any(rr["name"] == n and rr["status"] == "running"
                       for rr in st["runs"]):
                    continue
                cand = (run, rec); break
            if cand is None:
                continue
            run, rec = cand
            rec["attempts"] += 1
            rec["status"] = "running"; rec["gpu"] = g
            rec["started"] = datetime.now().isoformat(timespec="seconds")
            cmd = build_cmd(run, g, rec["attempts"])
            logf = open(QLOG / f"{run[0]}__{run[1]}.log", "ab")
            logf.write(f"\n===== attempt {rec['attempts']} gpu{g} "
                       f"{datetime.now().isoformat()} =====\n".encode())
            logf.flush()
            proc = subprocess.Popen(cmd, cwd=str(cmd_cwd(run)), stdout=logf,
                                    stderr=subprocess.STDOUT)
            rec["pid"] = proc.pid
            active[g] = (rec, proc, logf)
            save_state(st)
            print(f"[{datetime.now():%H:%M:%S}] START gpu{g} {run[0]}/{run[1]} "
                  f"attempt {rec['attempts']} pid={proc.pid}", flush=True)

        if not active and all(r["status"] in ("done", "failed")
                              for r in st["runs"]
                              if _allowed(r, only)):
            break
        time.sleep(20)

    save_state(st)
    n_fail = sum(r["status"] == "failed" for r in st["runs"] if _allowed(r, only))
    print("QUEUE FINISHED. failures=%d" % n_fail)
    sys.exit(1 if n_fail else 0)


def _allowed(rec, only):
    if not only:
        return True
    return rec["version"] in only or rec["name"] in only


if __name__ == "__main__":
    main()
