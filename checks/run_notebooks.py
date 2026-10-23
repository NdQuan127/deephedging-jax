"""
Executes notebooks end to end as a smoke test: DH_EPOCHS (default 1) epochs per training and no checkpoints.
Runs hold a machine-wide lock so that only one notebook executes at a time, each under a memory cap.
Executed copies go to --out; the source notebooks stay without outputs.

    python checks/run_notebooks.py notebooks/trainer.ipynb [...] --out /tmp/dh_nbout
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = "/tmp/deephedging_notebooks.lock"


def execute(path: str, out_dir: str, epochs: int, timeout: int) -> None:
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(path, as_version=4)
    os.environ.update(DH_EPOCHS=str(epochs), DH_CHECKPOINTS="")
    NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": os.path.dirname(os.path.abspath(path))}}).execute()
    os.makedirs(out_dir, exist_ok=True)
    nbformat.write(nb, os.path.join(out_dir, os.path.basename(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+")
    ap.add_argument("--out", default="/tmp/dh_nbout")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--memory", default="1500M")
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.inner:
        for nb in args.notebooks:
            execute(nb, args.out, args.epochs, args.timeout)
            print(f"ok {nb} -> {os.path.join(args.out, os.path.basename(nb))}", flush=True)
        return
    failed = []
    for nb in args.notebooks:
        cmd = ["flock", LOCK, "systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={args.memory}", "-p", "MemorySwapMax=0",
               "--nice=19", sys.executable, os.path.abspath(__file__), "--inner", "--out", args.out, "--epochs", str(args.epochs),
               "--timeout", str(args.timeout), os.path.abspath(nb)]
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode:
            failed.append(nb)
            print(f"FAIL {nb} (exit {r.returncode}{', memory cap hit' if r.returncode == 137 else ''})", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
