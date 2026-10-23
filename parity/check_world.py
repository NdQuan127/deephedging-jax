"""
Checks that the NumPy world port reproduces the original arrays exactly. Run it with the same
NumPy version as tf_reference.py: float32 exp and mean differ in the last bit across NumPy
releases, which also reorders paths with nearly equal terminal spots.

    python -m parity.check_world <ref_dir>
"""
import os
import sys
import numpy as np
from deephedging.world import WorldConfig, simulate
from .cases import WORLDS


def main():
    ref_dir, failures = sys.argv[1], []
    print(f"numpy {np.__version__}")
    for name, cfg in WORLDS.items():
        ref, w = np.load(os.path.join(ref_dir, f"world_{name}.npz")), simulate(WorldConfig(**cfg))
        arrays = {f"market_{k}": v for k, v in w.market.items()} | {f"features_{k}": v for k, v in w.features.items()}
        missing = sorted(set(ref.files) - set(arrays))
        for key, v in arrays.items():
            r = ref[key].astype(np.float32)
            r = r[:, :, None] if key.startswith("features_") and r.ndim == 2 else r
            diff = float(np.max(np.abs(v - r)))
            print(f"{'ok  ' if diff == 0. else 'FAIL'} {name} {key}: max abs diff {diff:.2e}")
            if diff != 0.:
                failures.append(f"{name}/{key}")
        if missing:
            failures.append(f"{name} missing {missing}")
            print(f"FAIL {name}: arrays not produced by the port: {missing}")
    print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
