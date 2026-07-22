#!/usr/bin/env python3
"""Select the top-X generations of an optimization batch by fitness, then emit a
temp folder of their candidate configs with a chosen set of synaptic weights
scaled by a constant factor (e.g. a simulated drug block).

Batch layout assumed (one best-candidate trial per generation):

    <batch_dir>/
      gen_<n>/
        trial_<n>_cfg.json       # candidate config (param values, under "simConfig")
        trial_<n>_fitness.json   # has top-level "fitness" (lower is better)

Generalises the earlier GABA-only variant: pass --weight-keys to choose which
simConfig weight params to scale and --label to name the outputs. Examples:

  # GABA -> 10% (inhibitory block)
  --weight-keys weightIE_GABA,weightII_GABA --scale 0.10 --label gaba

  # AMPA + NMDA -> 1% (AP5/NBQX glutamate block)
  --weight-keys weightEE_AMPA,weightEI_AMPA,weightEE_NMDA,weightEI_NMDA \
      --scale 0.01 --label ampa_nmda
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

# Default keys reproduce the original GABA behaviour for backward compatibility.
DEFAULT_WEIGHT_KEYS = ("weightIE_GABA", "weightII_GABA")


def find_generations(batch_dir):
    """Return a list of dicts {gen, cfg, fitness} for gens that have both a
    candidate cfg and a computed fitness, sorted by fitness ascending."""
    gens = []
    for gen_dir in glob.glob(os.path.join(batch_dir, "gen_*")):
        m = re.search(r"gen_(\d+)$", gen_dir)
        if not m:
            continue
        n = int(m.group(1))
        cfg = os.path.join(gen_dir, "trial_%d_cfg.json" % n)
        fit = os.path.join(gen_dir, "trial_%d_fitness.json" % n)
        if not (os.path.isfile(cfg) and os.path.isfile(fit)):
            continue
        try:
            with open(fit) as f:
                fitness = json.load(f)["fitness"]
        except (ValueError, KeyError, OSError) as e:
            print("  ! skipping gen_%d (bad fitness file: %s)" % (n, e))
            continue
        gens.append({"gen": n, "cfg": cfg, "fitness": fitness})
    gens.sort(key=lambda g: g["fitness"])
    return gens


def scale_weights(cfg_data, scale, keys):
    """Scale the given weight keys in-place. Returns list of (key, old, new)."""
    sim = cfg_data.get("simConfig", cfg_data)
    changes = []
    for key in keys:
        if key in sim:
            old = sim[key]
            new = old * scale
            sim[key] = new
            changes.append((key, old, new))
    return changes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("batch_dir", help="Path to the batch_<date>_<tag> directory")
    p.add_argument("--top", type=int, default=20,
                   help="Number of top generations (by fitness) to select (default: 20)")
    p.add_argument("--scale", type=float, default=0.10,
                   help="Factor to multiply the chosen weights by (default: 0.10)")
    p.add_argument("--weight-keys", default=",".join(DEFAULT_WEIGHT_KEYS),
                   help="Comma-separated simConfig weight keys to scale "
                        "(default: %(default)s)")
    p.add_argument("--label", default="gaba",
                   help="Short label used in output dir/file names (default: gaba)")
    p.add_argument("--output-dir", default=None,
                   help="Where to write the modified configs "
                        "(default: <batch_dir>/<label>_scaled_top_<top>_<pct>pct)")
    args = p.parse_args()

    batch_dir = os.path.abspath(args.batch_dir)
    if not os.path.isdir(batch_dir):
        sys.exit("Batch directory not found: %s" % batch_dir)

    weight_keys = [k.strip() for k in args.weight_keys.split(",") if k.strip()]
    pct = ("%g" % (args.scale * 100)).replace(".", "_")
    out_dir = args.output_dir or os.path.join(
        batch_dir, "%s_scaled_top_%d_%spct" % (args.label, args.top, pct))

    print("Batch:   %s" % batch_dir)
    print("Keys:    %s" % ", ".join(weight_keys))
    print("Scale:   %g  (-> %g%%)" % (args.scale, args.scale * 100))
    print("Top:     %d generations" % args.top)
    print("Output:  %s\n" % out_dir)

    gens = find_generations(batch_dir)
    if not gens:
        sys.exit("No generations with both a cfg and fitness.json were found.")
    print("Found %d generations with fitness; selecting best %d.\n"
          % (len(gens), args.top))

    selected = gens[:args.top]

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    manifest = []
    for rank, g in enumerate(selected, 1):
        with open(g["cfg"]) as f:
            cfg_data = json.load(f)
        changes = scale_weights(cfg_data, args.scale, weight_keys)
        out_name = "trial_%d_%s%spct_cfg.json" % (g["gen"], args.label, pct)
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w") as f:
            json.dump(cfg_data, f, indent=2)
        manifest.append({
            "rank": rank,
            "gen": g["gen"],
            "fitness": g["fitness"],
            "source_cfg": g["cfg"],
            "output_cfg": out_path,
            "weight_changes": [
                {"key": k, "old": old, "new": new} for (k, old, new) in changes
            ],
        })
        chg = ", ".join("%s %.4g->%.4g" % (k, o, n) for k, o, n in changes) or "(no keys matched!)"
        print("  #%-3d gen_%-4d fitness=%-10.4f  %s" % (rank, g["gen"], g["fitness"], chg))

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "batch_dir": batch_dir,
            "scale": args.scale,
            "weight_keys": weight_keys,
            "label": args.label,
            "top": args.top,
            "n_selected": len(selected),
            "selected": manifest,
        }, f, indent=2)

    print("\nWrote %d configs + manifest.json to:\n  %s" % (len(selected), out_dir))


if __name__ == "__main__":
    main()
