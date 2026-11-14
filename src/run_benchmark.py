"""
run_benchmark.py -- main faithfulness benchmark + sensitivity sweeps.
(Prisha / Member 1)

Produces:
    benchmark_results.csv        per-image ROAD, Deletion, Insertion
    benchmark_summary.csv        mean ROAD + bootstrap 95% CI per method
    benchmark_tests.csv          paired Wilcoxon vs uml_weighted (Holm-adjusted)
    benchmark_plot.png
    layer_sweep.csv              which layers to fuse
    deletion_baseline_sweep.csv  blur / mean / black sensitivity

Run:  python src/run_benchmark.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from common import (init, load_sample, multi_layer_cams, fuse_uniform,
                    fuse_weighted, library_cam, road_score, perturbation_auc,
                    calibrate_fusion_weights, OUTPUT_DIR, SEED,
                    N_MAIN, N_AUC, N_SWEEP)


def holm(pvals):
    """Holm-Bonferroni correction for multiple comparisons."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adj[idx] = running
    return adj


def main():
    model, layers, _, voc = init("resnet50")

    weights, per_layer_deletion = calibrate_fusion_weights(model, layers, voc)
    print("per-layer deletion:", np.round(per_layer_deletion, 4))
    print("fusion weights [l2,l3,l4]:", np.round(weights, 4))

    # ---- main benchmark ----
    rows = []
    for i in range(N_MAIN):
        _, _, x = load_sample(voc, i)
        cams, cls = multi_layer_cams(model, layers, x)
        maps = {
            "gradcam": cams[-1],
            "gradcam++": library_cam("gradcam++", model, layers[-1], x, cls),
            "layercam": library_cam("layercam", model, layers[-1], x, cls),
            "scorecam": library_cam("scorecam", model, layers[-1], x, cls),
            "uml_uniform": fuse_uniform(cams),
            "uml_weighted": fuse_weighted(cams, weights),
        }
        for name, heat in maps.items():
            row = {"image": i, "method": name,
                   "road": road_score(model, x, heat, cls),
                   "deletion": np.nan, "insertion": np.nan}
            if i < N_AUC:
                row["deletion"] = perturbation_auc(
                    model, x, heat, cls, "deletion", "blur", 20)[0]
                row["insertion"] = perturbation_auc(
                    model, x, heat, cls, "insertion", "blur", 20)[0]
            rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{N_MAIN}")

    bench = pd.DataFrame(rows)
    bench.to_csv(OUTPUT_DIR / "benchmark_results.csv", index=False)

    # ---- bootstrap CIs ----
    road = bench.pivot(index="image", columns="method", values="road")
    rng = np.random.default_rng(SEED)
    summary = []
    for m in road.columns:
        v = road[m].dropna().to_numpy()
        boot = np.array([rng.choice(v, len(v), replace=True).mean()
                         for _ in range(3000)])
        summary.append({"method": m, "road_mean": v.mean(),
                        "ci_low": np.quantile(boot, 0.025),
                        "ci_high": np.quantile(boot, 0.975), "n": len(v)})
    summary = pd.DataFrame(summary).sort_values("road_mean", ascending=False)
    print(summary.round(4).to_string(index=False))

    # ---- paired tests vs our method, Holm-adjusted ----
    others = [m for m in road.columns if m != "uml_weighted"]
    pvals = [wilcoxon(road[m], road["uml_weighted"]).pvalue for m in others]
    tests = pd.DataFrame({"method": others, "wilcoxon_p": pvals,
                          "holm_p": holm(pvals)})
    print("\nvs uml_weighted:\n", tests.round(5).to_string(index=False))
    summary.to_csv(OUTPUT_DIR / "benchmark_summary.csv", index=False)
    tests.to_csv(OUTPUT_DIR / "benchmark_tests.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(range(len(summary)), summary.road_mean,
            yerr=[summary.road_mean - summary.ci_low,
                  summary.ci_high - summary.road_mean],
            capsize=4, color="steelblue")
    plt.xticks(range(len(summary)), summary.method, rotation=35, ha="right")
    plt.ylabel("ROAD (higher = better)")
    plt.title(f"Faithfulness on VOC2007 ({N_MAIN} images, bootstrap 95% CI)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "benchmark_plot.png", dpi=150)

    # ---- layer-set sweep ----
    sets = {"layer2": [0], "layer3": [1], "layer4": [2],
            "layer3+4": [1, 2], "layer2+3+4": [0, 1, 2]}
    rows = []
    for i in range(N_SWEEP):
        _, _, x = load_sample(voc, i)
        cams, cls = multi_layer_cams(model, layers, x)
        for name, sel in sets.items():
            rows.append({"image": i, "layer_set": name,
                         "road": road_score(
                             model, x, fuse_uniform(cams[np.asarray(sel)]), cls)})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUTPUT_DIR / "layer_sweep.csv", index=False)
    print("\nlayer sweep:\n",
          sweep.groupby("layer_set").road.mean().sort_values(ascending=False).round(4))

    # ---- deletion-baseline sensitivity ----
    rows = []
    for i in range(N_AUC):
        _, _, x = load_sample(voc, i)
        cams, cls = multi_layer_cams(model, layers, x)
        maps = {"gradcam": cams[-1], "uml_uniform": fuse_uniform(cams),
                "uml_weighted": fuse_weighted(cams, weights)}
        for name, heat in maps.items():
            for b in ["blur", "mean", "black"]:
                rows.append({"image": i, "method": name, "baseline": b,
                             "deletion": perturbation_auc(
                                 model, x, heat, cls, "deletion", b, 20)[0]})
    db = pd.DataFrame(rows)
    db.to_csv(OUTPUT_DIR / "deletion_baseline_sweep.csv", index=False)
    print("\ndeletion by baseline:\n",
          db.groupby(["method", "baseline"]).deletion.mean().unstack().round(4))


if __name__ == "__main__":
    main()
