"""Builds every manuscript figure from results/summary.json and results/surrogate.json."""
import json, os, pickle, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import config as C

S = json.load(open("results/summary.json"))
SG = json.load(open("results/surrogate.json")) if os.path.exists("results/surrogate.json") else None
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                     "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
OK = {"Random": "#999999", "Traditional": "#E69F00", "Periodised": "#56B4E9", "AdaptiveRule": "#009E73",
      "GA": "#D4C000", "GA-matched": "#B8A800", "DQN": "#0072B2", "DoubleDQN": "#CC79A7", "DuelingDQN": "#D55E00", "D3QN": "#000000"}
LAB = {"Random": "Random", "Traditional": "Traditional", "Periodised": "Block period.", "AdaptiveRule": "Adaptive rule",
       "GA": "GA (full)", "GA-matched": "GA (matched)", "DQN": "DQN", "DoubleDQN": "Double DQN", "DuelingDQN": "Dueling DQN", "D3QN": "D3QN"}


def box(ax, x, y, w, h, txt, fc, ec="#333333", fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p, q, txt=None, rad=0.0, col="#333333"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.0, color=col,
                                 connectionstyle=f"arc3,rad={rad}"))
    if txt:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.02, txt, ha="center", fontsize=7, style="italic")


# ---------------------------------------------------------------- Fig 1 framework
fig, ax = plt.subplots(figsize=(7.2, 4.0)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
box(ax, 0.01, 0.66, 0.18, 0.28, "College volleyball\nclass: 2 × 90-min\nsessions per week,\n16 weeks", "#FFF3D6", fs=7.5)
box(ax, 0.24, 0.70, 0.23, 0.24, "Monitoring layer\nsession-RPE load, jump count,\nwellness score, EWMA ACWR", "#E3F1FB", fs=7)
box(ax, 0.24, 0.36, 0.23, 0.28, "Testing layer (every 4 weeks)\nstep index, vertical jump,\ngrip, agility, plank;\nserve, pass, spike, block", "#E3F1FB", fs=7)
box(ax, 0.52, 0.45, 0.19, 0.42, "State $s_t$\n(37 features)\nprofile z-scores,\nacute / chronic load,\nfatigue, sex,\nplan history", "#EAF6EA", fs=7)
box(ax, 0.76, 0.45, 0.23, 0.42, "D3QN planner\n\ndueling V / A streams\n+ double-Q target\n\n$Q(s_t, a)$ for\n60 session types", "#FBE3E3", fs=7.5)
box(ax, 0.52, 0.03, 0.47, 0.25, "Session prescription $a_t$ =\nconditioning focus × technical focus × intensity\nAER/PLY/RES/AGL/CST × SERVE/PASS/SPIKE/BLOCK × L/M/H\nprinted as a weekly sheet the coach accepts or edits", "#F3EAF7", fs=7)
box(ax, 0.01, 0.03, 0.43, 0.25, "Reward $r_t$ = 10 × $\\Delta$VRI\n− 3 × max(0, ACWR − 1.5) − 3 × injury\nVRI = weighted z-scores of five tests\nand four skill ratings", "#FFF3D6", fs=7)
arrow(ax, (0.19, 0.82), (0.24, 0.82)); arrow(ax, (0.19, 0.72), (0.24, 0.55))
arrow(ax, (0.47, 0.82), (0.52, 0.76)); arrow(ax, (0.47, 0.50), (0.52, 0.56))
arrow(ax, (0.71, 0.66), (0.76, 0.66)); arrow(ax, (0.875, 0.45), (0.875, 0.28))
arrow(ax, (0.52, 0.155), (0.44, 0.155)); arrow(ax, (0.10, 0.28), (0.10, 0.66))
arrow(ax, (0.36, 0.28), (0.78, 0.45), rad=-0.15, col="#9b2c2c")
ax.text(0.56, 0.335, "learning signal (replay buffer)", fontsize=6.5, style="italic", color="#9b2c2c")
ax.text(0.115, 0.45, "next\nsession", fontsize=6.5, style="italic", color="#555555")
plt.tight_layout(); plt.savefig("figures/fig1_framework.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 2 network
fig, ax = plt.subplots(figsize=(7.2, 2.6)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
xs = [0.02, 0.17, 0.32, 0.47]
labs = ["Input\n37", "Dense 256\nReLU", "Dense 256\nReLU", "Dense 128\nReLU"]
for x, l in zip(xs, labs):
    box(ax, x, 0.38, 0.11, 0.30, l, "#E3F1FB")
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, (a + 0.11, 0.53), (b, 0.53))
box(ax, 0.63, 0.66, 0.12, 0.20, "Value\nV(s)  (1)", "#EAF6EA")
box(ax, 0.63, 0.16, 0.12, 0.20, "Advantage\nA(s,a)  (60)", "#FBE3E3")
arrow(ax, (0.58, 0.58), (0.63, 0.75)); arrow(ax, (0.58, 0.48), (0.63, 0.26))
box(ax, 0.80, 0.38, 0.17, 0.30, "Q = V + A − mean(A)\n(60 action values)", "#F3EAF7")
arrow(ax, (0.75, 0.76), (0.80, 0.58)); arrow(ax, (0.75, 0.26), (0.80, 0.48))
ax.text(0.50, 0.02, "Online network selects a* = argmax Q(s', a); target network (soft update, tau = 0.01) evaluates Q-(s', a*)",
        ha="center", fontsize=7.5, style="italic")
plt.tight_layout(); plt.savefig("figures/fig2_network.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 3 study design
fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
box(ax, 0.00, 0.58, 0.17, 0.36, "VolleySim\nparameters\n(Table 2): norms,\nceilings, transfer,\nreliability", "#FFF3D6", fs=7)
box(ax, 0.21, 0.78, 0.17, 0.16, "Training cohort\nn = 2,000", "#E3F1FB", fs=7)
box(ax, 0.21, 0.58, 0.17, 0.16, "Validation cohort\nn = 500", "#E3F1FB", fs=7)
box(ax, 0.42, 0.58, 0.17, 0.36, "Nine planners\n4 rule-based /\nstatic, GA,\n4 DRL variants\nx 5 seeds", "#FBE3E3", fs=7)
box(ax, 0.63, 0.58, 0.17, 0.36, "Held-out test\ncohort n = 1,000\n(500 M / 500 F)\nevery planner on\nthe same students", "#EAF6EA", fs=7)
box(ax, 0.83, 0.58, 0.17, 0.36, "Stress cohort\nn = 1,000\nparameters\nperturbed\n±30 %", "#EAF6EA", fs=7)
box(ax, 0.10, 0.03, 0.40, 0.33, "Primary and secondary analyses\npaired contrasts (Wilcoxon, Holm),\nHedges g with 95 % CI, bootstrap CIs,\nreliability check (ICC, SEM, MDC95),\nrobustness and ablation", "#F3EAF7", fs=7)
box(ax, 0.56, 0.03, 0.42, 0.33, "Virtual randomised trial\n3,000 draws of 2 × n students,\nsex-stratified random allocation,\nWelch t-test → empirical power;\nanalytic sample-size calculation", "#F3EAF7", fs=7)
arrow(ax, (0.17, 0.86), (0.21, 0.86)); arrow(ax, (0.17, 0.66), (0.21, 0.66))
arrow(ax, (0.38, 0.86), (0.42, 0.86)); arrow(ax, (0.38, 0.66), (0.42, 0.66))
arrow(ax, (0.59, 0.76), (0.63, 0.76)); arrow(ax, (0.80, 0.76), (0.83, 0.76))
arrow(ax, (0.68, 0.58), (0.40, 0.36)); arrow(ax, (0.74, 0.58), (0.77, 0.36)); arrow(ax, (0.91, 0.58), (0.45, 0.36), rad=0.1)
plt.tight_layout(); plt.savefig("figures/fig3_design.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 4 learning curves
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axs[0]
for v, c in S["curves"].items():
    ep = np.array(c["ep"]); m = np.array(c["mean"]); sd = np.array(c["sd"]); n = len(c["final"])
    h = 2.776 * sd / np.sqrt(n) if n == 5 else 1.96 * sd / np.sqrt(max(n, 1))
    ax.plot(ep, m, color=OK[v], lw=1.3, label=LAB[v]); ax.fill_between(ep, m - h, m + h, color=OK[v], alpha=0.15)
trad = [r for r in S["main"] if r["policy"] == "Traditional"][0]["ret"]
rule = [r for r in S["main"] if r["policy"] == "AdaptiveRule"][0]["ret"]
ax.axhline(trad, color=OK["Traditional"], ls="--", lw=1); ax.text(ep[-1], trad + 0.05, "Traditional (test)", ha="right", fontsize=7)
ax.axhline(rule, color=OK["AdaptiveRule"], ls=":", lw=1); ax.text(ep[-1], rule + 0.05, "Adaptive rule (test)", ha="right", fontsize=7)
ax.set_xlabel("Training episode (64 virtual students each)"); ax.set_ylabel("Validation return (mean, 95 % CI)")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), ncol=2, frameon=False); ax.set_title("(a) DRL learning curves, 5 seeds")
ax = axs[1]
ax.plot(np.arange(1, len(S["ga_hist"]) + 1), S["ga_hist"], color="#7a6a00", lw=1.3)
ax.set_xlabel("GA generation"); ax.set_ylabel("Best return (fresh batch each gen.)"); ax.set_title("(b) GA convergence")
plt.tight_layout(); plt.savefig("figures/fig4_learning.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 5 main outcome
order = ["Random", "Traditional", "Periodised", "AdaptiveRule", "GA-matched", "GA", "DQN", "DoubleDQN", "DuelingDQN", "D3QN"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axs[0]
rows = {r["policy"]: r for r in S["main"]}
for i, v in enumerate(order):
    pk = glob.glob(f"results/{v}.pkl") or sorted(glob.glob(f"results/{v}_s*.pkl"))
    vals = np.mean([pickle.load(open(p, "rb"))["test"]["dvri_meas"] for p in pk], axis=0)
    parts = ax.violinplot(vals, positions=[i], widths=0.8, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(OK[v]); b.set_alpha(0.35); b.set_edgecolor("none")
    r = rows[v]; lo, hi = r["dvri_ci"]
    ax.errorbar(i, r["dvri"], yerr=[[r["dvri"] - lo], [hi - r["dvri"]]], fmt="o", color="k", ms=3.5, capsize=3, lw=1)
ax.set_xticks(range(len(order))); ax.set_xticklabels([LAB[v] for v in order], rotation=35, ha="right")
ax.set_ylabel("Change in measured VRI (z units)"); ax.set_title("(a) Semester change, n = 1,000 test students")
comp = {c["policy"]: c for c in S["paired_vs_d3qn"]}
ymax = ax.get_ylim()[1]
for i, v in enumerate(order[:-1]):
    p = comp[v]["p_holm"]
    mk = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.text(i, ymax * 0.97, mk, ha="center", fontsize=7)
ax = axs[1]
inj = [rows[v]["inj_per100"] for v in order]; ov = [rows[v]["over"] for v in order]
y = np.arange(len(order))
ax.barh(y + 0.2, inj, 0.4, color="#D55E00", label="Injuries / 100 students")
ax.barh(y - 0.2, ov, 0.4, color="#0072B2", label="Sessions ACWR > 1.5 / student")
ax.set_yticks(y); ax.set_yticklabels([LAB[v] for v in order]); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.12), fontsize=7, ncol=1)
ax.set_title("(b) Load-safety indicators")
plt.tight_layout(); plt.savefig("figures/fig5_outcomes.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 6 forest plot
E = S["effects_d3qn_vs_trad"]
fig, ax = plt.subplots(figsize=(3.4, 3.2))
for i, e in enumerate(E[::-1]):
    lo, hi = e["g_ci"]; col = "#0072B2" if i >= 4 else "#D55E00"
    ax.errorbar(e["g"], i, xerr=[[e["g"] - lo], [hi - e["g"]]], fmt="s", color=col, ms=4, capsize=2.5, lw=1)
    mk = "***" if e["p_holm"] < 0.001 else "**" if e["p_holm"] < 0.01 else "*" if e["p_holm"] < 0.05 else "ns"
    ax.text(max(hi, 0) + 0.05, i, mk, va="center", fontsize=7)
ax.axvline(0, color="k", lw=0.8)
for xv in (0.2, 0.5, 0.8):
    ax.axvline(xv, color="grey", lw=0.4, ls=":")
NM = {"AER": "Step index", "VJ": "Vertical jump", "GRIP": "Grip", "AGI": "Agility", "CORE": "Plank", "SERVE": "Serve", "PASS": "Pass", "SPIKE": "Spike", "BLOCK": "Block"}
ax.set_yticks(range(len(E))); ax.set_yticklabels([NM[e["outcome"]] for e in E[::-1]])
ax.set_xlabel("Hedges g, D3QN vs Traditional (95 % CI)")
ax.set_title("Outcome-specific effects")
plt.tight_layout(); plt.savefig("figures/fig6_forest.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 7 ACWR + composition
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axs[0]
for v, d in S["acwr_traj"].items():
    ax.plot(np.arange(1, 33), d["mean"], color=OK[v], lw=1.3, label=LAB[v] + " (mean)")
    ax.plot(np.arange(1, 33), d["p90"], color=OK[v], lw=0.8, ls="--")
ax.axhspan(1.5, 2.2, color="#D55E00", alpha=0.08); ax.text(12, 1.88, "ACWR > 1.5 (elevated risk zone)", fontsize=7)
ax.set_ylim(0.8, 2.0); ax.set_xlabel("Session (Tue / Fri, weeks 1-16)"); ax.set_ylabel("EWMA ACWR")
ax.legend(frameon=False, fontsize=6.5, ncol=2); ax.set_title("(a) Load progression (dashed: 90th percentile)")
ax = axs[1]
blocks = S["policy_blocks"]
M = np.array([blocks[str(b)]["cond"] + blocks[str(b)]["tech"] + blocks[str(b)]["inten"] for b in range(4)]).T
im = ax.imshow(M, aspect="auto", cmap="YlGnBu", vmin=0, vmax=0.9); ax.grid(False)
ax.set_yticks(range(12)); ax.set_yticklabels(C.COND + C.TECH + ["Int-L", "Int-M", "Int-H"], fontsize=7)
ax.set_xticks(range(4)); ax.set_xticklabels(["Wk 1-4", "Wk 5-8", "Wk 9-12", "Wk 13-16"], fontsize=7)
for i in range(12):
    for j in range(4):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="k" if M[i, j] < 0.5 else "w")
ax.axhline(4.5, color="w", lw=2); ax.axhline(8.5, color="w", lw=2)
ax.set_title("(b) D3QN prescription shares"); plt.colorbar(im, ax=ax, fraction=0.04)
plt.tight_layout(); plt.savefig("figures/fig7_load_policy.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 8 power
P = S["power"]; mc = P["monte_carlo"]
ns = sorted(int(k) for k in mc)
fig, ax = plt.subplots(figsize=(3.4, 2.7))
ax.plot(ns, [mc[str(n)]["power"] for n in ns], "o-", color="k", ms=3.5, lw=1.2, label="Monte-Carlo virtual RCT")
nn = np.arange(4, 61)
from scipy import stats as st
def pwr(d, n):
    df = 2 * n - 2; nc = d * np.sqrt(n / 2); tc = st.t.ppf(0.975, df)
    return 1 - st.nct.cdf(tc, df, nc) + st.nct.cdf(-tc, df, nc)
for key, col, lab in [("vs_traditional", OK["Traditional"], "Analytic, vs Traditional"),
                      ("vs_rule", OK["AdaptiveRule"], "Analytic, vs Adaptive rule")]:
    ax.plot(nn, [pwr(abs(P[key]["g"]), n) for n in nn], color=col, lw=1.3, label=f"{lab} (g = {P[key]['g']:.2f})")
ax.axhline(0.8, color="grey", ls=":", lw=0.8); ax.axvline(20, color="grey", ls="--", lw=0.8)
ax.text(21, 0.62, "n = 20\nper arm", fontsize=7)
ax.set_xlabel("Students per arm"); ax.set_ylabel("Power (two-sided, alpha = 0.05)"); ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=6.5, loc="lower right"); ax.set_title("Sample size for a field trial")
plt.tight_layout(); plt.savefig("figures/fig8_power.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 9 predictions vs outcomes
if SG is not None:
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    vc = S["value_cal"]; ax = axs[0]
    ax.scatter(vc["v0"], vc["G"], s=4, alpha=0.4, color="#0072B2", edgecolor="none")
    lim = [min(min(vc["v0"]), min(vc["G"])), max(max(vc["v0"]), max(vc["G"]))]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("Predicted value $V(s_0)$"); ax.set_ylabel("Realised discounted return")
    ax.set_title(f"(a) D3QN value, r = {vc['r']:.2f}")
    ax = axs[1]; o = SG["oof_sample"]
    ax.scatter(o["p"], o["y"], s=4, alpha=0.4, color="#009E73", edgecolor="none")
    lim = [min(min(o["p"]), min(o["y"])), max(max(o["p"]), max(o["y"]))]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("XGBoost predicted change in VRI"); ax.set_ylabel("Simulated change in VRI")
    ax.set_title(f"(b) Surrogate, R$^2$ = {SG['r2']:.2f}")
    ax = axs[2]; imp = SG["importance"][:10][::-1]
    ax.barh(range(len(imp)), [i["mean"] for i in imp], xerr=[i["sd"] for i in imp], color="#CC79A7", capsize=2)
    ax.set_yticks(range(len(imp))); ax.set_yticklabels([i["feature"].replace("early_dVRI_wk4", "early VRI change (wk 4)").replace("_", " ") for i in imp], fontsize=6.5)
    ax.set_xlabel("Permutation importance (drop in R$^2$)"); ax.set_title("(c) Drivers of change")
    plt.tight_layout(); plt.savefig("figures/fig9_pred_obs.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Fig 10 robustness + ablation
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
ax = axs[0]
rob = {r["policy"]: r for r in S["robustness"]}
for i, v in enumerate(order):
    a = rows[v]["dvri"]; b = rob[v]["dvri"]; lo, hi = rob[v]["dvri_ci"]
    ax.bar(i - 0.2, a, 0.4, color=OK[v], edgecolor="k", lw=0.5)
    ax.bar(i + 0.2, b, 0.4, color=OK[v], alpha=0.45, edgecolor="k", lw=0.5, hatch="//")
    ax.errorbar(i + 0.2, b, yerr=[[b - lo], [hi - b]], color="k", capsize=2, lw=0.8)
ax.set_xticks(range(len(order))); ax.set_xticklabels([LAB[v] for v in order], rotation=35, ha="right", fontsize=7)
ax.set_ylabel("Mean change in VRI"); ax.set_title("(a) Nominal (solid) vs perturbed (hatched)")
ax = axs[1]
A = S["ablation"]
ax.bar(range(len(A)), [a["ret"] for a in A], yerr=[a["ret_sd"] for a in A], color=["#000000", "#D55E00", "#0072B2", "#999999", "#009E73"][:len(A)], capsize=3)
for i, a in enumerate(A):
    ax.text(i, a["ret"] * 0.5, f"inj {a['inj_per100']:.1f}\nover {a['over']:.2f}", ha="center", color="w", fontsize=6.5)
ax.set_xticks(range(len(A))); ax.set_xticklabels(["Full D3QN", "No ACWR\npenalty", "No load\nstate", "Small\nnetwork", "+ safety\nshield"][:len(A)], fontsize=7)
ax.set_ylabel("Test return (mean ± SD, 3 seeds)"); ax.set_title("(b) Ablation")
plt.tight_layout(); plt.savefig("figures/fig10_robust_ablation.png", bbox_inches="tight"); plt.close()
print("figures written")
