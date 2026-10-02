"""Statistical analysis of the stored experiment outcomes.
Writes results/summary.json (every number quoted in the manuscript) and CSV tables."""
import os, json, pickle, glob, time
import numpy as np
import pandas as pd
from scipy import stats
import torch
import config as C
from volleysim import Cohort, VolleySim, run_policy, OBS_DIM
from baselines import (random_policy, traditional_policy, periodised_policy,
                       rule_policy, static_policy)
from agents import QNet

R = "results"
rng = np.random.default_rng(2026)
OUT = {}
BASE = ["Random", "Traditional", "Periodised", "AdaptiveRule", "GA-matched", "GA"]
DRL = ["DQN", "DoubleDQN", "DuelingDQN", "D3QN"]
OUTC = C.QUAL + C.TECH


def load(name):
    with open(f"{R}/{name}.pkl", "rb") as f:
        return pickle.load(f)


def seeds_of(v):
    return sorted(glob.glob(f"{R}/{v}_s*.pkl"))


def agg(v, split="test"):
    """Per-student outcomes; DRL variants averaged over training seeds."""
    if v in BASE:
        d = load(v)[split]
        return d, [d]
    runs = [pickle.load(open(p, "rb"))[split] for p in seeds_of(v)]
    keys = ["dvri_meas", "dvri_true", "ret", "injuries", "over", "missed", "jumps", "dP", "dS"]
    d = {k: np.mean([r[k] for r in runs], axis=0) for k in keys}
    return d, runs


def boot_ci(x, n=4000, f=np.mean):
    idx = rng.integers(0, len(x), (n, len(x)))
    b = f(x[idx], axis=1) if f is np.mean else np.array([f(x[i]) for i in idx])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp
    return d * (1 - 3 / (4 * (na + nb) - 9))


def g_ci(g, na, nb):
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb)))
    return g - 1.96 * se, g + 1.96 * se


# ---------------------------------------------------------------------------
pols = BASE + DRL
data = {v: agg(v) for v in pols}
test_co = Cohort(1000, C.SEED_COHORT_TEST)
male = test_co.sex == 0

# 1. main performance table ---------------------------------------------------
rows = []
for v in pols:
    d, runs = data[v]
    seed_means = [r["dvri_meas"].mean() for r in runs]
    rows.append(dict(
        policy=v, n_runs=len(runs),
        ret=float(d["ret"].mean()), ret_sd_seeds=float(np.std([r["ret"].mean() for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
        dvri=float(d["dvri_meas"].mean()), dvri_ci=boot_ci(d["dvri_meas"]),
        dvri_sd=float(d["dvri_meas"].std(ddof=1)),
        dvri_true=float(d["dvri_true"].mean()),
        dvri_seed_sd=float(np.std(seed_means, ddof=1)) if len(runs) > 1 else 0.0,
        inj_per100=float(100 * d["injuries"].mean()),
        over=float(d["over"].mean()), missed=float(d["missed"].mean()),
        jumps=float(d["jumps"].mean()),
        dvri_male=float(d["dvri_meas"][male].mean()), dvri_female=float(d["dvri_meas"][~male].mean()),
    ))
main = pd.DataFrame(rows)
main.to_csv(f"{R}/table_main.csv", index=False)
OUT["main"] = rows

# 2. paired comparisons vs D3QN -----------------------------------------------
ref = data["D3QN"][0]["dvri_meas"]
comp = []
for v in pols:
    if v == "D3QN":
        continue
    x = data[v][0]["dvri_meas"]
    diff = ref - x
    w = stats.wilcoxon(ref, x)
    comp.append(dict(policy=v, diff=float(diff.mean()), ci=boot_ci(diff),
                     dz=float(diff.mean() / diff.std(ddof=1)), p=float(w.pvalue),
                     rel=float(100 * diff.mean() / x.mean())))
padj = holm([c["p"] for c in comp])
for c, pa in zip(comp, padj):
    c["p_holm"] = float(pa)
OUT["paired_vs_d3qn"] = comp
pd.DataFrame(comp).to_csv(f"{R}/table_paired.csv", index=False)

# 3. outcome-specific effects: D3QN vs Traditional (between-subject g + paired) --
eff = []
dT, dD = data["Traditional"][0], data["D3QN"][0]
for k, nm in enumerate(OUTC):
    a = dD["dP"][:, k] if k < 5 else dD["dS"][:, k - 5]
    b = dT["dP"][:, k] if k < 5 else dT["dS"][:, k - 5]
    if k < 5 and C.LOWER_IS_BETTER[k]:
        a, b = -a, -b
    g = hedges_g(a, b); lo, hi = g_ci(g, len(a), len(b))
    diff = a - b
    eff.append(dict(outcome=nm, unit=(C.UNITS[k] if k < 5 else "pts"),
                    trad=float(b.mean()), d3qn=float(a.mean()),
                    diff=float(diff.mean()), diff_ci=boot_ci(diff), g=float(g), g_ci=(float(lo), float(hi)),
                    p=float(stats.wilcoxon(a, b).pvalue)))
pa = holm([e["p"] for e in eff])
for e, p in zip(eff, pa):
    e["p_holm"] = float(p)
OUT["effects_d3qn_vs_trad"] = eff
pd.DataFrame(eff).to_csv(f"{R}/table_effects.csv", index=False)
g_main = hedges_g(dD["dvri_meas"], dT["dvri_meas"])
OUT["g_main"] = float(g_main); OUT["g_main_ci"] = g_ci(g_main, 1000, 1000)
g_rule = hedges_g(dD["dvri_meas"], data["AdaptiveRule"][0]["dvri_meas"])
g_ga = hedges_g(dD["dvri_meas"], data["GA"][0]["dvri_meas"])
OUT["g_vs_rule"], OUT["g_vs_ga"] = float(g_rule), float(g_ga)

# 4. power analysis ---------------------------------------------------------------
def power_t(d, n, alpha=0.05):
    df = 2 * n - 2; nc = d * np.sqrt(n / 2)
    tc = stats.t.ppf(1 - alpha / 2, df)
    return float(1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))


def n_for(d, pw):
    n = 2
    while power_t(d, n) < pw:
        n += 1
    return n


def mde(n, pw=0.8):
    lo, hi = 0.01, 3
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if power_t(mid, n) < pw else (lo, mid)
    return hi


pw = {}
for name, g in [("vs_traditional", g_main), ("vs_rule", g_rule), ("vs_ga", g_ga)]:
    pw[name] = dict(g=float(g), n80=n_for(abs(g), 0.8), n90=n_for(abs(g), 0.9),
                    power_n20=power_t(abs(g), 20))
pw["mde_n20"] = mde(20)
# Monte-Carlo virtual RCTs (n = 20 per arm, sex-stratified)
idx_m, idx_f = np.where(male)[0], np.where(~male)[0]
ns = [5, 10, 15, 20, 25, 30, 40, 50, 60]
mc = {}
for n_arm in ns:
    hits = 0; diffs = []
    for _ in range(3000):
        half = n_arm // 2
        sm = rng.choice(idx_m, 2 * half, replace=False); sf = rng.choice(idx_f, 2 * (n_arm - half), replace=False)
        arm1 = np.r_[sm[:half], sf[:n_arm - half]]; arm0 = np.r_[sm[half:], sf[n_arm - half:]]
        y1, y0 = dD["dvri_meas"][arm1], dT["dvri_meas"][arm0]
        p = stats.ttest_ind(y1, y0, equal_var=False).pvalue
        hits += p < 0.05; diffs.append(y1.mean() - y0.mean())
    mc[n_arm] = dict(power=hits / 3000, analytic=power_t(abs(g_main), n_arm),
                     diff_ci=(float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))))
pw["monte_carlo"] = mc
OUT["power"] = pw

# 5. reliability check of the measurement model ------------------------------------
env = VolleySim(test_co); env.reset()
rel = []
sd_e = C.NORM_SD[test_co.sex] * np.sqrt((1 - C.ICC_TARGET) / C.ICC_TARGET)
lat = env.capacity()
t1 = lat + rng.normal(0, 1, lat.shape) * sd_e
t2 = lat + rng.normal(0, 1, lat.shape) * sd_e
S = env.skills(); sds = C.SKILL_SD * np.sqrt((1 - C.ICC_SKILL) / C.ICC_SKILL)
s1 = S + rng.normal(0, sds, S.shape); s2 = S + rng.normal(0, sds, S.shape)


def icc31(x, y):
    X = np.column_stack([x, y]); n, k = X.shape
    gm = X.mean(); msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    mse = (((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()) / ((n - 1) * (k - 1))
    icc = (msr - mse) / (msr + (k - 1) * mse)
    F = msr / mse; df1, df2 = n - 1, (n - 1) * (k - 1)
    fl = F / stats.f.ppf(0.975, df1, df2); fu = F * stats.f.ppf(0.975, df2, df1)
    return icc, (fl - 1) / (fl + k - 1), (fu - 1) / (fu + k - 1)


for k, nm in enumerate(OUTC):
    for sx, lab in [(male, "M"), (~male, "F")]:
        if k < 5:
            x, y, tgt = t1[sx, k], t2[sx, k], C.ICC_TARGET[k]
        else:
            x, y, tgt = s1[sx, k - 5], s2[sx, k - 5], C.ICC_SKILL
        icc, lo, hi = icc31(x, y)
        sem = np.sqrt(np.var(np.r_[x, y], ddof=1) * (1 - icc))
        rel.append(dict(test=nm, sex=lab, target=float(tgt), icc=float(icc), lo=float(lo), hi=float(hi),
                        sem=float(sem), mdc95=float(1.96 * np.sqrt(2) * sem)))
OUT["reliability"] = rel
pd.DataFrame(rel).to_csv(f"{R}/table_reliability.csv", index=False)

# 6. robustness under simulator misspecification -----------------------------------
rob = []
for v in pols:
    d, runs = agg(v, "stress")
    rob.append(dict(policy=v, dvri=float(d["dvri_meas"].mean()), dvri_ci=boot_ci(d["dvri_meas"]),
                    inj_per100=float(100 * d["injuries"].mean()), over=float(d["over"].mean())))
OUT["robustness"] = rob

# 7. ablations ---------------------------------------------------------------------
abl = []
for an in ["D3QN", "D3QN-noACWRpenalty", "D3QN-noLoadState", "D3QN-small", "D3QN-shield"]:
    files = seeds_of(an)[:3]
    runs = [pickle.load(open(p, "rb"))["test"] for p in files]
    if not runs:
        continue
    abl.append(dict(variant=an, n=len(runs),
                    ret=float(np.mean([r["ret"].mean() for r in runs])),
                    ret_sd=float(np.std([r["ret"].mean() for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
                    dvri=float(np.mean([r["dvri_meas"].mean() for r in runs])),
                    inj_per100=float(100 * np.mean([r["injuries"].mean() for r in runs])),
                    over=float(np.mean([r["over"].mean() for r in runs]))))
OUT["ablation"] = abl

# 8. learning curves ---------------------------------------------------------------
curves = {}
for v in DRL:
    cs = np.array([np.load(f"{R}/curve_{v}_s{s}.npy") for s in C.RL_SEEDS if os.path.exists(f"{R}/curve_{v}_s{s}.npy")])
    curves[v] = dict(ep=cs[0, :, 0].tolist(), mean=cs[:, :, 1].mean(0).tolist(),
                     sd=cs[:, :, 1].std(0, ddof=1).tolist() if len(cs) > 1 else [0] * cs.shape[1],
                     final=cs[:, -1, 1].tolist())
OUT["curves"] = curves
ga = np.load(f"{R}/ga_plan.npz")
OUT["ga_hist"] = ga["hist"].tolist(); OUT["ga_secs"] = float(ga["secs"])
OUT["ga_plan"] = ga["plan"].tolist()
gm = np.load(f"{R}/ga_matched_plan.npz"); OUT["ga_matched_plan"] = gm["plan"].tolist(); OUT["ga_matched_hist"] = gm["hist"].tolist()
OUT["tuning"] = json.load(open(f"{R}/tuning.json")) if os.path.exists(f"{R}/tuning.json") else {}

# 9. policy composition of D3QN (seed-pooled) ----------------------------------
acts = np.concatenate([pickle.load(open(p, "rb"))["test"]["actions"] for p in seeds_of("D3QN")], axis=1)
comp_tab = {}
for blk in range(4):
    a = acts[blk * 8:(blk + 1) * 8].ravel(); a = a[a >= 0]
    c, t, i = C.decode(a)
    comp_tab[blk] = dict(cond=np.bincount(c, minlength=5) / len(a), tech=np.bincount(t, minlength=4) / len(a),
                         inten=np.bincount(i, minlength=3) / len(a))
OUT["policy_blocks"] = {b: {k: v.tolist() for k, v in d.items()} for b, d in comp_tab.items()}
# modal plan per session (male / female) for the coach-facing template
modal = {}
for lab, sx in [("M", male), ("F", ~male)]:
    plan = []
    for s in range(C.N_STEPS):
        col = np.concatenate([pickle.load(open(p, "rb"))["test"]["actions"][s][sx] for p in seeds_of("D3QN")])
        col = col[col >= 0]
        c, t, i = C.decode(col)
        plan.append([C.COND[np.bincount(c, minlength=5).argmax()], C.TECH[np.bincount(t, minlength=4).argmax()],
                     C.INTENS[np.bincount(i, minlength=3).argmax()],
                     float(np.bincount(c, minlength=5).max() / len(c))])
    modal[lab] = plan
OUT["modal_plan"] = modal
# share of students receiving a plan different from the modal plan (personalisation)
a1 = pickle.load(open(seeds_of("D3QN")[0], "rb"))["test"]["actions"]
uniq = len({tuple(a1[:, j]) for j in range(a1.shape[1])})
OUT["unique_plans_seed1"] = uniq
# individual-level link: D3QN allocation of PLY+RES sessions vs baseline VJ z
zvj = (test_co.base[:, 1] - C.NORM_MEAN[test_co.sex, 1]) / C.NORM_SD[test_co.sex, 1]
c1, _, _ = C.decode(np.where(a1 >= 0, a1, 0))
plyres = ((c1 == 1) | (c1 == 2)).sum(0)
OUT["corr_plyres_vjz"] = [float(x) for x in stats.spearmanr(plyres, zvj)]

# ACWR trajectories (mean +/- sd) for figure
acw = {}
for v in ["Traditional", "AdaptiveRule", "GA", "D3QN"]:
    a = load(v)["test"]["acwr"] if v in BASE else pickle.load(open(seeds_of(v)[0], "rb"))["test"]["acwr"]
    acw[v] = dict(mean=a.mean(1).tolist(), p90=np.percentile(a, 90, axis=1).tolist())
OUT["acwr_traj"] = acw

# 10. inference latency -------------------------------------------------------
net = QNet(OBS_DIM, C.N_ACTIONS, True); net.load_state_dict(torch.load(f"{R}/model_D3QN_s1.pt"))
x = torch.randn(1, OBS_DIM)
t0 = time.perf_counter()
for _ in range(2000):
    with torch.no_grad():
        net(x).argmax()
OUT["latency_ms"] = (time.perf_counter() - t0) / 2000 * 1000
OUT["n_params"] = int(sum(p.numel() for p in net.parameters()))

# 11. calibration of the learned value function: predicted V(s0) vs realised
#     discounted return for every test student (D3QN, all seeds)
cal = []
for p in seeds_of("D3QN"):
    sd = p.replace(".pkl", ".pt").replace("/D3QN", "/model_D3QN")
    q = QNet(OBS_DIM, C.N_ACTIONS, True); q.load_state_dict(torch.load(sd))
    env = VolleySim(test_co); o = env.reset()
    with torch.no_grad():
        v0 = q(torch.as_tensor(o)).max(1).values.numpy()
    G = np.zeros(test_co.n); disc = 1.0; done = False
    while not done:
        with torch.no_grad():
            a = q(torch.as_tensor(o)).argmax(1).numpy()
        o, r, done = env.step(a); G += disc * r; disc *= 0.97
    cal.append((v0, G))
v0 = np.mean([c[0] for c in cal], 0); G = np.mean([c[1] for c in cal], 0)
OUT["value_cal"] = dict(r=float(np.corrcoef(v0, G)[0, 1]), mae=float(np.abs(v0 - G).mean()),
                        bias=float((v0 - G).mean()), v0=v0[:600].tolist(), G=G[:600].tolist(),
                        slope=float(np.polyfit(v0, G, 1)[0]))

json.dump(OUT, open(f"{R}/summary.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(main[["policy", "ret", "dvri", "dvri_ci", "inj_per100", "over"]].to_string())
print(pd.DataFrame(comp).to_string())
print("g main", g_main, "power", json.dumps({k: v for k, v in pw.items() if k != "monte_carlo"}, indent=0))
