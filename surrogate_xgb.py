"""XGBoost outcome surrogate: predicts a student's semester change in the
volleyball readiness index (VRI) from baseline profile + plan descriptors,
and ranks plan descriptors with permutation importance (coach-facing view)."""
import json, pickle
import numpy as np
import torch
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
import config as C
from volleysim import Cohort, run_policy, OBS_DIM
from baselines import random_policy, traditional_policy, periodised_policy, rule_policy, static_policy
from agents import QNet

rng = np.random.default_rng(77)
co = Cohort(4000, 77)
ga_plan = np.load("results/ga_plan.npz")["plan"]
net = QNet(OBS_DIM, C.N_ACTIONS, True); net.load_state_dict(torch.load("results/model_D3QN_s1.pt"))


def d3qn_eps(eps):
    r = np.random.default_rng(5)
    def pol(o, env):
        with torch.no_grad():
            a = net(torch.as_tensor(o)).argmax(1).numpy()
        m = r.random(len(a)) < eps
        a[m] = r.integers(0, C.N_ACTIONS, m.sum())
        return a
    return pol


groups = np.array_split(rng.permutation(co.n), 6)
pols = [random_policy(3), traditional_policy(), periodised_policy(), rule_policy(),
        static_policy(ga_plan), d3qn_eps(0.3)]
feats, y, ym = [], [], []
names = (["sex"] + [f"base_{q}" for q in C.QUAL + C.TECH]
         + [f"share_{c}" for c in C.COND] + [f"share_{t}" for t in C.TECH]
         + ["share_high", "share_low", "jumps_total", "acwr_mean", "acwr_peak",
            "sessions_acwr>1.5", "sessions_missed", "focus_switches", "early_dVRI_wk4"])
for g, pol in zip(groups, pols):
    sub = co.subset(g)
    o = run_policy(pol, sub)
    acts = o["actions"]
    valid = acts >= 0
    c, t, i = C.decode(np.where(valid, acts, 0))
    nv = valid.sum(0).clip(1)
    sc = np.stack([((c == k) & valid).sum(0) / nv for k in range(5)], 1)
    st = np.stack([((t == k) & valid).sum(0) / nv for k in range(4)], 1)
    sh = ((i == 2) & valid).sum(0) / nv; sl = ((i == 0) & valid).sum(0) / nv
    sw = ((np.diff(c, axis=0) != 0) & valid[1:] & valid[:-1]).sum(0)
    zb = np.column_stack([(o["P0"] - C.NORM_MEAN[sub.sex]) / C.NORM_SD[sub.sex] * np.where(C.LOWER_IS_BETTER, -1, 1),
                          (o["S0"] - 5) / C.SKILL_SCALE])
    X = np.column_stack([sub.sex, zb, sc, st, sh, sl, o["jumps"] / 100, o["acwr"].mean(0),
                         o["acwr"].max(0), o["over"], o["missed"], sw, o["dvri_wk4"]])
    feats.append(X); y.append(o["dvri_true"]); ym.append(o["dvri_meas"])
X = np.vstack(feats); y = np.concatenate(y); ym = np.concatenate(ym)

params = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, reg_lambda=1.0, gamma=0.0, random_state=0, n_jobs=2)
kf = KFold(5, shuffle=True, random_state=0)
oof = np.zeros_like(y); fold = []
for tr, te in kf.split(X):
    m = xgb.XGBRegressor(**params).fit(X[tr], y[tr])
    p = m.predict(X[te]); oof[te] = p
    fold.append(dict(mae=mean_absolute_error(y[te], p), r2=r2_score(y[te], p),
                     mae_base=mean_absolute_error(y[te], np.full(len(te), y[tr].mean()))))
# same model, noisy test-based target (shows the measurement-error ceiling)
oofm = np.zeros_like(ym)
for tr, te in kf.split(X):
    oofm[te] = xgb.XGBRegressor(**params).fit(X[tr], ym[tr]).predict(X[te])
meas_r2 = r2_score(ym, oofm); meas_mae = mean_absolute_error(ym, oofm)
corr_true_meas = float(np.corrcoef(y, ym)[0, 1])
# permutation importance on a 25 % hold-out
idx = rng.permutation(len(y)); te, tr = idx[:1000], idx[1000:]
m = xgb.XGBRegressor(**params).fit(X[tr], y[tr])
pi = permutation_importance(m, X[te], y[te], n_repeats=10, random_state=0, scoring="r2")
order = np.argsort(-pi.importances_mean)
res = dict(
    n=len(y), folds=fold,
    mae=float(np.mean([f["mae"] for f in fold])), mae_sd=float(np.std([f["mae"] for f in fold], ddof=1)),
    r2=float(np.mean([f["r2"] for f in fold])), r2_sd=float(np.std([f["r2"] for f in fold], ddof=1)),
    mae_base=float(np.mean([f["mae_base"] for f in fold])),
    pearson=float(np.corrcoef(y, oof)[0, 1]),
    meas_r2=float(meas_r2), meas_mae=float(meas_mae), corr_true_meas=corr_true_meas,
    meas_r2_ceiling=float(corr_true_meas ** 2),
    importance=[dict(feature=names[j], mean=float(pi.importances_mean[j]), sd=float(pi.importances_std[j]))
                for j in order[:14]],
    oof_sample=dict(y=y[:1500].tolist(), p=oof[:1500].tolist()),
)
# Bland-Altman style agreement
dif = oof - y
res["bias"] = float(dif.mean()); res["loa"] = (float(dif.mean() - 1.96 * dif.std(ddof=1)), float(dif.mean() + 1.96 * dif.std(ddof=1)))
json.dump(res, open("results/surrogate.json", "w"), indent=1)
print({k: v for k, v in res.items() if k not in ("oof_sample", "folds")})
