"""Validation-only tuning of the DRL training recipe (the test cohort is never used here).
Results are written to results/tuning.json and reported in the Supplementary table."""
import json, time
import numpy as np
import torch
import config as C
from volleysim import Cohort, run_policy
from baselines import static_policy, rule_policy
from agents import train

torch.set_num_threads(2)
tr = Cohort(2000, C.SEED_COHORT_TRAIN)
va = Cohort(500, C.SEED_COHORT_TRAIN + 1)
ga = np.load("results/ga_plan.npz")["plan"]
ref = dict(GA=float(run_policy(static_policy(ga), va)["ret"].mean()),
           rule=float(run_policy(rule_policy(), va)["ret"].mean()))
print("reference val returns", ref, flush=True)
cfgs = {
    "T0_1step_g097": dict(),
    "T1_3step_g099": dict(n_step=3, gamma=0.99),
    "T2_T1+demo": dict(n_step=3, gamma=0.99, demos=[static_policy(ga), rule_policy()], demo_episodes=60, eps_start=0.5),
}
SEEDS = {}
res = {"reference": ref}
for name, kw in cfgs.items():
    ep = kw.pop("episodes", 400)
    for s in SEEDS.get(name, [1, 2]):
        t = time.time()
        ag, curve = train("D3QN", s, tr, va, episodes=ep, eval_every=50, **kw)
        o = run_policy(ag.policy(), va)
        res[f"{name}_s{s}"] = dict(val=float(o["ret"].mean()), dvri=float(o["dvri_true"].mean()),
                                   inj=float(o["injuries"].mean()), over=float(o["over"].mean()),
                                   secs=round(time.time() - t))
        print(name, s, res[f"{name}_s{s}"], flush=True)
        json.dump(res, open("results/tuning.json", "w"), indent=1)
