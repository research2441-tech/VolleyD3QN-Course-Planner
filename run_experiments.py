"""Runs every experiment reported in the manuscript and stores raw outcomes in
./results.  Each run is check-pointed, so the script can be resumed.

    python run_experiments.py            # full study (~90 min on 2 CPU cores)
    python run_experiments.py --quick    # smoke test (~3 min)
"""
import os, sys, json, time, pickle
import numpy as np
import torch
import config as C
from volleysim import Cohort, run_policy, OBS_DIM
from baselines import (random_policy, traditional_policy, periodised_policy,
                       rule_policy, static_policy, ga_optimise)
from agents import train

QUICK = "--quick" in sys.argv
OUT = "results"
os.makedirs(OUT, exist_ok=True)
EPISODES = 40 if QUICK else 400
SEEDS = C.RL_SEEDS[:2] if QUICK else C.RL_SEEDS
ABL_SEEDS = C.RL_SEEDS[:1] if QUICK else C.RL_SEEDS[:3]

train_co = Cohort(2000, C.SEED_COHORT_TRAIN)
val_co = Cohort(500, C.SEED_COHORT_TRAIN + 1)
test_co = Cohort(1000, C.SEED_COHORT_TEST)
stress_co = Cohort(1000, C.SEED_COHORT_STRESS, perturb=0.30)


def save(name, pol):
    res = {"test": run_policy(pol, test_co), "stress": run_policy(pol, stress_co)}
    with open(f"{OUT}/{name}.pkl", "wb") as f:
        pickle.dump(res, f)


def done(name):
    return os.path.exists(f"{OUT}/{name}.pkl")


log = {}
t0 = time.time()
# ---------------- baselines ------------------------------------------------
for name, pol in [("Random", random_policy(7)), ("Traditional", traditional_policy()),
                  ("Periodised", periodised_policy()), ("AdaptiveRule", rule_policy())]:
    if not done(name):
        save(name, pol)
if not done("GA"):
    tg = time.time()
    best, hist = ga_optimise(train_co, pop=40, gens=10 if QUICK else 60, seed=0)
    np.savez(f"{OUT}/ga_plan.npz", plan=best, hist=hist, secs=time.time() - tg)
    save("GA", static_policy(best))
if not done("GA-matched"):
    # same number of simulated student-semesters as one DRL run (400 x 64 = 25,600)
    best_m, hist_m = ga_optimise(train_co, pop=20, gens=2 if QUICK else 19, batch=64, seed=0)
    np.savez(f"{OUT}/ga_matched_plan.npz", plan=best_m, hist=hist_m)
    save("GA-matched", static_policy(best_m))
print("baselines done", round(time.time() - t0), "s", flush=True)

# ---------------- value-based DRL family -----------------------------------
for variant in ["DQN", "DoubleDQN", "DuelingDQN", "D3QN"]:
    for s in SEEDS:
        name = f"{variant}_s{s}"
        if done(name):
            continue
        tr = time.time()
        ag, curve = train(variant, s, train_co, val_co, episodes=EPISODES, eval_every=20)
        np.save(f"{OUT}/curve_{name}.npy", curve)
        torch.save(ag.q.state_dict(), f"{OUT}/model_{name}.pt")
        save(name, ag.policy())
        print(name, "trained in", round(time.time() - tr), "s; val", curve[-1], flush=True)

# ---------------- D3QN ablations -------------------------------------------
# indices of load / readiness features in the observation vector
LOAD_IDX = [11, 12, 13, 14, 16, OBS_DIM - 1]
mask = np.ones(OBS_DIM, np.float32); mask[LOAD_IDX] = 0
ABL = {"D3QN-noACWRpenalty": dict(reward_acwr=False),
       "D3QN-noLoadState": dict(obs_mask=mask),
       "D3QN-small": dict(hidden=(64, 64, 32)),
       "D3QN-shield": dict(shield=True)}
for an, kw in ABL.items():
    for s in ABL_SEEDS:
        name = f"{an}_s{s}"
        if done(name):
            continue
        ag, curve = train("D3QN", s, train_co, val_co, episodes=EPISODES, eval_every=20, **kw)
        np.save(f"{OUT}/curve_{name}.npy", curve)
        save(name, ag.policy())
        print(name, "done", flush=True)
print("ALL DONE", round(time.time() - t0), "s", flush=True)
