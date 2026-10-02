# VolleyD3QN-Course-Planner

Code for the manuscript **"Optimization of Volleyball Training Courses based on Reinforcement Learning Techniques"** (Bian Yonghong, Inner Mongolia Minzu University).

The repository contains:

* **VolleySim**: a simulator of a 16-week college volleyball course. It covers fitness, fatigue and skill acquisition for 5 physical tests and 4 technique ratings, and models session-RPE load, EWMA ACWR, injury hazard and test measurement error derived from ICC values.
* **D3QN planner**: a dueling double deep Q-network that prescribes each class session as conditioning focus × technical focus × intensity (60 actions). DQN, Double DQN and Dueling DQN variants are included for comparison.
* **Benchmarks**: random, traditional syllabus, block periodisation, adaptive coach rule, and a genetic-algorithm static plan.
* **Analysis**: paired contrasts (Wilcoxon with Holm correction), Hedges g with 95 % CI, bootstrap CIs, ICC(3,1)/SEM/MDC95 reliability checks, analytic and Monte-Carlo power analysis, robustness and ablation runs, an XGBoost surrogate with permutation importance, and value-function calibration.
* **plan_week.py**: a coach-facing tool. It turns a class test sheet plus a session-RPE log into the next session's individual prescriptions and a station plan.

All data are synthetic and are generated from fixed seeds. No human participants are involved.

## Files (flat layout)

| File | Purpose |
|---|---|
| `config.py` | All simulator parameters (Table II-III of the paper), action space, reward weights |
| `volleysim.py` | Vectorised simulator (`Cohort`, `VolleySim`, `run_policy`) |
| `agents.py` | Q-network (plain / dueling), replay buffer, DQN-family training loop |
| `baselines.py` | Random, Traditional, Block-periodised, Adaptive-rule planners and the GA optimiser |
| `run_experiments.py` | Runs every experiment and stores outcomes in `results/` (checkpointed) |
| `analysis.py` | Statistics, writes `results/summary.json` and CSV tables |
| `surrogate_xgb.py` | XGBoost surrogate and permutation importance, writes `results/surrogate.json` |
| `figures.py` | Builds all figures into `figures/` |
| `plan_week.py` | Coach-facing next-session planner |
| `tune.py` | Validation-only comparison of training recipes (reported in Results IV-A) |
| `model_D3QN_s1.pt` | Trained D3QN weights (seed 1) used by `plan_week.py` |
| `summary.json`, `surrogate.json`, `tuning.json` | Result files behind every number in the paper (copies of `results/*.json`) |
| `table_*.csv` | Result tables (main outcomes, paired contrasts, effects, reliability) |
| `requirements.txt`, `LICENSE`, `CITATION.cff`, `.zenodo.json` | Environment and metadata |

## Reproduce

```bash
pip install -r requirements.txt
python run_experiments.py      # about 80 min on 2 CPU cores; add --quick for a 3-min smoke test
python tune.py               # validation-only recipe comparison (~20 min)
python analysis.py
python surrogate_xgb.py
python figures.py
```

## Use with a real class

```bash
python plan_week.py --make-example
python plan_week.py --students example_class.csv --log example_load_log.csv --session 4 --weeks-since-test 2
```

The planner was trained in simulation. Treat its output as a draft for the teacher to review, not as an automatic prescription, until a field trial has validated it. Test protocols and the field-trial sample size are given in the paper.

## Licence

MIT
