"""Benchmark planners: random, traditional syllabus, block periodisation,
adaptive coach rule and a genetic-algorithm static plan."""
import numpy as np
import config as C
from volleysim import run_policy


def random_policy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda o, env: rng.integers(0, C.N_ACTIONS, env.co.n)


# ---------------------------------------------------------------------------
# Traditional college syllabus: technique-led, moderate intensity, conditioning
# mostly aerobic with a jump block in the second half.
TRAD_PLAN = []
for w in range(C.N_WEEKS):
    if w < 4:
        pair = [("AER", "SERVE"), ("AER", "PASS")]
    elif w < 8:
        pair = [("AGL", "PASS"), ("AER", "SPIKE")]
    elif w < 12:
        pair = [("PLY", "SPIKE"), ("AER", "BLOCK")]
    else:
        pair = [("AER", "SPIKE"), ("AGL", "PASS")]
    for cf, tf in pair:
        TRAD_PLAN.append(C.encode(C.COND.index(cf), C.TECH.index(tf), 1))
TRAD_PLAN = np.array(TRAD_PLAN)


def static_policy(plan):
    plan = np.asarray(plan)
    return lambda o, env: np.full(env.co.n, plan[env.t]) if plan.ndim == 1 else plan[:, env.t]


def traditional_policy():
    return static_policy(TRAD_PLAN)


# ---------------------------------------------------------------------------
# Block periodisation: 3:1 loading (L-M-H then deload), conditioning and
# technical foci rotated so that every block meets each focus.
PER_PLAN = []
cyc_i = [0, 1, 2, 0]
for s in range(C.N_STEPS):
    w = s // 2
    it = cyc_i[w % 4]
    cf = [1, 3, 2, 4, 0][s % 5]
    tf = [2, 1, 3, 0][s % 4]
    PER_PLAN.append(C.encode(cf, tf, it))
PER_PLAN = np.array(PER_PLAN)


def periodised_policy():
    return static_policy(PER_PLAN)


# ---------------------------------------------------------------------------
def rule_policy():
    """Adaptive expert rule: train the weakest measured quality and skill;
    choose intensity from ACWR (load-management heuristic)."""
    # map weakest quality -> conditioning focus
    q2c = np.array([0, 1, 2, 3, 4])

    def pol(o, env):
        zp = o[:, 0:5]
        zs = o[:, 5:9]
        cf = q2c[np.argmin(zp, axis=1)]
        tf = np.argmin(zs, axis=1)
        acwr = env.acwr
        it = np.where(acwr > 1.3, 0, np.where(acwr < 1.0, 2, 1))
        # avoid repeating the identical conditioning focus twice in a row
        rep = (env.has_last > 0) & (env.last[:, 0] == cf)
        second = np.argsort(zp, axis=1)[:, 1]
        cf = np.where(rep, q2c[second], cf)
        return C.encode(cf, tf, it)
    return pol


# ---------------------------------------------------------------------------
def ga_optimise(cohort, pop=40, gens=60, pc=0.8, pm=0.03, seed=0, batch=200,
                verbose=False):
    """Genetic algorithm over a fixed 32-session plan (integer genes 0..59),
    fitness = mean simulated return on a training cohort."""
    rng = np.random.default_rng(seed)
    P = rng.integers(0, C.N_ACTIONS, (pop, C.N_STEPS))
    P[0] = TRAD_PLAN
    P[1] = PER_PLAN
    hist = []

    def fitness(plans):
        f = np.zeros(len(plans))
        idx = rng.choice(cohort.n, size=min(batch, cohort.n), replace=False)
        sub = cohort.subset(idx)
        for k, plan in enumerate(plans):
            f[k] = run_policy(static_policy(plan), sub)["ret"].mean()
        return f

    fit = fitness(P)
    for g in range(gens):
        new = [P[np.argmax(fit)].copy()]                     # elitism
        while len(new) < pop:
            i1, i2 = [max(rng.choice(pop, 3, replace=False), key=lambda j: fit[j])
                      for _ in range(2)]                      # tournament
            a, b = P[i1].copy(), P[i2].copy()
            if rng.random() < pc:
                cut = rng.integers(1, C.N_STEPS)
                a[cut:], b[cut:] = P[i2][cut:], P[i1][cut:]
            for ch in (a, b):
                m = rng.random(C.N_STEPS) < pm
                ch[m] = rng.integers(0, C.N_ACTIONS, m.sum())
                new.append(ch)
        P = np.array(new[:pop])
        fit = fitness(P)
        hist.append(fit.max())
        if verbose and g % 10 == 0:
            print("GA gen", g, round(fit.max(), 3))
    best = P[np.argmax(fit)]
    return best, np.array(hist)
