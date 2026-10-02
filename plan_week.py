"""Coach-facing tool: turns the latest class test sheet and the session-RPE log
into the next session's prescription for every student, plus a class-level
station plan (students grouped by recommended conditioning focus).

    python plan_week.py --make-example          # writes example_class.csv / example_load_log.csv
    python plan_week.py --students example_class.csv --log example_load_log.csv \
                        --session 9 --weeks-since-test 0

Student CSV columns: id, sex (M/F), step_index, vj_cm, grip_kg, agility_s,
plank_s, serve, pass, spike, block, wellness (0 = fresh ... 2 = very tired),
injured (0/1).  Load-log CSV columns: id, day, srpe_load (AU), jumps,
conditioning (AER/PLY/RES/AGL/CST), technical (SERVE/PASS/SPIKE/BLOCK), intensity (L/M/H).
"""
import argparse
import numpy as np
import pandas as pd
import torch
import config as C
from agents import QNet
from volleysim import OBS_DIM


def make_example(n=24, seed=3):
    rng = np.random.default_rng(seed)
    sex = np.array(["M", "F"] * (n // 2))
    sx = (sex == "F").astype(int)
    base = C.NORM_MEAN[sx] + rng.normal(0, 1, (n, 5)) * C.NORM_SD[sx]
    sk = np.clip(C.SKILL_MEAN[sx] + rng.normal(0, 1, (n, 4)), 1, 9)
    df = pd.DataFrame(dict(id=[f"S{i + 1:02d}" for i in range(n)], sex=sex))
    for k, c in enumerate(["step_index", "vj_cm", "grip_kg", "agility_s", "plank_s"]):
        df[c] = base[:, k].round(1)
    for j, c in enumerate(["serve", "pass", "spike", "block"]):
        df[c] = sk[:, j].round(1)
    df["wellness"] = rng.uniform(0.2, 1.2, n).round(2)
    df["injured"] = 0
    df.to_csv("example_class.csv", index=False)
    rows = []
    for sid in df.id:
        for d, (cf, tf, it) in enumerate([("AER", "SERVE", "M"), ("AGL", "PASS", "M"), ("PLY", "SPIKE", "M"), ("CST", "BLOCK", "L")]):
            rows.append(dict(id=sid, day=[1, 4, 8, 11][d], srpe_load=C.SRPE[C.INTENS.index(it)] * 90,
                             jumps=60, conditioning=cf, technical=tf, intensity=it))
    pd.DataFrame(rows).to_csv("example_load_log.csv", index=False)
    print("wrote example_class.csv and example_load_log.csv")


def ewma(log, sid, today):
    a = c = C.BACKGROUND_LOAD
    lg = log[log.id == sid].set_index("day")["srpe_load"].to_dict()
    for d in range(1, today + 1):
        x = C.BACKGROUND_LOAD + lg.get(d, 0.0)
        a = C.LAMBDA_A * x + (1 - C.LAMBDA_A) * a
        c = C.LAMBDA_C * x + (1 - C.LAMBDA_C) * c
    return a, c


def build_obs(df, log, session, wst):
    n = len(df)
    sx = (df.sex.values == "F").astype(int)
    P = df[["step_index", "vj_cm", "grip_kg", "agility_s", "plank_s"]].values.astype(float)
    S = df[["serve", "pass", "spike", "block"]].values.astype(float)
    sign = np.where(C.LOWER_IS_BETTER, -1.0, 1.0)
    zp = sign * (P - C.NORM_MEAN[sx]) / C.NORM_SD[sx]
    zs = (S - 5.0) / C.SKILL_SCALE
    today = int(log.day.max()) if len(log) else 0
    O = np.zeros((n, OBS_DIM), np.float32)
    for r, sid in enumerate(df.id):
        a, c = ewma(log, sid, today)
        lg = log[log.id == sid].sort_values("day")
        oc, ot = np.zeros(5), np.zeros(4); li = 0.0
        if len(lg):
            last = lg.iloc[-1]
            oc[C.COND.index(last.conditioning)] = 1; ot[C.TECH.index(last.technical)] = 1
            li = C.INTENS.index(last.intensity) / 2
        cc = np.array([(lg.conditioning == x).sum() for x in C.COND]) / max(session, 1)
        ct = np.array([(lg.technical == x).sum() for x in C.TECH]) / max(session, 1)
        jw = lg[lg.day > today - 7].jumps.sum() if len(lg) else 0
        O[r] = np.r_[zp[r], zs[r], wst / C.TEST_EVERY_WEEKS, session / C.N_STEPS, a / 500, c / 500, a / c,
                     df.wellness.values[r], sx[r], df.injured.values[r], oc, ot, li, cc, ct, jw / 200]
    return O


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-example", action="store_true")
    ap.add_argument("--students"); ap.add_argument("--log")
    ap.add_argument("--session", type=int, default=0, help="index of the next session (0-31)")
    ap.add_argument("--weeks-since-test", type=int, default=0)
    ap.add_argument("--model", default=None, help="trained D3QN weights (default: model_D3QN_s1.pt or results/model_D3QN_s1.pt)")
    a = ap.parse_args()
    if a.make_example:
        make_example(); return
    df = pd.read_csv(a.students)
    log = pd.read_csv(a.log) if a.log else pd.DataFrame(columns=["id", "day", "srpe_load", "jumps", "conditioning", "technical", "intensity"])
    import os
    path = a.model or ("model_D3QN_s1.pt" if os.path.exists("model_D3QN_s1.pt") else "results/model_D3QN_s1.pt")
    net = QNet(OBS_DIM, C.N_ACTIONS, True); net.load_state_dict(torch.load(path)); net.eval()
    O = build_obs(df, log, a.session, a.weeks_since_test)
    with torch.no_grad():
        act = net(torch.as_tensor(O)).argmax(1).numpy()
    c, t, i = C.decode(act)
    out = df[["id", "sex"]].copy()
    out["conditioning"] = [C.COND[x] for x in c]; out["technical"] = [C.TECH[x] for x in t]
    out["intensity"] = [C.INTENS[x] for x in i]
    out.loc[df.injured.values == 1, ["conditioning", "technical", "intensity"]] = ["REST/REHAB", "-", "-"]
    print("\nIndividual prescriptions for session", a.session + 1)
    print(out.to_string(index=False))
    print("\nStation plan (conditioning block, 30 min):")
    for g, sub in out.groupby("conditioning"):
        print(f"  {g:10s} {len(sub):2d} students: {', '.join(sub.id)}")
    tech = out[out.technical != "-"].technical.mode()
    print("\nWhole-class technical block (55 min):", tech.iloc[0] if len(tech) else "-")
    print("Intensity mix:", out.intensity.value_counts().to_dict())
    out.to_csv("next_session_plan.csv", index=False)
    print("\nSaved next_session_plan.csv")


if __name__ == "__main__":
    main()
