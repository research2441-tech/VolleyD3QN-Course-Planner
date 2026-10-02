"""Shared configuration for the VolleySim / D3QN study.

All physiological constants below are modelling assumptions calibrated to
ranges reported for recreationally active college students.  They are listed
in Table 2 of the manuscript and can be changed here without touching the
rest of the code.
"""
import numpy as np

SEED_COHORT_TRAIN = 11
SEED_COHORT_TEST = 2026
SEED_COHORT_STRESS = 909
RL_SEEDS = [1, 2, 3, 4, 5]

# --- course calendar -------------------------------------------------------
N_WEEKS = 16
SESSIONS_PER_WEEK = 2
N_STEPS = N_WEEKS * SESSIONS_PER_WEEK          # 32 decisions per semester
GAPS = [3, 4]                                  # days between sessions (Tue/Fri)
TEST_EVERY_WEEKS = 4                           # formal testing every 4 weeks
SESSION_MIN = 90                               # minutes per class

# --- action space ----------------------------------------------------------
COND = ["AER", "PLY", "RES", "AGL", "CST"]     # conditioning block focus
TECH = ["SERVE", "PASS", "SPIKE", "BLOCK"]     # technical block focus
INTENS = ["L", "M", "H"]
N_ACTIONS = len(COND) * len(TECH) * len(INTENS)  # 60


def decode(a):
    a = np.asarray(a)
    c = a // (len(TECH) * len(INTENS))
    t = (a // len(INTENS)) % len(TECH)
    i = a % len(INTENS)
    return c, t, i


def encode(c, t, i):
    return (c * len(TECH) + t) * len(INTENS) + i


SRPE = np.array([3.0, 5.0, 7.0])               # session RPE by intensity
DOSE = np.array([0.60, 1.00, 1.35])            # adaptive stimulus by intensity
TECH_DOSE = np.array([0.80, 1.00, 1.10])       # technical repetitions by intensity
JUMPS = np.array([[10, 15, 20],                # jumps per session [cond, intensity]
                  [40, 80, 120],
                  [5, 10, 15],
                  [20, 35, 50],
                  [5, 8, 10]], dtype=float)
TECH_JUMPS = np.array([10, 15, 45, 50], dtype=float)  # serve, pass, spike, block

# --- physical qualities ----------------------------------------------------
QUAL = ["AER", "VJ", "GRIP", "AGI", "CORE"]
UNITS = ["index", "cm", "kg", "s", "s"]
# sex-specific baseline mean / sd (0 = male, 1 = female)
NORM_MEAN = np.array([[58.0, 45.0, 42.0, 10.8, 95.0],
                      [54.0, 32.0, 26.0, 12.0, 80.0]])
NORM_SD = np.array([[9.0, 6.0, 6.0, 0.7, 30.0],
                    [8.0, 5.0, 4.0, 0.8, 28.0]])
LOWER_IS_BETTER = np.array([False, False, False, True, False])
# maximal relative change reachable in one semester (fraction of baseline)
CEIL = np.array([0.15, 0.10, 0.08, 0.06, 0.45])
# transfer of each conditioning focus (rows) to each quality (cols)
ETA = np.array([[1.00, 0.10, 0.05, 0.25, 0.15],   # aerobic intervals
                [0.20, 1.00, 0.10, 0.45, 0.20],   # plyometric jumps
                [0.05, 0.40, 1.00, 0.15, 0.35],   # resistance
                [0.30, 0.35, 0.05, 1.00, 0.20],   # footwork / agility
                [0.10, 0.20, 0.25, 0.25, 1.00]])  # core stability + dynamic stretch
THETA = np.array([0.090, 0.080, 0.070, 0.085, 0.110])  # learning rate of adaptation
DETRAIN = np.array([0.006, 0.005, 0.004, 0.006, 0.008])  # daily loss of adaptation
# prerequisite gating: rate multiplier G_k = G0_k + G1_k * (PRE_k . A)
PRE = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.5],      # jump power needs strength + trunk
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0, 0.5],      # agility needs power + trunk
                [0.0, 0.0, 0.0, 0.0, 0.0]])
G0 = np.array([1.0, 0.50, 1.0, 0.55, 1.0])
G1 = np.array([0.0, 1.20, 0.0, 1.00, 0.0])
REPEAT_DOSE = 0.75                              # stimulus kept when a focus is repeated
FATIGUE_MASK = np.array([0.05, 0.03, 0.01, 0.025, 0.05])  # acute fatigue masking (fraction)

# --- skills (0-10 rubric) --------------------------------------------------
SKILL_MEAN = np.array([[4.2, 4.0, 3.8, 3.4],
                       [4.0, 4.1, 3.2, 2.8]])
SKILL_SD = 1.0
BETA = np.array([[1.00, 0.15, 0.05, 0.05],     # serve practice
                 [0.10, 1.00, 0.10, 0.10],     # pass/reception practice
                 [0.10, 0.10, 1.00, 0.25],     # spike practice
                 [0.00, 0.05, 0.25, 1.00]])    # block practice
ALPHA = 0.012                                   # base technical learning rate
SKILL_DECAY = 0.004                             # daily loss of the unrehearsed gain
# physical enablers of skill expression (rows skills, cols qualities)
OMEGA = np.array([[0.00, 0.10, 0.45, 0.00, 0.45],   # serve: grip + trunk
                  [0.10, 0.10, 0.00, 0.55, 0.25],   # pass: agility + core
                  [0.00, 0.60, 0.15, 0.20, 0.05],   # spike: jump + agility
                  [0.00, 0.70, 0.00, 0.30, 0.00]])  # block: jump + lateral speed
PHYS_TO_SKILL = 1.2                            # points added when all enablers at ceiling

# --- fatigue / load --------------------------------------------------------
TAU_H = 2.5                                    # fatigue time constant (days)
H50 = 0.9                                      # fatigue at which adaptation halves
BACKGROUND_LOAD = 110.0                        # AU/day habitual activity
LAMBDA_A = 2.0 / (7 + 1)                       # EWMA acute
LAMBDA_C = 2.0 / (28 + 1)                      # EWMA chronic
ACWR_SAFE = 1.5
INJ_B0, INJ_B1, INJ_B2, INJ_B3 = -6.8, 2.4, 0.9, 0.6

# --- readiness index weights -----------------------------------------------
W_PHYS = np.array([0.08, 0.14, 0.05, 0.10, 0.08])
W_SKILL = np.array([0.13, 0.15, 0.14, 0.13])
SKILL_SCALE = 1.5

# --- measurement reliability (ICC) used to set test noise -------------------
ICC_TARGET = np.array([0.85, 0.95, 0.96, 0.90, 0.85])
ICC_SKILL = 0.80

# --- reward ----------------------------------------------------------------
R_SCALE = 10.0
PSI_ACWR = 3.0
PSI_INJ = 3.0

# --- load-safety shield (observable features only) --------------------------
SHIELD_ACWR = 1.25
SHIELD_WELL = 1.3
