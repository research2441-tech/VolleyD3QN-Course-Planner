"""VolleySim: a vectorised multi-component fitness-fatigue and skill-acquisition
simulator of a 16-week college volleyball course.

Each virtual student carries five latent physical adaptations (aerobic, vertical
jump, grip strength, agility, core endurance), four technique levels (serve,
pass, spike, block), a fatigue state and EWMA acute/chronic loads.  One
environment step is one 90-min class session.  All random numbers that affect
outcomes (measurement noise, injury draws) are pre-drawn per student so that
different policies can be compared on exactly the same students (common random
numbers / paired design).
"""
import numpy as np
import config as C


class Cohort:
    """A cohort of virtual students with pre-drawn random streams."""

    def __init__(self, n, seed, perturb=0.0):
        rng = np.random.default_rng(seed)
        self.n = n
        self.sex = np.r_[np.zeros(n // 2), np.ones(n - n // 2)].astype(int)
        rng.shuffle(self.sex)
        # correlated baseline physical z-scores
        corr = np.array([[1.0, 0.30, 0.20, 0.30, 0.25],
                         [0.30, 1.0, 0.35, 0.45, 0.25],
                         [0.20, 0.35, 1.0, 0.20, 0.30],
                         [0.30, 0.45, 0.20, 1.0, 0.20],
                         [0.25, 0.25, 0.30, 0.20, 1.0]])
        z = rng.multivariate_normal(np.zeros(5), corr, size=n)
        z = np.clip(z, -2.5, 2.5)
        sign = np.where(C.LOWER_IS_BETTER, -1.0, 1.0)
        self.base = C.NORM_MEAN[self.sex] + sign * z * C.NORM_SD[self.sex]
        sk = C.SKILL_MEAN[self.sex] + 0.35 * z[:, [1]] + rng.normal(0, C.SKILL_SD * 0.94, (n, 4))
        self.skill0 = np.clip(sk, 1.0, 8.0)
        # individual response traits (high / low responders)
        self.rho = rng.lognormal(0.0, 0.35, n)        # physical trainability
        self.alpha = rng.lognormal(0.0, 0.30, n)      # technical learning rate
        self.kappa = rng.lognormal(0.0, 0.25, n)      # fatigue sensitivity
        self.suscept = rng.lognormal(0.0, 0.30, n)    # injury susceptibility
        # optional structural misspecification (robustness test)
        if perturb > 0:
            u = lambda size: rng.uniform(1 - perturb, 1 + perturb, size)
            self.eta_mult = u((n, 5, 5))
            self.beta_mult = u((n, 4, 4))
            self.ceil_mult = u((n, 5))
            self.tau_mult = u(n)
        else:
            self.eta_mult = np.ones((n, 5, 5))
            self.beta_mult = np.ones((n, 4, 4))
            self.ceil_mult = np.ones((n, 5))
            self.tau_mult = np.ones(n)
        # pre-drawn random streams
        n_tests = C.N_WEEKS // C.TEST_EVERY_WEEKS + 1
        sd_e = C.NORM_SD[self.sex] * np.sqrt((1 - C.ICC_TARGET) / C.ICC_TARGET)
        self.noise_phys = rng.normal(0, 1, (n_tests + 1, n, 5)) * sd_e[None]
        sd_s = C.SKILL_SD * np.sqrt((1 - C.ICC_SKILL) / C.ICC_SKILL)
        self.noise_skill = rng.normal(0, sd_s, (n_tests + 1, n, 4))
        self.u_inj = rng.uniform(0, 1, (C.N_STEPS, n))
        self.inj_dur = rng.integers(2, 5, (C.N_STEPS, n))
        self.wellness_noise = rng.normal(0, 0.15, (C.N_STEPS + 1, n))

    def subset(self, idx):
        c = object.__new__(Cohort)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                if v.ndim >= 2 and v.shape[0] != self.n and v.shape[1] == self.n:
                    setattr(c, k, v[:, idx])
                elif v.shape[0] == self.n:
                    setattr(c, k, v[idx])
                else:
                    setattr(c, k, v)
            else:
                setattr(c, k, v)
        c.n = len(idx)
        return c


OBS_DIM = 9 + 1 + 1 + 3 + 1 + 1 + 1 + 5 + 4 + 1 + 5 + 4 + 1


class VolleySim:
    def __init__(self, cohort, reward_acwr=True):
        self.co = cohort
        self.reward_acwr = reward_acwr
        self.sign = np.where(C.LOWER_IS_BETTER, -1.0, 1.0)

    # ------------------------------------------------------------------
    def reset(self):
        co, n = self.co, self.co.n
        self.t = 0
        self.A = np.zeros((n, 5))
        self.s = co.skill0.copy()
        self.h = np.zeros(n)
        self.acute = np.full(n, C.BACKGROUND_LOAD)
        self.chronic = np.full(n, C.BACKGROUND_LOAD)
        self.inj_left = np.zeros(n, dtype=int)
        self.last = np.zeros((n, 3), dtype=int)
        self.has_last = np.zeros(n)
        self.cnt_c = np.zeros((n, 5))
        self.cnt_t = np.zeros((n, 4))
        self.jumps_week = np.zeros(n)
        self.n_inj = np.zeros(n)
        self.missed = np.zeros(n)
        self.over = np.zeros(n)
        self.acwr = np.ones(n)
        self.total_jumps = np.zeros(n)
        self.test_idx = 0
        self.weeks_since_test = 0
        self.meas = self._measure()
        self.meas0 = self.meas.copy()
        self.vri0_true = self.vri_true()
        self.vri_prev = self.vri0_true.copy()
        self.log_acwr = np.zeros((C.N_STEPS, n))
        self.log_action = np.full((C.N_STEPS, n), -1)
        return self.obs()

    # ------------------------------------------------------------------
    def capacity(self):
        ceil = C.CEIL[None] * self.co.ceil_mult
        return self.co.base * (1 + self.sign * ceil * self.A)

    def skills(self):
        return np.clip(self.s + C.PHYS_TO_SKILL * self.A @ C.OMEGA.T, 0, 10)

    def _zphys(self, P):
        sex = self.co.sex
        return self.sign * (P - C.NORM_MEAN[sex]) / C.NORM_SD[sex]

    def vri_from(self, P, S):
        return self._zphys(P) @ C.W_PHYS + ((S - 5.0) / C.SKILL_SCALE) @ C.W_SKILL

    def vri_true(self):
        return self.vri_from(self.capacity(), self.skills())

    def _measure(self):
        mask = 1 - self.sign * C.FATIGUE_MASK * np.minimum(self.h, 2.0)[:, None] / 2.0
        P = self.capacity() * mask + self.co.noise_phys[self.test_idx]
        S = np.clip(self.skills() + self.co.noise_skill[self.test_idx], 0, 10)
        self.test_idx += 1
        m = np.concatenate([P, S], axis=1)
        if not hasattr(self, "meas_hist") or self.test_idx == 1:
            self.meas_hist = []
        self.meas_hist.append(m)
        return m

    def meas_vri(self, m):
        return self.vri_from(m[:, :5], m[:, 5:])

    # ------------------------------------------------------------------
    def obs(self):
        n = self.co.n
        zp = self._zphys(self.meas[:, :5])
        zs = (self.meas[:, 5:] - 5.0) / C.SKILL_SCALE
        oc = np.zeros((n, 5)); ot = np.zeros((n, 4))
        idx = self.has_last > 0
        oc[idx, self.last[idx, 0]] = 1
        ot[idx, self.last[idx, 1]] = 1
        wellness = self.h + self.co.wellness_noise[min(self.t, C.N_STEPS)]
        steps = max(self.t, 1)
        o = np.column_stack([
            zp, zs,
            np.full(n, self.weeks_since_test / C.TEST_EVERY_WEEKS),
            np.full(n, self.t / C.N_STEPS),
            self.acute / 500.0, self.chronic / 500.0, self.acwr,
            wellness, self.co.sex, (self.inj_left > 0).astype(float),
            oc, ot, self.last[:, 2] / 2.0 * self.has_last,
            self.cnt_c / steps, self.cnt_t / steps,
            self.jumps_week / 200.0,
        ])
        return o.astype(np.float32)

    # ------------------------------------------------------------------
    def step(self, a):
        co, n, t = self.co, self.co.n, self.t
        c, tf, it = C.decode(a)
        train = self.inj_left == 0
        L = np.where(train, C.SRPE[it] * C.SESSION_MIN, 0.0)
        jumps = np.where(train, C.JUMPS[c, it] + C.TECH_JUMPS[tf] * C.TECH_DOSE[it], 0.0)
        # session-day load update (EWMA)
        day_load = C.BACKGROUND_LOAD + L
        self.acute = C.LAMBDA_A * day_load + (1 - C.LAMBDA_A) * self.acute
        self.chronic = C.LAMBDA_C * day_load + (1 - C.LAMBDA_C) * self.chronic
        self.acwr = self.acute / self.chronic
        # injury hazard
        logit = (C.INJ_B0 + C.INJ_B1 * np.maximum(0, self.acwr - 1.3)
                 + C.INJ_B2 * jumps / 100.0 + C.INJ_B3 * self.h)
        p_inj = np.clip(co.suscept / (1 + np.exp(-logit)), 0, 0.5)
        new_inj = train & (co.u_inj[t] < p_inj)
        # adaptation (only students who trained)
        psi = 1.0 / (1.0 + (self.h / C.H50) ** 2)
        eta = C.ETA[c] * co.eta_mult[np.arange(n), c]                 # (n,5)
        gate = C.G0[None] + C.G1[None] * (self.A @ C.PRE.T)
        rep = np.where((self.has_last > 0) & (self.last[:, 0] == c), C.REPEAT_DOSE, 1.0)
        dA = (co.rho[:, None] * C.THETA[None] * eta * gate * (C.DOSE[it] * rep)[:, None]
              * psi[:, None] * (1 - self.A))
        self.A += np.where(train[:, None], dA, 0.0)
        enab = 0.6 + 1.2 * (self.A @ C.OMEGA.T) / C.OMEGA.sum(1)[None]
        beta = C.BETA[tf] * co.beta_mult[np.arange(n), tf]
        ds = (co.alpha[:, None] * C.ALPHA * beta * C.TECH_DOSE[it][:, None]
              * psi[:, None] * enab * (10 - self.s))
        self.s += np.where(train[:, None], ds, 0.0)
        self.h += co.kappa * L / 600.0
        # bookkeeping
        self.over += train & (self.acwr > C.ACWR_SAFE)
        self.log_acwr[t] = self.acwr
        self.log_action[t] = np.where(train, a, -1)
        self.cnt_c[np.arange(n)[train], c[train]] += 1
        self.cnt_t[np.arange(n)[train], tf[train]] += 1
        self.last = np.where(train[:, None], np.column_stack([c, tf, it]), self.last)
        self.has_last = np.maximum(self.has_last, train.astype(float))
        self.total_jumps += jumps
        self.jumps_week = jumps if t % 2 == 0 else self.jumps_week + jumps
        self.missed += ~train
        self.inj_left = np.where(train, self.inj_left, self.inj_left - 1)
        self.inj_left = np.where(new_inj, co.inj_dur[t], self.inj_left)
        self.n_inj += new_inj
        # rest days until next session
        gap = C.GAPS[t % 2]
        tau_h = C.TAU_H * co.tau_mult
        for _ in range(gap - 1):
            self.acute = C.LAMBDA_A * C.BACKGROUND_LOAD + (1 - C.LAMBDA_A) * self.acute
            self.chronic = C.LAMBDA_C * C.BACKGROUND_LOAD + (1 - C.LAMBDA_C) * self.chronic
        self.h *= np.exp(-gap / tau_h)
        self.A *= (1 - C.DETRAIN) ** gap
        self.s = co.skill0 + (self.s - co.skill0) * (1 - C.SKILL_DECAY) ** gap
        self.t += 1
        # reward
        vri = self.vri_true()
        r = C.R_SCALE * (vri - self.vri_prev) - C.PSI_INJ * new_inj
        if self.reward_acwr:
            r = r - C.PSI_ACWR * np.maximum(0, self.acwr - C.ACWR_SAFE) * train
        self.vri_prev = vri
        # periodic testing
        if self.t % (C.TEST_EVERY_WEEKS * C.SESSIONS_PER_WEEK) == 0:
            self.meas = self._measure()
            self.weeks_since_test = 0
        elif self.t % 2 == 0:
            self.weeks_since_test += 1
        done = self.t >= C.N_STEPS
        return self.obs(), r.astype(np.float32), done

    # ------------------------------------------------------------------
    def summary(self):
        P0, S0 = self.meas0[:, :5], self.meas0[:, 5:]
        P1, S1 = self.meas[:, :5], self.meas[:, 5:]
        return dict(
            dvri_meas=self.meas_vri(self.meas) - self.meas_vri(self.meas0),
            dvri_true=self.vri_true() - self.vri0_true,
            dP=P1 - P0, dS=S1 - S0, P0=P0, P1=P1, S0=S0, S1=S1,
            injuries=self.n_inj.copy(), missed=self.missed.copy(),
            over=self.over.copy(), jumps=self.total_jumps.copy(),
            acwr=self.log_acwr.copy(), actions=self.log_action.copy(),
            dvri_wk4=self.meas_vri(self.meas_hist[1]) - self.meas_vri(self.meas_hist[0]),
        )


def run_policy(policy, cohort, reward_acwr=True):
    """Roll out a policy (callable obs, env -> actions) on every student."""
    env = VolleySim(cohort, reward_acwr)
    o = env.reset()
    ret = np.zeros(cohort.n)
    done = False
    while not done:
        a = policy(o, env)
        o, r, done = env.step(a)
        ret += r
    out = env.summary()
    out["ret"] = ret
    return out
