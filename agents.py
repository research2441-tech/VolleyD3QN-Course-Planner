"""Value-based deep RL agents: DQN, Double DQN, Dueling DQN and D3QN
(dueling architecture + double-Q target).  One class, two switches."""
import numpy as np
import torch
import torch.nn as nn
import config as C
from volleysim import Cohort, VolleySim, OBS_DIM, run_policy

torch.set_num_threads(2)


class QNet(nn.Module):
    def __init__(self, n_in, n_out, dueling, hidden=(256, 256, 128)):
        super().__init__()
        h1, h2, h3 = hidden
        self.body = nn.Sequential(nn.Linear(n_in, h1), nn.ReLU(),
                                  nn.Linear(h1, h2), nn.ReLU(),
                                  nn.Linear(h2, h3), nn.ReLU())
        self.dueling = dueling
        if dueling:
            self.v = nn.Linear(h3, 1)
            self.a = nn.Linear(h3, n_out)
        else:
            self.q = nn.Linear(h3, n_out)

    def forward(self, x):
        z = self.body(x)
        if self.dueling:
            a = self.a(z)
            return self.v(z) + a - a.mean(dim=1, keepdim=True)
        return self.q(z)


class Replay:
    def __init__(self, cap, dim):
        self.o = np.zeros((cap, dim), np.float32)
        self.o2 = np.zeros((cap, dim), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.g = np.ones(cap, np.float32)
        self.cap, self.i, self.full = cap, 0, False

    def add(self, o, a, r, o2, d, g=None):
        n = len(a)
        idx = (self.i + np.arange(n)) % self.cap
        self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx] = o, a, r, o2, d
        self.g[idx] = 1.0 if g is None else g
        self.i = (self.i + n) % self.cap
        self.full = self.full or self.i < n

    def sample(self, bs, rng):
        hi = self.cap if self.full else self.i
        j = rng.integers(0, hi, bs)
        return self.o[j], self.a[j], self.r[j], self.o2[j], self.d[j], self.g[j]


VARIANTS = {"DQN": (False, False), "DoubleDQN": (True, False),
            "DuelingDQN": (False, True), "D3QN": (True, True)}


class Agent:
    def __init__(self, variant="D3QN", seed=0, lr=3e-4, gamma=0.97, tau=0.01,
                 batch=256, buffer=300_000, hidden=(256, 256, 128), obs_mask=None, shield=False):
        self.double, dueling = VARIANTS[variant]
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.q = QNet(OBS_DIM, C.N_ACTIONS, dueling, hidden)
        self.qt = QNet(OBS_DIM, C.N_ACTIONS, dueling, hidden)
        self.qt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma, self.tau, self.batch = gamma, tau, batch
        self.buf = Replay(buffer, OBS_DIM)
        self.mask = None if obs_mask is None else torch.tensor(obs_mask, dtype=torch.float32)
        self.shield = shield
        self.is_high = torch.tensor(C.decode(np.arange(C.N_ACTIONS))[2] == 2)

    def allowed(self, o):
        """Load-safety shield on observable features only: no high-intensity
        session when the current ACWR exceeds 1.25 or the wellness score shows
        marked fatigue (> 1.3)."""
        o = torch.as_tensor(o)
        risky = (o[:, 13] > C.SHIELD_ACWR) | (o[:, 14] > C.SHIELD_WELL)
        return ~(risky[:, None] & self.is_high[None, :])

    def _x(self, o):
        x = torch.as_tensor(o)
        return x * self.mask if self.mask is not None else x

    def act(self, o, eps=0.0):
        with torch.no_grad():
            q = self.q(self._x(o))
            if self.shield:
                ok = self.allowed(o)
                q = q.masked_fill(~ok, -1e9)
            a = q.argmax(1).numpy()
        if eps > 0:
            m = self.rng.random(len(a)) < eps
            rnd = self.rng.integers(0, C.N_ACTIONS, m.sum())
            if self.shield:
                # a blocked random action is downgraded to the same session at moderate intensity
                bad = ~ok.numpy()[m, rnd]
                rnd[bad] = rnd[bad] - 1
            a[m] = rnd
        return a

    def policy(self):
        return lambda o, env: self.act(o, 0.0)

    def update(self):
        o, a, r, o2, d, gk = self.buf.sample(self.batch, self.rng)
        ok2 = self.allowed(o2) if self.shield else None
        o, o2 = self._x(o), self._x(o2)
        a, r, d, gk = torch.as_tensor(a), torch.as_tensor(r), torch.as_tensor(d), torch.as_tensor(gk)
        with torch.no_grad():
            if self.double:
                qo = self.q(o2)
                if ok2 is not None:
                    qo = qo.masked_fill(~ok2, -1e9)
                a2 = qo.argmax(1, keepdim=True)
                q2 = self.qt(o2).gather(1, a2).squeeze(1)
            else:
                qt = self.qt(o2)
                if ok2 is not None:
                    qt = qt.masked_fill(~ok2, -1e9)
                q2 = qt.max(1).values
            y = r + gk * (1 - d) * q2
        q = self.q(o).gather(1, a[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.qt.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
        return float(loss.detach())


def push_episode(buf, O, A, R, gamma, n_step):
    """Store n-step transitions of one synchronous episode (T steps x B students)."""
    T = len(A)
    for t in range(T):
        k = min(n_step, T - t)
        G = sum((gamma ** j) * R[t + j] for j in range(k))
        done = float(t + k >= T)
        o2 = O[t + k] if t + k < len(O) else O[-1]
        buf.add(O[t], A[t], G.astype(np.float32), o2, np.full(len(A[t]), done), np.full(len(A[t]), gamma ** k, np.float32))


def train(variant, seed, train_cohort, val_cohort, episodes=500, envs=64,
          eps_frac=0.6, eps_end=0.05, updates_per_step=2, eval_every=25,
          reward_acwr=True, obs_mask=None, hidden=(256, 256, 128), verbose=False,
          gamma=0.97, n_step=1, demos=None, demo_episodes=0, eps_start=1.0, shield=False):
    ag = Agent(variant, seed, hidden=hidden, obs_mask=obs_mask, gamma=gamma, shield=shield)
    rng = np.random.default_rng(1000 + seed)
    curve = []
    # optional demonstration roll-outs (e.g. GA plan, adaptive rule) to warm-start the buffer
    for k in range(demo_episodes):
        pol = demos[k % len(demos)]
        idx = rng.choice(train_cohort.n, envs, replace=False)
        env = VolleySim(train_cohort.subset(idx), reward_acwr)
        O, A, R = [env.reset()], [], []
        done = False
        while not done:
            a = pol(O[-1], env)
            m = rng.random(envs) < 0.1
            a = np.where(m, rng.integers(0, C.N_ACTIONS, envs), a)
            o2, r, done = env.step(a)
            A.append(a); R.append(r); O.append(o2)
        push_episode(ag.buf, O, A, R, gamma, n_step)
    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (eps_frac * episodes))
        idx = rng.choice(train_cohort.n, envs, replace=False)
        env = VolleySim(train_cohort.subset(idx), reward_acwr)
        O, A, R = [env.reset()], [], []
        done = False
        while not done:
            a = ag.act(O[-1], eps)
            o2, r, done = env.step(a)
            A.append(a); R.append(r); O.append(o2)
            if ag.buf.i > 2000 or ag.buf.full:
                for _ in range(updates_per_step):
                    ag.update()
        push_episode(ag.buf, O, A, R, gamma, n_step)
        if (ep + 1) % eval_every == 0 or ep == 0:
            ret = run_policy(ag.policy(), val_cohort, reward_acwr)["ret"].mean()
            curve.append((ep + 1, ret))
            if verbose:
                print(variant, seed, ep + 1, round(eps, 3), round(ret, 3), flush=True)
    return ag, np.array(curve)
