"""
IMU noise characterisation for the Crazyflie -> KFConfig for kf_crazyflie.py.

    python imu_allan.py record static.npz --minutes 60     drone still, motors off
    python imu_allan.py analyze static.npz [--plot]       -> imu_noise.json
    python imu_allan.py --selftest                        synthetic data, known answer

Allan variance of each accelerometer axis, fitted (least squares, log space)
with exactly the error model the filter uses:

    AVAR(tau) = N^2 / tau                    white noise / velocity random walk
              + AVAR_GM(tau; sigma, T)       correlated noise, Gauss-Markov
              + K^2 tau / 3                  bias random walk

    acc_vrw = N [m/s/sqrt(s)],  gm_sigma = sigma [m/s^2],  gm_tau = T [s],
    bias_rw = K [m/s^2/sqrt(s)]  (needs a long log: the curve has to turn up)

Reading the slopes off by eye (slope -1/2 -> N, floor -> B, slope +1/2 -> K)
is badly biased when the terms overlap, which they do on MEMS parts; the
textbook numbers (N, floor B, tau_min) are still reported for comparison.

The onboard attitude is characterised too: its static roll/pitch wander,
times g, is the horizontal acceleration error from gravity leakage
(tilt_sigma, tilt_tau: std and 1/e autocorrelation time).

Static, motors-off data gives the sensor's own noise. In flight, vibration
adds a lot on top: either record a second log with the drone clamped down
and motors at hover thrust, or treat acc_vrw/gm_sigma as lower bounds.
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np

G = 9.81


# ── Allan deviation ─────────────────────────────────────────────────────────
def allan_deviation(x, dt, n_taus=60):
    """Overlapping Allan deviation of rate signal x sampled at dt. Returns (taus, adev)."""
    x = np.asarray(x, float)
    n = len(x)
    theta = np.concatenate(([0.0], np.cumsum(x) * dt))
    ms = np.unique(np.logspace(0, math.log10((n - 1) // 2), n_taus).astype(int))
    taus, adev = [], []
    for m in ms:
        tau = m * dt
        d = theta[2 * m:] - 2.0 * theta[m:-m] + theta[:-2 * m]
        taus.append(tau)
        adev.append(math.sqrt(np.sum(d * d) / (2.0 * tau * tau * len(d))))
    return np.array(taus), np.array(adev)


def gm_avar(tau, sigma, T):
    """Allan variance of a first-order Gauss-Markov process (std sigma, time constant T)."""
    q = 2.0 * sigma ** 2 / T
    r = tau / T
    return q * T ** 2 / tau * (1.0 - (3.0 - 4.0 * np.exp(-r) + np.exp(-2.0 * r)) / (2.0 * r))


def model_avar(tau, N, sigma, T, K):
    return N ** 2 / tau + gm_avar(tau, sigma, T) + K ** 2 * tau / 3.0


def fit_noise(taus, adev, duration, tau_fit_min=0.05):
    """Least-squares fit of white + Gauss-Markov + bias random walk to the Allan curve.

    Fits in log space, each point weighted by its confidence (at long tau
    there are only a few independent clusters, so those points are noisy).
    Below tau_fit_min the firmware's low-pass filter bends the curve, so it's
    left out. Several starting time constants guard against local minima.
    K is None when the log is too short for the random walk to show.
    """
    from scipy.optimize import least_squares

    m = taus >= tau_fit_min
    t, a = taus[m], adev[m]
    i_min = int(np.argmin(adev))
    B, tau_min = float(adev[i_min]), float(taus[i_min])

    # relative std of an overlapping Allan deviation estimate ~ 1/sqrt(2 (n_clusters - 1))
    n_clusters = np.maximum(duration / t, 2.0)
    w = np.sqrt(2.0 * (n_clusters - 1.0))

    def resid(lx):
        N, s, T, K = np.exp(lx)
        return w * (0.5 * np.log(model_avar(t, N, s, T, K)) - np.log(a))

    lo = np.log([1e-9, 1e-9, t[0] / 10, 1e-12])
    hi = np.log([1e2, 1e2, t[-1] * 10, 1e2])
    sol = None
    for T0 in np.logspace(math.log10(t[0]), math.log10(t[-1]), 6):
        # initial guess from the textbook slopes
        x0 = np.log([a[0] * math.sqrt(t[0]), B / 0.62, T0,
                     max(a[-1] * math.sqrt(3.0 / t[-1]), 1e-9)])
        cand = least_squares(resid, np.clip(x0, lo + 1e-6, hi - 1e-6), bounds=(lo, hi))
        if sol is None or cand.cost < sol.cost:
            sol = cand
    N, sigma, T, K = (float(v) for v in np.exp(sol.x))
    log_err = sol.fun / w

    # random walk not resolved if it's < 10 % of the variance at the longest tau
    if K ** 2 * t[-1] / 3.0 < 0.1 * a[-1] ** 2:
        K = None
    return {"N": N, "gm_sigma": sigma, "gm_tau": T, "K": K,
            "B": B, "tau_min": tau_min, "fit_rms_log": float(np.sqrt(np.mean(log_err ** 2)))}


def gm_fit(x, dt):
    """Std and 1/e autocorrelation time of a (zero-mean-ised) signal."""
    x = np.asarray(x, float) - np.mean(x)
    sigma = float(np.std(x))
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= ac[0]
    below = np.nonzero(ac < 1.0 / math.e)[0]
    tau = float(below[0] * dt) if len(below) else float(n * dt)
    return sigma, tau


# ── analysis ────────────────────────────────────────────────────────────────
def _check_rate(t, name):
    d = np.diff(t)
    dt = float(np.median(d))
    gaps = np.mean(d > 1.5 * dt) * 100
    if gaps > 1.0:
        print(f"[allan] WARNING: {name}: {gaps:.1f}% of samples late or dropped — "
              f"the Allan curve assumes uniform sampling")
    return dt


def analyze(acc, t_acc, att=None, t_att=None):
    """acc: (n, 3) m/s^2. att: (n, 2) roll/pitch, rad. Returns (params, curves)."""
    dt = _check_rate(t_acc, "acc")
    params = {"dt": dt, "duration_s": float(t_acc[-1] - t_acc[0]), "axes": {}}
    curves = {}
    for j, ax in enumerate("xyz"):
        taus, adev = allan_deviation(acc[:, j], dt)
        fit = fit_noise(taus, adev, params["duration_s"])
        params["axes"][ax] = fit
        curves[ax] = (taus, adev)

    if att is not None:
        dt_att = _check_rate(t_att, "attitude")
        tilt = [gm_fit(att[:, j], dt_att) for j in range(2)]
        params["tilt_sigma_rad"] = max(s for s, _ in tilt)
        params["tilt_tau_s"] = float(np.median([tau for _, tau in tilt]))
    return params, curves


def to_kf_params(params):
    """Conservative (worst axis) mapping onto KFConfig fields."""
    axes = params["axes"].values()
    out = {
        "acc_vrw": max(a["N"] for a in axes),
        "gm_sigma": max(a["gm_sigma"] for a in axes),
        "gm_tau": float(np.median([a["gm_tau"] for a in axes])),
    }
    ks = [a["K"] for a in axes if a["K"] is not None]
    if ks:
        out["bias_rw"] = max(ks)
    else:
        print("[allan] no bias random walk region — record longer; keeping the default bias_rw")
    if "tilt_sigma_rad" in params:
        out["tilt_sigma"] = G * math.sin(params["tilt_sigma_rad"])
        out["tilt_tau"] = params["tilt_tau_s"]
    return out


def print_report(params, kf):
    print(f"\n{params['duration_s'] / 60:.1f} min at {1 / params['dt']:.0f} Hz")
    print(f"{'axis':>4} {'N [m/s/rt s]':>13} {'GM sigma':>10} {'GM tau [s]':>10} "
          f"{'K [m/s^2/rt s]':>15} {'B [m/s^2]':>10} {'fit rms':>8}")
    for ax, a in params["axes"].items():
        k = f"{a['K']:.2e}" if a["K"] is not None else "—"
        print(f"{ax:>4} {a['N']:13.2e} {a['gm_sigma']:10.2e} {a['gm_tau']:10.1f} {k:>15} "
              f"{a['B']:10.2e} {a['fit_rms_log']:8.3f}")
    if "tilt_sigma_rad" in params:
        print(f"tilt wander {math.degrees(params['tilt_sigma_rad']):.3f} deg, "
              f"tau {params['tilt_tau_s']:.1f} s")
    print("\nKFConfig:", json.dumps({k: float(f"{v:.4g}") for k, v in kf.items()}))


def plot(curves, params):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (taus, adev) in curves.items():
        ax.loglog(taus, adev, label=f"acc {name}")
        a = params["axes"][name]
        fit = model_avar(taus, a["N"], a["gm_sigma"], a["gm_tau"], a["K"] or 0.0)
        ax.loglog(taus, np.sqrt(fit), '--', color='gray', lw=0.8)
    ax.set_xlabel("tau [s]")
    ax.set_ylabel("Allan deviation [m/s^2]")
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ── recording ───────────────────────────────────────────────────────────────
def record(path, minutes, uri):
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

    cflib.crtp.init_drivers()
    acc_rows, att_rows = [], []

    def acc_cb(ts, d, _):
        acc_rows.append((ts / 1000.0, d['acc.x'], d['acc.y'], d['acc.z']))

    def att_cb(ts, d, _):
        att_rows.append((ts / 1000.0, d['stabilizer.roll'], d['stabilizer.pitch']))

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        confs = []
        for name, variables, cb in (('Allan_acc', ('acc.x', 'acc.y', 'acc.z'), acc_cb),
                                    ('Allan_att', ('stabilizer.roll', 'stabilizer.pitch'), att_cb)):
            conf = LogConfig(name=name, period_in_ms=10)
            for v in variables:
                conf.add_variable(v, 'float')
            scf.cf.log.add_config(conf)
            conf.data_received_cb.add_callback(cb)
            conf.start()
            confs.append(conf)

        t_end = time.time() + minutes * 60
        try:
            while time.time() < t_end:
                left = t_end - time.time()
                print(f"\r[allan] recording — don't touch the table — {left / 60:5.1f} min left, "
                      f"{len(acc_rows)} samples", end='', flush=True)
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[allan] stopped early")
        finally:
            for conf in confs:
                conf.stop()

    acc = np.array(acc_rows)
    att = np.array(att_rows)
    np.savez(path, t_acc=acc[:, 0], acc=acc[:, 1:] * G,
             t_att=att[:, 0], att=np.radians(att[:, 1:]))
    print(f"\n[allan] saved {len(acc)} samples to {path}")


# ── self-test ───────────────────────────────────────────────────────────────
def synth(n, dt, N, gm_sigma, gm_tau, K, rng):
    from scipy.signal import lfilter
    phi = math.exp(-dt / gm_tau)
    white = rng.normal(0, N / math.sqrt(dt), n)
    gm = lfilter([1.0], [1.0, -phi], rng.normal(0, gm_sigma * math.sqrt(1 - phi ** 2), n))
    rw = np.cumsum(rng.normal(0, K * math.sqrt(dt), n))
    return white + gm + rw


def selftest():
    rng = np.random.default_rng(0)
    dt, n = 0.01, 100 * 3600 * 2      # 2 h at 100 Hz
    true = {"acc_vrw": 2e-3, "gm_sigma": 3e-3, "gm_tau": 20.0, "bias_rw": 5e-5}
    acc = np.stack([synth(n, dt, true["acc_vrw"], true["gm_sigma"], true["gm_tau"],
                          true["bias_rw"], rng) for _ in range(3)], axis=1)
    params, curves = analyze(acc, np.arange(n) * dt)
    kf = to_kf_params(params)
    print(f"{'':>9} {'true':>9} {'fitted':>9}")
    for k, v in true.items():
        print(f"{k:>9} {v:9.2e} {kf.get(k, float('nan')):9.2e}")
    return true, kf, params, curves


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', nargs='?', choices=['record', 'analyze'])
    ap.add_argument('file', nargs='?')
    ap.add_argument('--minutes', type=float, default=60.0)
    ap.add_argument('--out', default='imu_noise.json')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    if args.selftest:
        selftest()
        sys.exit(0)
    if args.cmd == 'record':
        from cflib.utils import uri_helper
        record(args.file, args.minutes,
               uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E702'))
    elif args.cmd == 'analyze':
        d = np.load(args.file)
        params, curves = analyze(d['acc'], d['t_acc'], d['att'], d['t_att'])
        kf = to_kf_params(params)
        print_report(params, kf)
        with open(args.out, 'w') as f:
            json.dump({"kf": kf, "allan": params}, f, indent=2)
        print(f"written to {args.out} — load with KFConfig.from_json('{args.out}')")
        if args.plot:
            plot(curves, params)
    else:
        ap.print_help()
