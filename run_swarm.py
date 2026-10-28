"""
Fly a consensus pattern with N drones, in simulation or on hardware.

    python run_swarm.py --sim --pattern circle
    python run_swarm.py --sim --pattern eight --n 3 --topology chain --loss 0.1
    python run_swarm.py --sim --pattern circle --n 3 --scenario leave_join
    python run_swarm.py --pattern hover --duration 8          # hardware, 2 drones

Hardware: drones must sit at their ORIGINS below, all facing the same way.
Place them under their slot at t = 0 (for 2 drones every pattern starts with
slot 0 at y = -0.3 and slot 1 at y = +0.3), otherwise they cross while settling.
Start with `--pattern hover`, then line, circle, eight, swap.
"""
import argparse

import numpy as np

from drone import ConsensusSwarm, Drone, SimDrone
from patterns import get_pattern

# id -> (uri, origin in the shared frame). Edit to match your setup.
HARDWARE = {
    0: ("radio://0/80/2M/E7E7E7E701", (0.0, -0.3, 0.0)),
    1: ("radio://0/80/2M/E7E7E7E702", (0.0, 0.3, 0.0)),
}


def build(args):
    swarm = ConsensusSwarm(sim=args.sim, height=args.height)
    ids = list(range(args.n)) if args.sim else sorted(HARDWARE)[: args.n]
    for k, i in enumerate(ids):
        if args.sim:   # start each drone under its slot at t = 0
            _, d, _ = get_pattern(args.pattern)(0.0, k, len(ids))
            swarm.add_drone(SimDrone(i, start=(d[0], d[1], 0.0)), link=None)
        else:
            uri, origin = HARDWARE[i]
            swarm.add_drone(Drone(uri, i, origin=origin), link=None)

    # wire the graph
    kw = dict(delay=args.delay, loss=args.loss)
    if args.topology == "all":
        swarm.topology.connect_all(**kw)
    elif args.topology == "chain":              # 0 <-> 1 <-> 2 ...
        for a, b in zip(ids, ids[1:]):
            swarm.topology.add_link(a, b, **kw)
    elif args.topology == "leader":             # 0 -> everyone, nobody -> 0 (directed)
        for j in ids[1:]:
            swarm.topology.add_edge(ids[0], j, **kw)
        for a, b in zip(ids[1:], ids[2:]):
            swarm.topology.add_link(a, b, **kw)
    return swarm


def scenario_events(name, swarm):
    if name == "leave_join":     # last drone lands at 8 s, takes off again at 14 s
        last = max(swarm.members)
        return [(8.0, lambda s: s.leave(last)), (14.0, lambda s: s.join(last))]
    if name == "blind":          # drone 0 loses its own position from 6 s to 12 s
        def blind(on):
            return lambda s: setattr(s.members[0].drone, "blind", on)
        return [(6.0, blind(True)), (12.0, blind(False))]
    return []


def plot(swarm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = sorted(swarm.members)
    t = np.array([r["t"] for r in swarm.log])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for i in ids:
        P = np.array([r["p"].get(i, [np.nan] * 3) for r in swarm.log])
        ax[0].plot(P[:, 0], P[:, 1], label=f"drone {i}")
    ax[0].set_aspect("equal"); ax[0].set_title("top view"); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]")

    sep = np.array([r.get("min_sep", np.nan) for r in swarm.log])
    ax[1].plot(t, sep); ax[1].axhline(swarm.min_separation, color="r", ls=":", label="min safe")
    ax[1].set_title("minimum separation"); ax[1].set_xlabel("t [s]"); ax[1].legend(fontsize=8)

    err = []
    for r in swarm.log:
        C = np.array(list(r["c"].values()))
        err.append(np.max(np.linalg.norm(C - C.mean(0), axis=1)) if len(C) > 1 else np.nan)
    ax[2].semilogy(t, np.array(err) + 1e-9)
    ax[2].set_title("centre disagreement (max |c_i - mean|)"); ax[2].set_xlabel("t [s]")
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--pattern", default="hover")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--height", type=float, default=0.5)
    ap.add_argument("--topology", choices=["all", "chain", "leader"], default="all")
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--loss", type=float, default=0.0)
    ap.add_argument("--scenario", choices=["none", "leave_join", "blind"], default="none")
    ap.add_argument("--plot", default=None, help="save a PNG of the run")
    args = ap.parse_args()

    if not args.sim:
        import cflib.crtp
        cflib.crtp.init_drivers()

    swarm = build(args)
    print(swarm.topology.describe())
    print(f"lambda2 = {swarm.topology.lambda2():.3f}")

    # with --sim, give drones deliberately different centre copies to show agreement
    swarm.connect_all()
    try:
        swarm.takeoff_all()
        if args.sim:
            for k, m in enumerate(swarm.members.values()):
                m.agent.c = np.array([0.1 * k, -0.1 * k, args.height])
        swarm.run(args.pattern, args.duration, events=scenario_events(args.scenario, swarm))
        swarm.land_all()
    finally:
        swarm.disconnect_all()

    if args.plot:
        plot(swarm, args.plot)
