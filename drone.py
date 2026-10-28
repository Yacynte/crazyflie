"""
Building blocks for distributed consensus with Crazyflies.

    Drone            one real Crazyflie (cflib): connect, state, arm, takeoff, velocity, land
    SimDrone         same interface, simulated kinematics (no hardware needed)
    Message          what one drone broadcasts each step (fits one P2P packet)
    Channel          one directed link: delay, packet loss
    Topology         who sends WHAT to WHOM (directed edges + field filters)
    ConsensusAgent   the per-drone logic; only sees its own state + its inbox
    ConsensusSwarm   owns everything: add/remove drones, route messages, run patterns

Rule of thumb: an agent never chooses its neighbours. The Topology decides
what reaches it, so the agent code stays identical when it moves onboard.

Quick start (simulation):

    from drone import SimDrone, ConsensusSwarm
    swarm = ConsensusSwarm(sim=True)
    swarm.add_drone(SimDrone(0, "cf1", start=(0, -0.3, 0)))
    swarm.add_drone(SimDrone(1, "cf2", start=(0,  0.3, 0)))
    swarm.connect_all(); swarm.takeoff_all(0.5)
    swarm.run("circle", duration=16)
    swarm.land_all(); swarm.disconnect_all()

Frames: with the Flow deck each Crazyflie starts its estimate at (0, 0) where it
was reset. Give every Drone its `origin` (where it sits in the shared frame) and
place all drones facing the same direction. With OptiTrack later, origin = 0.
"""
from __future__ import annotations

import math
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from patterns import get_pattern

DEFAULT_FIELDS = ("p", "c", "d")
VEC_FIELDS = ("p", "v", "c", "d")          # all 3-vectors, packable
P2P_MAX_BYTES = 60                         # Crazyflie P2P payload limit


# ==========================================================================
# Message
# ==========================================================================
@dataclass
class Message:
    sender: int
    seq: int
    stamp: float                            # sender's clock, s
    data: dict = field(default_factory=dict)  # field name -> np.ndarray(3)

    def filtered(self, fields):
        """Copy with only the allowed fields (what the Topology lets through)."""
        keep = {k: v.copy() for k, v in self.data.items() if k in fields}
        return Message(self.sender, self.seq, self.stamp, keep)

    def pack(self) -> bytes:
        """Binary form for the radio: id(B) seq(H) stamp_ms(I) mask(B) + 3 floats/field."""
        mask, payload = 0, b""
        for bit, name in enumerate(VEC_FIELDS):
            if name in self.data:
                mask |= 1 << bit
                payload += struct.pack("<3f", *self.data[name])
        raw = struct.pack("<BHIB", self.sender, self.seq & 0xFFFF,
                          int(self.stamp * 1000) & 0xFFFFFFFF, mask) + payload
        if len(raw) > P2P_MAX_BYTES:
            raise ValueError(f"message is {len(raw)} B, P2P limit is {P2P_MAX_BYTES} B")
        return raw

    @staticmethod
    def unpack(raw: bytes) -> "Message":
        sender, seq, stamp_ms, mask = struct.unpack_from("<BHIB", raw, 0)
        stamp = stamp_ms / 1000.0          # wraps every ~50 days: use differences only
        off, data = struct.calcsize("<BHIB"), {}
        for bit, name in enumerate(VEC_FIELDS):
            if mask & (1 << bit):
                data[name] = np.array(struct.unpack_from("<3f", raw, off))
                off += 12
        return Message(sender, seq, stamp, data)


# ==========================================================================
# Channel: one directed link
# ==========================================================================
class Channel:
    def __init__(self, delay=0.0, loss=0.0, seed=None):
        self.delay = float(delay)           # s
        self.loss = float(loss)             # probability a packet is dropped
        self._rng = np.random.default_rng(seed)
        self._queue = deque()               # (deliver_time, msg)
        self.sent = self.dropped = 0

    def send(self, msg: Message, now: float):
        self.sent += 1
        if self._rng.random() < self.loss:
            self.dropped += 1
            return
        self._queue.append((now + self.delay, msg))

    def receive(self, now: float):
        out = []
        while self._queue and self._queue[0][0] <= now + 1e-9:
            out.append(self._queue.popleft()[1])
        return out


# ==========================================================================
# Topology: directed graph with per-edge field filters and channels
# ==========================================================================
class Topology:
    def __init__(self):
        self.nodes: set[int] = set()
        self.edges: dict[tuple[int, int], dict] = {}   # (src, dst) -> {"fields", "channel"}

    # ---- nodes ----------------------------------------------------------
    def add_node(self, i):
        self.nodes.add(i)

    def remove_node(self, i):
        self.nodes.discard(i)
        for e in [e for e in self.edges if i in e]:
            del self.edges[e]

    # ---- edges ----------------------------------------------------------
    def add_edge(self, src, dst, fields=DEFAULT_FIELDS, delay=0.0, loss=0.0, seed=None):
        """src -> dst: dst receives `fields` of src's message."""
        if src == dst:
            raise ValueError("no self-loops")
        self.nodes.update((src, dst))
        self.edges[(src, dst)] = {"fields": set(fields),
                                  "channel": Channel(delay, loss, seed)}

    def add_link(self, a, b, **kw):
        """Bidirectional link with the same settings both ways."""
        self.add_edge(a, b, **kw)
        self.add_edge(b, a, **kw)

    def remove_edge(self, src, dst):
        self.edges.pop((src, dst), None)

    def connect_all(self, **kw):
        for a in self.nodes:
            for b in self.nodes:
                if a != b:
                    self.add_edge(a, b, **kw)

    def set_channel(self, src, dst, delay=None, loss=None):
        ch = self.edges[(src, dst)]["channel"]
        if delay is not None:
            ch.delay = delay
        if loss is not None:
            ch.loss = loss

    # ---- queries --------------------------------------------------------
    def in_neighbours(self, i):
        """Nodes whose messages reach i."""
        return sorted(s for (s, d) in self.edges if d == i)

    def out_neighbours(self, i):
        return sorted(d for (s, d) in self.edges if s == i)

    def laplacian(self, order=None):
        """L = D_in - A with A[i, j] = 1 if j -> i (i listens to j)."""
        order = sorted(self.nodes) if order is None else list(order)
        idx = {n: k for k, n in enumerate(order)}
        L = np.zeros((len(order), len(order)))
        for (s, d) in self.edges:
            if s in idx and d in idx:
                L[idx[d], idx[s]] -= 1.0
                L[idx[d], idx[d]] += 1.0
        return L, order

    def lambda2(self, order=None):
        """Algebraic connectivity of the symmetrised graph: 0 means split, bigger means faster."""
        L, _ = self.laplacian(order)
        if len(L) < 2:
            return 0.0
        Ls = 0.5 * (L + L.T)
        return float(np.sort(np.linalg.eigvalsh(Ls))[1])

    def has_spanning_tree(self, order=None):
        """Consensus on a directed graph converges iff some node reaches all others."""
        nodes = sorted(self.nodes) if order is None else list(order)
        if len(nodes) <= 1:
            return True
        live = set(nodes)
        for root in nodes:
            seen, stack = {root}, [root]
            while stack:
                u = stack.pop()
                for v in self.out_neighbours(u):
                    if v in live and v not in seen:
                        seen.add(v)
                        stack.append(v)
            if seen == live:
                return True
        return False

    def describe(self):
        lines = [f"nodes: {sorted(self.nodes)}"]
        for (s, d), e in sorted(self.edges.items()):
            ch = e["channel"]
            lines.append(f"  {s} -> {d}  fields={sorted(e['fields'])}  "
                         f"delay={ch.delay:.3f}s loss={ch.loss:.0%}")
        return "\n".join(lines)


# ==========================================================================
# Drones: one interface, two backends
# ==========================================================================
class BaseDrone:
    """Interface every drone backend implements."""

    def __init__(self, drone_id: int, name: str, origin=(0.0, 0.0, 0.0)):
        self.id = int(drone_id)
        self.name = name
        self.origin = np.array(origin, float)     # position of local (0,0,0) in the shared frame
        self.connected = False
        self.blind = False                         # True: own position not available
        self.expected_start = None                 # (x, y) where it should sit before takeoff

    # shared-frame state
    def position(self) -> np.ndarray: raise NotImplementedError
    def velocity(self) -> np.ndarray: raise NotImplementedError

    # life cycle
    def connect(self): raise NotImplementedError
    def disconnect(self): raise NotImplementedError
    def arm(self): pass
    def takeoff(self, height, duration): raise NotImplementedError
    def land(self, duration): raise NotImplementedError
    def send_velocity(self, v, yaw_rate=0.0): raise NotImplementedError
    def stop(self): raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}(id={self.id}, name='{self.name}')"


class Drone(BaseDrone):
    """A real Crazyflie through cflib. The link stays open until disconnect()."""

    LOG_PERIOD_MS = 20        # 50 Hz state logging

    def __init__(self, uri, drone_id, name=None, origin=(0.0, 0.0, 0.0),
                 require_flow_deck=True, cache="./cache"):
        super().__init__(drone_id, name or f"cf{drone_id}", origin)
        self.uri = uri
        self.require_flow_deck = require_flow_deck
        self.cache = cache
        self.scf = None
        self._log = None
        self._lock = threading.Lock()
        self._p_local = np.zeros(3)
        self._v = np.zeros(3)
        self.last_state_time = None

    @property
    def cf(self):
        return self.scf.cf

    def connect(self, timeout=10.0):
        from cflib.crazyflie import Crazyflie
        from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

        print(f"{self.name}: connecting to {self.uri} ...")
        self.scf = SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache=self.cache))
        self.scf.open_link()

        if self.require_flow_deck:
            if int(self.cf.param.get_value("deck.bcFlow2")) != 1:
                self.scf.close_link()
                raise RuntimeError(f"{self.name}: Flow deck not detected")
            print(f"{self.name}: Flow deck attached")

        try:   # older firmware needs the high-level commander enabled explicitly
            self.cf.param.set_value("commander.enHighLevel", "1")
        except Exception:
            pass

        self.reset_estimator()
        self._start_logging()
        t0 = time.time()
        while self.last_state_time is None:
            if time.time() - t0 > timeout:
                raise RuntimeError(f"{self.name}: no state estimate received")
            time.sleep(0.02)
        self.connected = True
        print(f"{self.name}: ready, local p = {np.round(self._p_local, 3)}")

    def reset_estimator(self):
        self.cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        self.cf.param.set_value("kalman.resetEstimation", "0")
        time.sleep(1.0)   # let the EKF settle

    def _start_logging(self):
        from cflib.crazyflie.log import LogConfig
        conf = LogConfig(name=f"state_{self.id}", period_in_ms=self.LOG_PERIOD_MS)
        for v in ("x", "y", "z", "vx", "vy", "vz"):     # 6 floats = 24 B < 26 B limit
            conf.add_variable(f"stateEstimate.{v}", "float")
        self.cf.log.add_config(conf)
        conf.data_received_cb.add_callback(self._on_state)
        conf.start()
        self._log = conf

    def _on_state(self, timestamp, data, _conf):
        with self._lock:
            self._p_local = np.array([data["stateEstimate.x"], data["stateEstimate.y"],
                                      data["stateEstimate.z"]])
            self._v = np.array([data["stateEstimate.vx"], data["stateEstimate.vy"],
                                data["stateEstimate.vz"]])
            self.last_state_time = time.time()

    def local_position(self):
        """Raw onboard estimate, before adding the origin."""
        with self._lock:
            return self._p_local.copy()

    def position(self):
        with self._lock:
            return self._p_local + self.origin

    def velocity(self):
        with self._lock:
            return self._v.copy()

    def arm(self):
        self.cf.supervisor.send_arming_request(True)
        time.sleep(0.5)

    def takeoff(self, height, duration):
        """Non-blocking: the onboard high-level commander flies the climb."""
        self.cf.high_level_commander.takeoff(height, duration)

    def send_velocity(self, v, yaw_rate=0.0):
        self.cf.commander.send_velocity_world_setpoint(float(v[0]), float(v[1]),
                                                       float(v[2]), yaw_rate)

    def land(self, duration):
        """Hand control back to the high-level commander, then land (non-blocking)."""
        self.cf.commander.send_notify_setpoint_stop()
        self.cf.high_level_commander.land(0.0, duration)

    def stop(self):
        """Emergency: motors off immediately."""
        try:
            self.cf.commander.send_stop_setpoint()
        except Exception:
            pass

    def disconnect(self):
        if self.scf is None:
            return
        try:
            if self._log is not None:
                self._log.stop()
            self.cf.supervisor.send_arming_request(False)
        except Exception:
            pass
        self.scf.close_link()
        self.connected = False
        print(f"{self.name}: disconnected")


class SimDrone(BaseDrone):
    """Kinematic stand-in: velocity command -> first-order response -> position."""

    def __init__(self, drone_id, name=None, start=(0.0, 0.0, 0.0), tau=0.15,
                 noise=0.003, seed=None):
        super().__init__(drone_id, name or f"sim{drone_id}", origin=(0.0, 0.0, 0.0))
        self.p = np.array(start, float)
        self.v = np.zeros(3)
        self.v_cmd = np.zeros(3)
        self.tau, self.noise = tau, noise
        self._rng = np.random.default_rng(seed if seed is not None else drone_id)
        self._hl = None                      # ("takeoff"/"land", z_start, z_goal, t, T)

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def position(self):
        return self.p.copy()

    def velocity(self):
        return self.v.copy()

    def takeoff(self, height, duration):
        self._hl = ["takeoff", self.p[2], height, 0.0, duration]

    def land(self, duration):
        self._hl = ["land", self.p[2], 0.0, 0.0, duration]

    def send_velocity(self, v, yaw_rate=0.0):
        self._hl = None                      # low-level setpoint overrides high-level
        self.v_cmd = np.asarray(v, float)

    def stop(self):
        self.v_cmd = np.zeros(3)
        self._hl = None

    def advance(self, dt):
        if self._hl is not None:             # smooth climb / descent, xy held
            kind, z0, z1, t, T = self._hl
            t = min(t + dt, T)
            self._hl[3] = t
            s = 0.5 * (1 - math.cos(math.pi * t / T)) if T > 0 else 1.0
            z_new = z0 + (z1 - z0) * s
            self.v = np.array([0.0, 0.0, (z_new - self.p[2]) / dt])
            self.p[2] = z_new
            return
        self.v += (self.v_cmd - self.v) * min(dt / self.tau, 1.0)
        self.p += self.v * dt + self._rng.normal(0.0, self.noise, 3)
        self.p[2] = max(self.p[2], 0.0)


# ==========================================================================
# ConsensusAgent: the per-drone brain
# ==========================================================================
class ConsensusAgent:
    """
    Two consensus layers:
      reference consensus : every drone keeps its own copy c of the formation
                            centre and averages it with its neighbours
      formation consensus : drone tracks slot c + d_i and also aligns
                            (p_i - d_i) with its neighbours'

    The agent sees ONLY its own state and its inbox (latest message per sender).
    """

    def __init__(self, drone_id, k_p=1.0, k_f=0.8, k_c=1.0, v_max=0.5,
                 stale_after=0.5):
        self.id = drone_id
        self.k_p, self.k_f, self.k_c, self.v_max = k_p, k_f, k_c, v_max
        self.stale_after = stale_after       # s: ignore neighbours older than this
        self.c = None                        # own copy of the centre (None until initialised)
        self.slot, self.n = 0, 1
        self.seq = 0
        self.d = np.zeros(3)
        self.inbox: dict[int, Message] = {}       # latest message per sender
        self.rx_time: dict[int, float] = {}       # when it arrived (receiver clock)

    def set_slot(self, slot, n):
        self.slot, self.n = slot, n

    def forget(self, sender):
        self.inbox.pop(sender, None)
        self.rx_time.pop(sender, None)

    def receive(self, msg: Message, now: float):
        old = self.inbox.get(msg.sender)
        if old is None or msg.seq > old.seq:      # keep only the newest
            self.inbox[msg.sender] = msg
            self.rx_time[msg.sender] = now

    def fresh_neighbours(self, now):
        return [m for s, m in self.inbox.items()
                if now - self.rx_time.get(s, -1e9) <= self.stale_after]

    def init_centre(self, p, now, pattern_fn, t):
        """Joining drone: take neighbours' centre if any, else derive from own position."""
        nb = [m.data["c"] for m in self.fresh_neighbours(now) if "c" in m.data]
        if nb:
            self.c = np.mean(nb, axis=0)
        else:
            _, d, _ = pattern_fn(t, self.slot, self.n)
            self.c = np.asarray(p, float) - d

    def step(self, t, now, dt, p, pattern_fn):
        """Return velocity command. p may be None if the drone is blind."""
        c_dot, d, d_dot = pattern_fn(t, self.slot, self.n)
        self.d = d
        if self.c is None:
            self.init_centre(p if p is not None else d, now, pattern_fn, t)

        nbrs = self.fresh_neighbours(now)

        # reference consensus on the centre copy
        dc = np.zeros(3)
        for m in nbrs:
            if "c" in m.data:
                dc += m.data["c"] - self.c
        self.c = self.c + dt * (c_dot + self.k_c * dc)

        if p is None:                        # blind and no estimate: hover in place
            return np.zeros(3)

        # slot tracking + formation consensus
        v = c_dot + d_dot - self.k_p * (p - self.c - d)
        for m in nbrs:
            if "p" in m.data and "d" in m.data:
                v -= self.k_f * ((p - d) - (m.data["p"] - m.data["d"]))

        speed = np.linalg.norm(v)
        if speed > self.v_max:
            v *= self.v_max / speed
        return v

    def message(self, now, p, v=None) -> Message:
        self.seq += 1
        data = {}
        if self.c is not None:            # a joining drone listens before it talks
            data["c"], data["d"] = self.c.copy(), self.d.copy()
        if p is not None:
            data["p"] = np.asarray(p, float).copy()
        if v is not None:
            data["v"] = np.asarray(v, float).copy()
        return Message(self.id, self.seq, now, data)


# ==========================================================================
# ConsensusSwarm: membership, routing, flight phases
# ==========================================================================
@dataclass
class Member:
    drone: BaseDrone
    agent: ConsensusAgent
    status: str = "idle"          # idle -> taking_off -> active -> landing -> idle
    ready_at: float = 0.0         # time the current high-level manoeuvre finishes


class ConsensusSwarm:
    """
    Owns drones, agents and the topology.

    add_drone(..., link="all")   joins the drone and links it to every member
    remove_drone(id)             lands it (if flying) and removes all its edges
    join(id) / leave(id)         take off / land one drone while the others keep flying
    run(pattern, duration)       the consensus loop; `events` let you script
                                 changes mid-run: [(t, fn(swarm)), ...]
    """

    def __init__(self, sim=False, dt=0.05, height=0.5, min_separation=0.35,
                 agent_kwargs=None):
        self.sim = sim
        self.dt = dt
        self.height = height
        self.min_separation = min_separation
        self.agent_kwargs = agent_kwargs or {}
        self.members: dict[int, Member] = {}
        self.topology = Topology()
        self.lock = threading.RLock()
        self._sim_time = 0.0
        self.log = []                 # one dict per control tick

    # ---- time -----------------------------------------------------------
    def now(self):
        return self._sim_time if self.sim else time.time()

    def sleep(self, seconds):
        if not self.sim:
            time.sleep(max(seconds, 0.0))
            return
        steps = max(int(round(seconds / self.dt)), 1)
        for _ in range(steps):
            for m in self.members.values():
                m.drone.advance(self.dt)
            self._sim_time += self.dt

    # ---- membership -----------------------------------------------------
    def add_drone(self, drone: BaseDrone, link="all", fields=DEFAULT_FIELDS,
                  delay=0.0, loss=0.0):
        """link: 'all', a list of member ids, or None (wire it yourself)."""
        with self.lock:
            if drone.id in self.members:
                raise ValueError(f"drone id {drone.id} already in swarm")
            if self.sim and not isinstance(drone, SimDrone):
                raise TypeError("sim swarm only accepts SimDrone")
            self.members[drone.id] = Member(drone, ConsensusAgent(drone.id, **self.agent_kwargs))
            self.topology.add_node(drone.id)
            others = [i for i in self.members if i != drone.id] if link == "all" else (link or [])
            for j in others:
                self.topology.add_link(drone.id, j, fields=fields, delay=delay, loss=loss)
            self._assign_slots()
        return drone

    def remove_drone(self, drone_id, land=True):
        with self.lock:
            m = self.members.get(drone_id)
            if m is None:
                return
            if land and m.status in ("active", "taking_off"):
                m.drone.land(2.0)
            self.topology.remove_node(drone_id)
            del self.members[drone_id]
            for other in self.members.values():          # forget its messages
                other.agent.forget(drone_id)
            self._assign_slots()
            self._check_graph()

    def active_ids(self):
        return sorted(i for i, m in self.members.items() if m.status == "active")

    def _assign_slots(self):
        """Slot = rank among flying members (an operator decision, not consensus)."""
        ids = sorted(i for i, m in self.members.items() if m.status in ("active", "taking_off"))
        for k, i in enumerate(ids):
            self.members[i].agent.set_slot(k, len(ids))

    def _check_graph(self):
        ids = self.active_ids()
        if len(ids) > 1 and not self.topology.has_spanning_tree(ids):
            print(f"WARNING: topology among {ids} has no spanning tree, "
                  f"consensus will split into groups")

    # ---- whole-swarm phases ----------------------------------------------
    def connect_all(self):
        """Connect in parallel (each cflib connection takes a few seconds)."""
        errors = {}

        def _c(m):
            try:
                m.drone.connect()
            except Exception as e:
                errors[m.drone.id] = e

        threads = [threading.Thread(target=_c, args=(m,)) for m in self.members.values()]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        if errors:
            raise RuntimeError(f"connection failed: {errors}")

    def preflight_check(self, tol=0.15):
        """Before takeoff: positions must match where the drones really sit, and
        no two drones may be closer than min_separation. Catches frame errors
        (e.g. an origin counted twice), which no in-flight check can detect."""
        P = {}
        for i, m in sorted(self.members.items()):
            p = m.drone.position()
            P[i] = p
            msg = f"  {m.drone.name}: shared-frame p = {np.round(p, 2)}"
            exp = m.drone.expected_start
            if exp is not None:
                err = np.linalg.norm(p[:2] - np.asarray(exp, float))
                msg += f"  expected {tuple(exp)}  error {err:.2f} m"
                if err > tol:
                    print(msg)
                    raise RuntimeError(f"{m.drone.name}: estimate is {err:.2f} m from its "
                                       f"expected start. Check origin / frame settings.")
            print(msg)
        ids = sorted(P)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                d = np.linalg.norm(P[ids[a]] - P[ids[b]])
                if d < self.min_separation:
                    raise RuntimeError(f"drones {ids[a]} and {ids[b]} are {d:.2f} m apart on the ground")
        print("preflight OK")

    def takeoff_all(self, height=None, duration=2.5):
        """Everyone climbs together (commands go out back-to-back from one thread)."""
        height = self.height if height is None else height
        for m in self.members.values():
            m.drone.arm()
        for m in self.members.values():
            m.drone.takeoff(height, duration)
            m.status = "taking_off"
        self.sleep(duration + 0.5)
        for m in self.members.values():
            m.status = "active"
        self._assign_slots()
        self._check_graph()

    def land_all(self, duration=2.5):
        for m in self.members.values():
            if m.status != "idle":
                m.drone.land(duration)
                m.status = "landing"
        self.sleep(duration + 0.5)
        for m in self.members.values():
            m.status = "idle"

    def disconnect_all(self):
        for m in self.members.values():
            m.drone.disconnect()

    def emergency_stop(self):
        for m in self.members.values():
            m.drone.stop()

    # ---- single-drone phases (non-blocking, safe during run) --------------
    def join(self, drone_id, height=None, duration=2.5):
        m = self.members[drone_id]
        if not m.drone.connected:
            m.drone.connect()
        m.drone.arm()
        m.drone.takeoff(self.height if height is None else height, duration)
        m.status, m.ready_at = "taking_off", self.now() + duration + 0.5
        m.agent.c = None                        # re-initialise from neighbours on entry
        self._assign_slots()

    def leave(self, drone_id, duration=2.5):
        m = self.members[drone_id]
        m.drone.land(duration)
        m.status, m.ready_at = "landing", self.now() + duration + 0.5
        for other in self.members.values():
            other.agent.forget(drone_id)
        self._assign_slots()

    # ---- the consensus loop -------------------------------------------------
    def run(self, pattern, duration, events=None, verbose=True):
        """
        Each tick:
          1. each active drone reads ITS OWN state
          2. its agent produces a message
          3. the topology filters it per edge and pushes it into that edge's channel
          4. channels deliver whatever has arrived (delay / loss applied)
          5. each agent computes its velocity from own state + inbox only
        """
        fn = get_pattern(pattern) if isinstance(pattern, str) else pattern
        events = sorted(events or [], key=lambda e: e[0])
        t0 = self.now()
        next_tick = t0
        ev = 0
        try:
            while True:
                now = self.now()
                t = now - t0
                if t >= duration:
                    break
                while ev < len(events) and t >= events[ev][0]:
                    events[ev][1](self)
                    ev += 1
                self._update_transitions(now)

                with self.lock:
                    ids = self.active_ids()
                    states = {}
                    for i in ids:
                        d = self.members[i].drone
                        states[i] = None if d.blind else d.position()

                    # 2-3: broadcast through the topology
                    for i in ids:
                        msg = self.members[i].agent.message(now, states[i])
                        for j in self.topology.out_neighbours(i):
                            if j in ids:
                                e = self.topology.edges[(i, j)]
                                e["channel"].send(msg.filtered(e["fields"]), now)
                    # 4: delivery
                    for (s, d), e in self.topology.edges.items():
                        for msg in e["channel"].receive(now):
                            if d in self.members:
                                self.members[d].agent.receive(msg, now)
                    # 5: local decisions
                    cmds = {i: self.members[i].agent.step(t, now, self.dt, states[i], fn)
                            for i in ids}
                    for i, v in cmds.items():
                        self.members[i].drone.send_velocity(v)

                    self._record(t, ids)

                next_tick += self.dt
                if self.sim:
                    self.sleep(self.dt)
                else:
                    self.sleep(next_tick - time.time())
        except BaseException:
            print("run aborted: landing everyone")
            self.land_all()
            raise

        # leave a clean state: hover until the next run / land call
        for i in self.active_ids():
            self.members[i].drone.send_velocity(np.zeros(3))

    def _update_transitions(self, now):
        for m in self.members.values():
            if m.status == "taking_off" and m.ready_at and now >= m.ready_at:
                m.status = "active"
                self._assign_slots()
                self._check_graph()
            elif m.status == "landing" and m.ready_at and now >= m.ready_at:
                m.status = "idle"

    def _record(self, t, ids):
        rec = {"t": t, "p": {}, "c": {}, "status": {i: m.status for i, m in self.members.items()}}
        for i in ids:
            rec["p"][i] = self.members[i].drone.position()
            rec["c"][i] = self.members[i].agent.c.copy()
        if len(ids) > 1:
            P = np.array([rec["p"][i] for i in ids])
            dmin = min(np.linalg.norm(P[a] - P[b])
                       for a in range(len(ids)) for b in range(a + 1, len(ids)))
            rec["min_sep"] = dmin
            if dmin < self.min_separation:
                print(f"t={t:5.2f}s WARNING: separation {dmin:.2f} m < {self.min_separation} m")
        self.log.append(rec)
