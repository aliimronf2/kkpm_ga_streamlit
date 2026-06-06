"""
ga_engine.py
============
Core GA engine — implementasi persis dari notebook v2.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────
# Type alias
# ─────────────────────────────────────────────────────────────
Gene       = Tuple[int, str]
Chromosome = List[Gene]


# ─────────────────────────────────────────────────────────────
# Problem
# ─────────────────────────────────────────────────────────────
@dataclass
class TaskAssignmentProblem:
    n              : int
    cost           : np.ndarray
    visitor_demand : Dict[str, int]
    avail          : List[List[str]]
    max_jam        : List[int]
    jam_shift      : Dict[str, int] = field(
        default_factory=lambda: {'pagi': 3, 'siang': 4, 'sore': 5}
    )

    def __post_init__(self):
        self.cost = np.array(self.cost, dtype=float)
        assert self.cost.shape == (self.n, self.n)
        assert len(self.avail)   == self.n
        assert len(self.max_jam) == self.n

    @property
    def shifts(self) -> List[str]:
        return list(self.visitor_demand.keys())

    @property
    def V_total(self) -> int:
        return sum(self.visitor_demand.values())

    def ideal_count(self) -> Dict[str, float]:
        return {s: round(self.n * v / self.V_total, 2)
                for s, v in self.visitor_demand.items()}


# ─────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────
def generate_problem(
    n              : int                   = 9,
    visitor_demand : Dict[str, int]        = None,
    cost_range     : Tuple[int, int]       = (5, 25),
    max_jam_range  : Tuple[int, int]       = (4, 10),
    seed           : int                   = 42,
) -> TaskAssignmentProblem:
    if visitor_demand is None:
        visitor_demand = {'pagi': 100, 'siang': 200, 'sore': 500}

    shifts        = list(visitor_demand.keys())
    jam_shift_map = {'pagi': 3, 'siang': 4, 'sore': 5}
    rng           = np.random.RandomState(seed)

    cost    = rng.randint(cost_range[0], cost_range[1] + 1, size=(n, n)).astype(float)
    max_jam = [int(rng.randint(max_jam_range[0], max_jam_range[1] + 1)) for _ in range(n)]

    avail = []
    for i in range(n):
        feasible = [s for s in shifts if jam_shift_map.get(s, 0) <= max_jam[i]]
        if not feasible:
            feasible = [min(shifts, key=lambda s: jam_shift_map.get(s, 0))]
        k      = min(int(rng.choice([1, 1, 2, len(feasible)])), len(feasible))
        chosen = list(rng.choice(feasible, size=k, replace=False))
        avail.append(chosen)

    return TaskAssignmentProblem(
        n=n, cost=cost,
        visitor_demand=visitor_demand,
        avail=avail, max_jam=max_jam,
    )


# ─────────────────────────────────────────────────────────────
# Fitness
# ─────────────────────────────────────────────────────────────
def visitor_disproportion(chromo: Chromosome, prob: TaskAssignmentProblem) -> float:
    """Vd = (Σ_t Vt |Nt/N - Vt/V_total|) / V_total  →  skala [0,1]"""
    N       = prob.n
    V_total = prob.V_total
    sc      = {s: 0 for s in prob.shifts}
    for _, shift in chromo:
        if shift in sc:
            sc[shift] += 1
    Vd = sum(
        prob.visitor_demand[s] * abs(sc[s] / N - prob.visitor_demand[s] / V_total)
        for s in prob.shifts
    )
    return Vd / V_total


def fitness(
    chromo : Chromosome,
    prob   : TaskAssignmentProblem,
    alpha  : float = 60.0,
) -> Dict:
    """
    fitness = total_cost + alpha * (Vd + 2*avail_viol + jam_viol)

    Feasible = avail_viol==0 AND jam_viol==0
    """
    total_cost = 0.0
    avail_viol = 0
    jam_viol   = 0
    sc         = {s: 0 for s in prob.shifts}

    for i, (task, shift) in enumerate(chromo):
        total_cost += prob.cost[i][task]
        if shift in sc:
            sc[shift] += 1
        if shift not in prob.avail[i]:
            avail_viol += 1
        if prob.jam_shift.get(shift, 0) > prob.max_jam[i]:
            jam_viol += 1

    Vd      = visitor_disproportion(chromo, prob)
    penalty = alpha * (Vd + 2 * avail_viol + jam_viol)

    return {
        'fitness'    : total_cost + penalty,
        'total_cost' : total_cost,
        'penalty'    : penalty,
        'Vd'         : Vd,
        'Vd_raw'     : Vd * prob.V_total,
        'avail_viol' : avail_viol,
        'jam_viol'   : jam_viol,
        'shift_count': dict(sc),
        'ideal_count': prob.ideal_count(),
        'feasible'   : (avail_viol == 0 and jam_viol == 0),
    }


# ─────────────────────────────────────────────────────────────
# GA Operators
# ─────────────────────────────────────────────────────────────
def random_chromosome(n: int, prob: TaskAssignmentProblem) -> Chromosome:
    tasks = list(range(n))
    random.shuffle(tasks)
    return [(t, random.choice(prob.shifts)) for t in tasks]


def tournament_select(
    population : List[Chromosome],
    fit_vals   : List[float],
    k          : int = 3,
) -> Chromosome:
    candidates = random.sample(range(len(population)), k)
    best       = min(candidates, key=lambda i: fit_vals[i])
    return copy.deepcopy(population[best])


def order_crossover(
    p1 : Chromosome,
    p2 : Chromosome,
    n  : int,
) -> Tuple[Chromosome, Chromosome]:
    a = random.randint(0, n - 2)
    b = random.randint(a + 1, n - 1)

    def _ox(pa, pb):
        child = [None] * n
        for i in range(a, b + 1):
            child[i] = pa[i]
        used = {g[0] for g in child if g}
        fill = [g for g in pb if g[0] not in used]
        ptr  = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[ptr]; ptr += 1
        return child

    return _ox(p1, p2), _ox(p2, p1)


def mutate(
    chromo          : Chromosome,
    n               : int,
    prob            : TaskAssignmentProblem,
    mut_rate        : float = 0.10,
    shift_flip_rate : float = 0.10,
) -> Chromosome:
    c = list(chromo)

    if random.random() < mut_rate:
        i, j    = random.sample(range(n), 2)
        ti, si  = c[i]; tj, sj = c[j]
        c[i] = (tj, si); c[j] = (ti, sj)

    if random.random() < shift_flip_rate:
        i     = random.randint(0, n - 1)
        task, shift = c[i]
        other = [s for s in prob.shifts if s != shift]
        if other:
            c[i] = (task, random.choice(other))

    return c


# ─────────────────────────────────────────────────────────────
# GA Config & Result
# ─────────────────────────────────────────────────────────────
@dataclass
class GAConfig:
    pop_size        : int   = 120
    max_gen         : int   = 400
    mut_rate        : float = 0.12
    shift_flip_rate : float = 0.12
    tournament_k    : int   = 3
    elitism         : int   = 2
    alpha           : float = 70.0
    early_stop      : int   = 100


@dataclass
class GAResult:
    best_chromosome  : Chromosome
    best_eval        : Dict
    history_best     : List[float]
    history_avg      : List[float]
    history_feasible : List[int]
    generations_run  : int
    elapsed_time     : float
    converged_at     : Optional[int]


# ─────────────────────────────────────────────────────────────
# Main GA Loop
# ─────────────────────────────────────────────────────────────
def run_ga(
    prob      : TaskAssignmentProblem,
    config    : GAConfig,
    stop_flag = None,   # callable() → bool
    callback  = None,   # callable(gen, best_eval, hb, ha, hf)
) -> GAResult:
    n   = prob.n
    cfg = config

    population   = [random_chromosome(n, prob) for _ in range(cfg.pop_size)]
    best_chromo  = None
    best_fit     = float('inf')
    best_eval_d  = None
    no_improve   = 0
    converged_at = None

    history_best     = []
    history_avg      = []
    history_feasible = []

    t0 = time.time()

    for gen in range(cfg.max_gen):
        if stop_flag and stop_flag():
            break

        evals    = [fitness(c, prob, cfg.alpha) for c in population]
        fit_vals = [e['fitness'] for e in evals]

        bi  = int(np.argmin(fit_vals))
        gbf = fit_vals[bi]
        gaf = float(np.mean(fit_vals))
        gfe = sum(1 for e in evals if e['feasible'])

        history_best.append(gbf)
        history_avg.append(gaf)
        history_feasible.append(gfe)

        if gbf < best_fit:
            best_fit    = gbf
            best_chromo = copy.deepcopy(population[bi])
            best_eval_d = evals[bi]
            no_improve  = 0
            if best_eval_d['feasible'] and converged_at is None:
                converged_at = gen
        else:
            no_improve += 1

        if callback and gen % 5 == 0:
            callback(gen, best_eval_d, history_best, history_avg, history_feasible)

        if no_improve >= cfg.early_stop:
            break

        sorted_idx = sorted(range(cfg.pop_size), key=lambda i: fit_vals[i])
        elites     = [copy.deepcopy(population[i]) for i in sorted_idx[:cfg.elitism]]

        new_pop = list(elites)
        while len(new_pop) < cfg.pop_size:
            p1 = tournament_select(population, fit_vals, cfg.tournament_k)
            p2 = tournament_select(population, fit_vals, cfg.tournament_k)
            c1, c2 = order_crossover(p1, p2, n)
            new_pop.append(mutate(c1, n, prob, cfg.mut_rate, cfg.shift_flip_rate))
            if len(new_pop) < cfg.pop_size:
                new_pop.append(mutate(c2, n, prob, cfg.mut_rate, cfg.shift_flip_rate))
        population = new_pop[:cfg.pop_size]

    if callback:
        callback(gen, best_eval_d, history_best, history_avg, history_feasible)

    return GAResult(
        best_chromosome  = best_chromo,
        best_eval        = best_eval_d,
        history_best     = history_best,
        history_avg      = history_avg,
        history_feasible = history_feasible,
        generations_run  = gen + 1,
        elapsed_time     = time.time() - t0,
        converged_at     = converged_at,
    )
