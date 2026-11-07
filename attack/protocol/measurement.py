"""Typed missingness for gate / objective values: None means NOT MEASURED, never 0.

Every "phantom" number this project has shipped came from `x or 0` on a value that was absent:
`I=1.000` printed for an unmeasured gate, F=1.0 (fully failing) for a degradation round whose
detector had nothing to measure, and a rollback that accepted any proposal when the incumbent J
was missing. The helpers here are the only sanctioned way to turn a possibly-missing value into a
number, a comparison, or a display string, so the None case is always handled on purpose.
"""
from __future__ import annotations

GATES = ("induction", "activation", "following")
LETTER = {"induction": "I", "activation": "A", "following": "F"}


def failure_rate(success):
    """eps = 1 - success, preserving None."""
    return None if success is None else round(1 - success, 3)


def success_rate(eps):
    return None if eps is None else 1 - eps


def fmt(v, spec=".2f") -> str:
    return "NA" if v is None else format(v, spec)


def gate_delta(eps_before: dict, eps_after: dict) -> str:
    """`I 0.40->0.60 A NA->1.00 ...` in SUCCESS rates; NA for an unmeasured side."""
    return " ".join(f"{LETTER[g]} {fmt(success_rate(eps_before.get(g)))}->"
                    f"{fmt(success_rate(eps_after.get(g)))}" for g in GATES)


def gates_worse(eps_before: dict, eps_after: dict, tol: float = 1e-9) -> list[str]:
    """Letters of gates whose SUCCESS rate dropped. A gate unmeasured on either side is not
    comparable, so it is never reported as worse (nor as better)."""
    out = []
    for g in GATES:
        b, a = eps_before.get(g), eps_after.get(g)
        if b is not None and a is not None and (1 - a) < (1 - b) - tol:
            out.append(LETTER[g])
    return out


def accept(j_new, j_incumbent, tol: float = 1e-9, *, explore_at_zero: bool = False,
           tie_new=None, tie_incumbent=None, min_gain: float = 0.0) -> bool:
    """Rollback verdict. A proposal with NO measurement is never accepted. Against an incumbent with
    no measurement, any measured proposal is accepted (there is nothing to be worse than).

    `explore_at_zero` (opt-in) accepts a proposal while BOTH scores are zero. Strict `>` cannot leave
    the zero band: observed on rq1_ae_gptoss_ours, five consecutive rounds of `J 0.0->0.0 (rb)`, and on
    the WikiTQ x SkillGrad cells. Every round then re-proposes from the SAME committed policy, so the
    edit chain never compounds and the optimizer burns its whole budget on one starting point.

    This is deliberately NOT the default, and it is NOT `--no-rollback` (which accepts everything and
    therefore also accepts regressions). It is the middle policy the comparison needs: random-walk while
    there is no signal, strict hill-climbing the moment there is one. Whether it actually helps is an
    empirical question -- it must be measured against the current policy at equal budget, not assumed,
    and "we showed the agent more gates" is not evidence either way.

    `tie_new` / `tie_incumbent` (opt-in, 2026-10-20) break an EXACT tie on J with a secondary score,
    which the QA driver sets to the strict induction rate (`--tie-break strict_induction`). A tied
    proposal is accepted only if that score strictly improves; an unmeasured incumbent counts as 0.
    The secondary must be polarity-aware: with the pre-10-20 probe, rq1_wtq_sg_sonnet_ours_m16's I=0.6
    round was 5/5 anti-injection defenses, and a tie-break on that would have committed them.

    `min_gain` (opt-in, 2026-10-20; the QA driver's --accept-min-evolves N sets it to N/k) requires an
    improvement of at least that much on J. Per-evolve outcomes are near-binary (H ~ 1 or 0), so with k=5 a
    0.2 step is ONE evolve's coin flip: pilotfp T2S Sonnet committed J=0.235 and tested at 0.000. Ties and
    the zero band keep their own rules above; an exact tie is still a tie.
    """
    if j_new is None:
        return False
    if j_incumbent is None:
        return True
    if explore_at_zero and j_new <= tol and j_incumbent <= tol:
        # Both at zero: no information to hill-climb on, so let the edit chain move rather than
        # restarting from the same policy. Once either side is non-zero, strict improvement resumes.
        return True
    if tie_new is not None and abs(j_new - j_incumbent) <= tol:
        return tie_new > (tie_incumbent or 0.0) + tol
    return j_new > j_incumbent + max(tol, min_gain - tol)


BAND_RULE = "strict_induction_band"


def band_verdict(j_new, j_incumbent, *, delta_succeeding, k: int, n: int, strict_new, strict_incumbent,
                 tol: float = 1e-9):
    """--tie-break strict_induction_band (2026-10-21, user decision): (accepted, branch).

    --accept-min-evolves N plus an exact-tie-only strict_induction tie-break left a dead zone: a proposal whose
    J moved within noise (0 -> 0.02, +-1 succeeding evolve) was rejected even when strict induction rose
    0.4 -> 0.8, so stage progress could not accumulate exactly where ASR is near 0. With dS = the change in
    succeeding evolves and dJ = J_new - J_incumbent:
      clear gain   dS >= N or dJ >= N/k            -> accept  (today's min-gain, f24b1b0 semantics)
      clear loss   dS <= -N or dJ <= -N/k          -> reject
      noise band   otherwise, exact ties included  -> accept iff strict induction STRICTLY rises (an unmeasured
                                                      incumbent counts as 0; the polarity-aware strict probe)
    With N = 0 the band is the exact tie alone, i.e. strict_induction. A proposal with no measurement is never
    accepted; against an unmeasured incumbent any measured proposal is (nothing to be worse than). Degradation
    passes its own analogues (succeeding = DASR >= 0.5 or a gain over the baseline; J = 1 - TACC). The FINAL
    commit is unchanged: argmax measured J, an exact tie broken by strict induction.
    """
    if j_new is None:
        return False, "unmeasured"
    if j_incumbent is None:
        return True, "gain"
    dj = j_new - j_incumbent
    ds = delta_succeeding if delta_succeeding is not None else 0
    n = int(n or 0)
    thr = n / k if n else 0.0
    if (n and ds >= n) or (dj >= thr - tol if n else dj > tol):
        return True, "gain"
    if (n and ds <= -n) or (dj <= -thr + tol if n else dj < -tol):
        return False, "loss"
    return (strict_new is not None and strict_new > (strict_incumbent or 0.0) + tol), "band"
