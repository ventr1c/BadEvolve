"""Agent-optimizer (design B): a READ-ONLY tool-using agent proposes the poison, deterministically.

Flow per round (driver: `scripts/run_agent_optimization.py`):
  measure (k-repeat evolve) -> route -> `optimizer_harness.propose_spec` (headless read-only
  `claude -p` reads that round's evidence, emits ONE JSON spec) -> `pipeline.materialize_poison`
  stamps it deterministically -> re-measure -> accept iff joint success J improved, else rollback.
No LLM ever writes the poison files; the agent only reads evidence and returns JSON.

THREE ARMS, selected by the driver's `--gate-mode`. Ours vs the clean baseline differ in ONE
dimension (feedback granularity); the third arm isolates the two halves of that difference:

  factored (DEFAULT, ours)   seed carries gate-tagged cues (induction / activation) plus the rule
                             r; the optimizer sees per-gate failure rates eps{I,A,F}, is told the
                             binding gate (argmax eps, upstream-wins) and edits only that gate's
                             component.  Prompt: `optimizer_prompt.build_optimizer_prompt`.
  single (A1 BASELINE)       seed carries ONLY r (no per-gate cues); the optimizer sees ONE
                             aggregate score (J), no gate list, no per-gate eps, no routing, and
                             rewrites r freely. Its reads go through the READ WHITELIST, so no
                             per-gate artifact is even discoverable.  Prompt:
                             `optimizer_prompt.build_single_objective_prompt`.
  single_gatevis             r-only seed and no routing, like `single`, but the whole run dir stays
                             readable, so the agent CAN find the per-gate record: "sees the gates,
                             is not routed by them". This started as a leak in `single` (the agent
                             cited `measurement.json`'s eps); it is kept as a deliberate arm
                             because it separates *having* the decomposition from *being routed by*
                             it -- the two things `factored` bundles together.

Everything else is shared by construction: victim, backbone, poison budget (n_base x (n_trig +
n_clean)), k-repeat measurement, the accept/rollback rule, the evidence paths handed to the agent,
and the final k-repeat eval on the disjoint test split. That is what makes the comparison a claim
about gate factorisation rather than about budget or scaffolding.

READ WHITELIST. `optimizer_harness.propose_spec(allow=[...])` materializes a symlink workspace
containing exactly the whitelisted paths and points the agent there instead of the whole run dir,
so what it may read is granted positively. Symlinks can be walked out of (`<link>/../..` reaches
the real run dir), so it shapes discoverability rather than sandboxing: anything that must stay
unreadable is ALSO written outside the run dir (`runs/<run>__private_gates/`).

The two prompt builders are kept SEPARATE (not one branching template) because a leak of gate
structure into the baseline would silently invalidate the comparison;
`tests/test_single_objective_baseline.py` pins the isolation.
"""
