"""
Attack families that plug into the WI13 adaptive substrate.

Each family is a thin composition over the substrate (``redteam/*``) -- it
adds *no* new infrastructure and, like the rest of ``redteam``, is strictly
read-only against the deployed gates: it reaches them only through the
``SessionRunner`` / ablation harness and never imports ``palisade.gates``
(the read-only guard in ``tests/test_readonly.py`` pins this).

Families:

- ``embedding/`` -- the A family (WI14): gradient-free EmbeddingGemma-space
  attacks against the G3 RAG boundary (A1 natural-norm poisoning, A2
  hybrid-retrieval seam, A3 contract-aware data-value poisoning).
- ``llm/``  -- the B family (WI15, pending): LLM-as-optimizer attacks.
- ``code/`` -- the C family (WI16, pending): correctness sabotage.
"""
