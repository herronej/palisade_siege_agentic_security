"""
Synthetic salt-thermophysical-properties corpus and benign-query
workload for the G3 evaluation.

## Why synthetic

A real evaluation would index VISTA's actual molten-salt KB
(~hundreds of papers) and run gradient-optimized attacks against
a real embedding model. Both prerequisites need infrastructure
beyond the scope of this issue: the KB needs to be built locally
with `build_rag.py`, and the attacks need a GPU + the same
embedding model the deployment uses. So this harness produces a
synthetic corpus that mimics the *shape* of the real one
(scientific English about salt thermophysical properties) but is
small enough to run in a few seconds on CPU.

## What's pinned

- Deterministic generation under a fixed RNG seed. Calling
  `generate_corpus(seed=42)` always produces the same chunks and
  the same query list -- the report ships with the seed=42
  output so anyone re-running the harness sees identical
  numbers.
- Gold-chunk labels per benign query. Each query has exactly one
  "correct" chunk in the corpus; the harness uses that label to
  measure FPR (the fraction of benign queries where hybrid
  retrieval demotes the gold chunk out of the top-1 spot
  relative to vector-only).
- Chunk text is real(ish) English. The vocabulary is drawn from
  a curated list of molten-salt domain terms; sentences are
  templated but include enough lexical variation that BM25 has
  signal to work with.

## What's NOT modeled

- Real document structure (titles, citations, multi-page papers).
- Embedding-model-specific semantic similarity (we model vector
  similarity as a function of token overlap; see
  `runner.simulate_vector_similarity`).
- Stop-word distributions matching scientific literature corpora.
  The simple template generator overweights domain terms, which
  is fine for the BM25-vs-vector signal we want to measure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# -----------------------------------------------------------------
# Domain vocabulary
# -----------------------------------------------------------------
#
# The corpus is built from a small molten-salt thermophysical-
# properties vocabulary. Curated by hand so the lexical overlap
# between queries and chunks looks like real scientific text:
# specific terms ("FLiBe", "873 K", "viscosity") drive the
# semantic signal, common-bridge tokens ("the", "in", "for")
# carry the prose. The list is short on purpose -- ~40 terms is
# plenty for a 100-chunk corpus, and a longer list would dilute
# the BM25 signal in ways that obscure the defense being tested.

_SALTS = [
    "FLiBe", "FLiNaK", "NaCl", "KCl", "LiF",
    "BeF2", "ZrF4", "NaF", "Li2BeF4", "LiCl",
]

_PROPERTIES = [
    "density", "viscosity", "thermal conductivity",
    "heat capacity", "vapor pressure", "surface tension",
    "electrical conductivity", "melting point",
    "corrosion rate", "tritium solubility",
]

_TEMPERATURES = ["773 K", "873 K", "973 K", "1073 K", "1173 K"]

_CONTAINERS = [
    "Hastelloy N", "316 stainless steel", "Inconel 600",
    "nickel alloy", "graphite crucible",
]

_REACTOR_CONTEXTS = [
    "molten salt reactor", "MSR primary loop", "FHR coolant",
    "secondary salt loop", "tritium recovery system",
    "fluoride salt cooled reactor", "thorium fuel cycle",
]


# -----------------------------------------------------------------
# Templates
# -----------------------------------------------------------------
#
# Each template produces one chunk. The placeholders are filled
# from the vocabulary lists above. Three template families
# capture the lexical patterns the BM25 signal needs:
#
# - PROPERTY_VALUE: "<property> of <salt> at <temp> ..."
# - CORROSION:      "Corrosion of <container> in <salt> ..."
# - REACTOR:        "<reactor_context> ... <salt> ..."
#
# The templates are intentionally long enough (~30-50 words) so
# BM25 has multiple terms to score against; too-short chunks
# would force most of the score onto a single term.

_TEMPLATES = [
    "The {property} of {salt} at {temp} has been measured by multiple "
    "experimental campaigns and is consistent with the empirical "
    "correlation reported in MSTDB-TP. The reported uncertainty band "
    "covers a five percent range around the recommended value.",

    "Corrosion of {container} in contact with {salt} at {temp} proceeds "
    "primarily by selective dissolution of chromium and minor "
    "constituents. The corrosion rate is dominated by impurity content "
    "of the salt rather than alloy composition above a threshold purity.",

    "In the {reactor} the coolant {salt} circulates at temperatures "
    "around {temp}. Operating experience demonstrates that the {property} "
    "of the salt evolves slowly over the campaign as impurities are "
    "introduced through structural-material interactions.",

    "Experimental determination of {property} for {salt} requires "
    "careful control of the atmospheric oxygen partial pressure since "
    "trace oxidation shifts the property by several percent at "
    "operating temperatures near {temp}.",

    "The {property} of {salt} relevant to {reactor} design has been "
    "the subject of recent re-evaluation. Updated values fall within "
    "the uncertainty band of the legacy MSRE-era measurements but "
    "the recommended correlation has been refined.",
]


# -----------------------------------------------------------------
# Public dataclasses
# -----------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """A single corpus chunk with a stable identifier."""

    chunk_id: str
    text: str
    # Topic tags so attacks can target specific subsets (e.g.,
    # "all chunks about viscosity"). One template instantiation
    # produces one tag set; the corpus generator records what
    # terms were drawn so attacks can match on them.
    salt: str
    prop: str


@dataclass(frozen=True)
class BenignQuery:
    """
    A benign retrieval query with a gold-chunk label.

    `gold_chunk_id` is the chunk the corpus generator constructed
    to satisfy the query -- the retrieval baseline that hybrid
    must not regress. FPR is measured as "fraction of benign
    queries where the hybrid path demotes the gold chunk out of
    the top-1 spot relative to vector-only retrieval."
    """

    query: str
    gold_chunk_id: str
    # The salt + property pair so attack generators can craft
    # poisoned chunks that target the query without scanning the
    # query text.
    salt: str
    prop: str


@dataclass(frozen=True)
class Corpus:
    """
    The full generated corpus + query workload, returned by
    `generate_corpus`. Frozen so the runner can't mutate it
    mid-evaluation.
    """

    chunks: tuple[Chunk, ...]
    queries: tuple[BenignQuery, ...]
    seed: int


# -----------------------------------------------------------------
# Generation
# -----------------------------------------------------------------


def generate_corpus(
    *,
    seed: int = 42,
    n_chunks: int = 100,
    n_queries: int = 20,
) -> Corpus:
    """
    Build the synthetic corpus + benign-query workload.

    Deterministic: same `seed` -> same chunks and queries (the
    insertion order is RNG-driven so the seed pins both the
    content and the ordering).

    `n_chunks` is the total corpus size. Of those, `n_queries`
    chunks are designated as "gold" chunks (one per query); the
    rest are filler that BM25 has to discriminate against. The
    default 100/20 gives a 5:1 filler-to-gold ratio, which is
    enough to make naive retrieval non-trivial without bloating
    the run time.

    Args:
        seed: RNG seed. The report's headline numbers were
            generated with seed=42.
        n_chunks: total chunks to generate. Must be >= n_queries.
        n_queries: number of benign queries (and gold chunks).

    Returns:
        A `Corpus` dataclass with `chunks`, `queries`, and the
        seed used (so the report can record provenance).
    """
    if n_chunks < n_queries:
        raise ValueError(
            f"n_chunks ({n_chunks}) must be >= n_queries ({n_queries})"
        )
    rng = random.Random(seed)

    chunks: list[Chunk] = []
    queries: list[BenignQuery] = []

    # First, create n_queries (salt, property, temp) gold tuples
    # and one chunk per tuple. The temperature is included in
    # both the gold chunk's text AND the query so the gold chunk
    # has a unique three-token signature; this is what keeps the
    # benign-workload FPR honest under hybrid retrieval (a
    # synthetic corpus without the temp discriminator has filler
    # chunks colliding on (salt, prop) and inflating FPR
    # spuriously).
    gold_tuples: set[tuple[str, str, str]] = set()
    # Per-gold (salt, temp) pairs are off-limits for filler. The
    # corrosion template mentions "corrosion rate" naturally even
    # when its assigned prop differs, so two chunks sharing
    # (salt, temp) can compete on the same query even if their
    # nominal prop tags differ. Excluding (salt, temp) for filler
    # eliminates the residual benign-workload FPR that otherwise
    # comes from template-text bleeding.
    gold_salt_temp: set[tuple[str, str]] = set()
    for i in range(n_queries):
        # Pick a (salt, prop, temp) tuple not already used by an
        # earlier gold so each query maps to a distinct chunk.
        while True:
            salt = rng.choice(_SALTS)
            prop = rng.choice(_PROPERTIES)
            temp = rng.choice(_TEMPERATURES)
            if (salt, prop, temp) not in gold_tuples:
                gold_tuples.add((salt, prop, temp))
                gold_salt_temp.add((salt, temp))
                break
        # Lock the gold chunk to template 0 (the property/value
        # template) since the queries below ask for properties.
        chunk_text = _TEMPLATES[0].format(
            property=prop, salt=salt, temp=temp,
        )
        chunk_id = f"gold-{i:03d}"
        chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, salt=salt, prop=prop))
        queries.append(
            BenignQuery(
                # The temperature in the query is the discriminator
                # that lets BM25 distinguish the gold chunk from
                # filler chunks that share (salt, prop) by chance.
                query=f"What is the {prop} of {salt} at {temp}?",
                gold_chunk_id=chunk_id,
                salt=salt,
                prop=prop,
            )
        )

    # Filler chunks: random template, random salt/property/temp.
    # Avoid (salt, temp) pairs already used by gold so a filler
    # never lexically competes with a gold on the same
    # three-keyword query (even when the templates' natural
    # vocabulary -- e.g., "corrosion rate" inside the corrosion
    # template -- would otherwise bleed prop-related tokens into
    # filler chunks that don't carry that prop nominally).
    n_filler = n_chunks - n_queries
    for j in range(n_filler):
        template = rng.choice(_TEMPLATES)
        salt = rng.choice(_SALTS)
        prop = rng.choice(_PROPERTIES)
        # Reroll (salt, temp) until we miss every gold pair. The
        # pool of 10 salts x 5 temps = 50 pairs is plenty larger
        # than n_queries=20, so a collision-free pick is fast.
        for _ in range(50):
            temp = rng.choice(_TEMPERATURES)
            if (salt, temp) not in gold_salt_temp:
                break
            salt = rng.choice(_SALTS)
        container = rng.choice(_CONTAINERS)
        reactor = rng.choice(_REACTOR_CONTEXTS)
        text = template.format(
            property=prop,
            salt=salt,
            temp=temp,
            container=container,
            reactor=reactor,
        )
        chunks.append(
            Chunk(
                chunk_id=f"filler-{j:04d}",
                text=text,
                salt=salt,
                prop=prop,
            )
        )

    # Shuffle so gold chunks are not all at the top of the list
    # (which would make insertion-order tiebreakers in the merge
    # do the wrong thing).
    rng.shuffle(chunks)

    return Corpus(
        chunks=tuple(chunks),
        queries=tuple(queries),
        seed=seed,
    )
