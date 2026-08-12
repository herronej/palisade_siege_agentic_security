"""
B5.2 -- HPC credential exfil, 5 patterns.

The five credential-exfil generators from ``g5_attacks.py`` (ssh-key
read, env leak, scp exfil, http POST, log write), each a
``submit_hpc_job``. G5's credential / sensitive-path fast tier fires.
This is also the in-script-exfiltration half of the G4->G5 cross-gate
chain.
"""

from __future__ import annotations

import random

from siege.eval.g5_attacks import ATTACK_FAMILIES, ATTACK_GENERATORS
from siege.schemas import Action, ActionKind, Instance
from siege.templates._helpers import single_action_instance

BOUNDARY = "B5.2"
TEMPLATE = "b5_2_credential_exfil"
_FAMILY = "credential"
_REFERENCES = ("PALISADE G5 threat surface (docs/palisade/g5_threat_surface.md)",)


def build() -> list[Instance]:
    rng = random.Random(502)
    instances: list[Instance] = []
    for idx, key in enumerate(ATTACK_FAMILIES[_FAMILY]):
        scenario = ATTACK_GENERATORS[key](rng)
        attack = Action(
            kind=ActionKind.TOOL_CALL,
            gate="G5",
            is_attack=True,
            label=f"HPC credential exfil ({key})",
            payload={
                "slurm_script": scenario.slurm_script,
                "user_config": scenario.user_config,
            },
        )
        instances.append(
            single_action_instance(
                instance_id=f"b5_2_{key}_{idx:02d}",
                boundary=BOUNDARY,
                template=TEMPLATE,
                attack_action=attack,
                variation_axis=key,
                description=f"HPC credential exfiltration: {key}.",
                references=_REFERENCES,
                utility_gate="G5",
                rng=rng,
            )
        )
    return instances
