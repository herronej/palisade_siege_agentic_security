"""
Live-agent harness adapter (WI17): model-in-the-loop SIEGE.

The default ``SessionRunner`` replays *authored* attack actions through the
real gate fast-tiers offline. ``LiveSessionRunner`` instead drives a real
``ProjectAgent``: it feeds the instance's authored prompt to the agent,
lets the agent *decide* and emit real tool calls, and routes the authored
prompt + the agent's emitted actions through the **same** gate stack -- now
with the **Q-LLM slow tier** wired -- producing the **same ``Trace``** the
existing scorer consumes. So the same corpus and BU/UA/ASR metrics measure
the deployed system end-to-end, not just the deterministic gate layer.

This is the adapter the ``Trace`` contract was designed for (see
``session_runner`` module docstring). It reuses ``SessionRunner`` wholesale:
the live path only changes *where the actions come from* (the agent),
assembling a dynamic ``Instance`` and running it through the unmodified
runner. **No gate or agent code is changed.**

CI uses ``ScriptedAgentDriver`` + a mocked Q-LLM (no model/MCP/network).
``ProjectAgentDriver`` wraps a real ``ProjectAgent`` for a live deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from siege.schemas import Action, ActionKind, Instance, Session, Turn
from siege.session_runner import SessionRunner
from siege.trace_recorder import Trace


# -----------------------------------------------------------------
# Agent driver protocol
# -----------------------------------------------------------------


@dataclass(frozen=True)
class AgentTurn:
    """What an agent did in response to one prompt.

    ``tool_calls`` are ``(tool_name, args)`` the agent attempted (routed to
    G2/G4/G5 by tool name); ``response_text`` is the final assistant text
    the scorer reads for a leak / utility check.
    """

    tool_calls: tuple[tuple[str, dict[str, Any]], ...] = ()
    response_text: str = ""


class AgentDriver(Protocol):
    """Drives one agent turn. The seam between 'what the policy emits' and
    'what the harness runs' -- a mock in CI, a real ``ProjectAgent`` live."""

    async def run_turn(
        self, prompt: str, *, message_history: list[str] | None = None
    ) -> AgentTurn: ...


class ScriptedAgentDriver:
    """A mocked agent: canned behavior keyed by a substring of the prompt.

    Models a real agent's *decisions* without a model -- the CI stand-in for
    ``ProjectAgentDriver``. ``responses`` maps a case-insensitive prompt
    substring to an ``AgentTurn``; ``default`` is used on no match (a benign
    refusal by default).
    """

    def __init__(
        self,
        responses: dict[str, AgentTurn] | None = None,
        *,
        default: AgentTurn | None = None,
    ) -> None:
        self._responses = responses or {}
        self._default = default or AgentTurn(response_text="I can't help with that.")

    async def run_turn(
        self, prompt: str, *, message_history: list[str] | None = None
    ) -> AgentTurn:
        low = prompt.lower()
        for needle, turn in self._responses.items():
            if needle.lower() in low:
                return turn
        return self._default


class ProjectAgentDriver:
    """Drives a real ``ProjectAgent`` via ``run_stream`` (live deployment only).

    Collects the agent's ``FunctionToolCallEvent``s and the terminal result
    text into an ``AgentTurn``. Lazily touches the agent so importing this
    module never pulls in the MCP/model stack. **Not exercised in CI** --
    it requires a configured model + MCP servers.
    """

    def __init__(self, project_agent: Any) -> None:
        self._agent = project_agent

    async def run_turn(
        self, prompt: str, *, message_history: list[str] | None = None
    ) -> AgentTurn:  # pragma: no cover - requires a live model + MCP
        from pydantic_ai.messages import FunctionToolCallEvent

        tool_calls: list[tuple[str, dict[str, Any]]] = []
        response_text = ""
        async for event in self._agent.run_stream(prompt):
            if isinstance(event, FunctionToolCallEvent):
                part = getattr(event, "part", event)
                name = getattr(part, "tool_name", "")
                args = getattr(part, "args", {})
                if isinstance(args, str):
                    import json

                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {"_raw": args}
                tool_calls.append((name, dict(args or {})))
            elif getattr(event, "event_kind", "") == "agent_run_result":
                result = getattr(event, "result", None)
                response_text = str(getattr(result, "output", result) or "")
        return AgentTurn(tool_calls=tuple(tool_calls), response_text=response_text)


# -----------------------------------------------------------------
# Live instance assembly
# -----------------------------------------------------------------


def _tool_action(name: str, args: dict[str, Any], *, is_attack: bool) -> Action:
    return Action(
        kind=ActionKind.TOOL_CALL,
        is_attack=is_attack,
        label=f"agent:{name}",
        payload={"tool_name": name, "args": dict(args)},
    )


async def build_live_instance(authored: Instance, driver: AgentDriver) -> Instance:
    """Drive the agent over ``authored``'s prompts; assemble a live ``Instance``.

    Each authored turn becomes: the authored actions (still gated + scored)
    + the agent's emitted ``tool_call`` actions (gated + recorded, not the
    primary scored attack) + a ``RESPONSE`` action carrying the agent's text.
    """
    new_sessions: list[Session] = []
    for session in authored.sessions:
        history: list[str] = []
        new_turns: list[Turn] = []
        for turn in session.turns:
            actions: list[Action] = list(turn.actions)
            prompt = "\n".join(
                str(a.payload.get("user_prompt", ""))
                for a in turn.actions
                if a.kind is ActionKind.PROMPT
            ).strip()
            if prompt:
                agent_turn = await driver.run_turn(prompt, message_history=list(history))
                for name, args in agent_turn.tool_calls:
                    actions.append(_tool_action(name, args, is_attack=False))
                actions.append(
                    Action(
                        kind=ActionKind.RESPONSE,
                        label=agent_turn.response_text,
                        payload={"_response_text": agent_turn.response_text},
                    )
                )
                history.append(agent_turn.response_text)
            new_turns.append(Turn(actions=tuple(actions), note=turn.note))
        new_sessions.append(Session(session_id=session.session_id, turns=tuple(new_turns)))
    return replace(authored, sessions=tuple(new_sessions))


# -----------------------------------------------------------------
# LiveSessionRunner
# -----------------------------------------------------------------


class LiveSessionRunner:
    """Run an instance model-in-the-loop, into the same ``Trace``.

    Args mirror ``SessionRunner`` plus an ``agent_driver``. ``quarantine_agents``
    wires the per-gate Q-LLM slow tier (the whole point of the live path).
    """

    def __init__(
        self,
        *,
        agent_driver: AgentDriver,
        quarantine_agents: dict[str, Any] | None = None,
        memory_store: Any | None = None,
        settings: Any | None = None,
    ) -> None:
        self._driver = agent_driver
        self._runner = SessionRunner(
            memory_store=memory_store,
            quarantine_agents=quarantine_agents,
            settings=settings,
        )

    async def run(self, instance: Instance, config: Any) -> Trace:
        live = await build_live_instance(instance, self._driver)
        return await self._runner.run(live, config)
