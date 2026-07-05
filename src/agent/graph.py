"""LangGraph state graph: input_guard -> agent <-> tools -> output_guard -> END.

Guards are nodes with conditional edges; refusals route straight to END without
touching the main model. The output guard grants one regeneration attempt with
the violation injected as feedback, then falls back to a safe response.

Everything injectable (LLM, guard classifier, groundedness judge) has a
parameter so unit tests run with fakes and zero quota.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agent.prompts import REGENERATION_PROMPT, SYSTEM_PROMPT
from src.agent.tools import ALL_TOOLS
from src.config import load_config
from src.guardrails import input_guard as ig
from src.guardrails.output_guard import SAFE_FALLBACK, check_output

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    guard: dict            # input guard decision, logged + surfaced to the client
    chunks: list[str]      # retrieved chunks this turn, for the output guard
    regens: int            # regeneration attempts used this turn


def _last_human(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def _turn_tool_chunks(messages: list[BaseMessage]) -> list[str]:
    """Chunks retrieved since the last human message (what the model actually saw)."""
    chunks: list[str] = []
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if getattr(m, "name", None) == "search_life_info" and isinstance(m.content, str):
            if not m.content.startswith("No relevant information"):
                chunks.append(m.content)
    return chunks


def build_graph(
    llm=None,
    *,
    classify: Callable[[str], str] | None | bool = True,
    judge: Callable[[str], str] | None | bool = True,
    checkpointer=None,
    config: dict | None = None,
):
    """Compile the agent graph.

    llm       — any LangChain chat model (tests pass a fake); default: shared Gemini.
    classify  — input-guard classifier fn, True for the real flash-lite call, None to skip.
    judge     — groundedness judge fn, True for the real flash-lite call, None to skip.
    """
    cfg = config or load_config("agent")

    if llm is None:
        from src.llm import get_chat_model

        llm = get_chat_model("agent", streaming=True, config=cfg)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    if classify is True or judge is True:
        from src.llm import generate

        if classify is True:
            classify = lambda p: generate(p, role="guard", config=cfg)  # noqa: E731
        if judge is True:
            judge = lambda p: generate(p, role="judge", config=cfg)  # noqa: E731

    guard_cfg = cfg["guards"]
    max_regens = guard_cfg["output"].get("max_regenerations", 1)
    memory_msgs = cfg["agent"].get("memory_turns", 10) * 2

    # ---- nodes ------------------------------------------------------------

    def input_guard_node(state: AgentState) -> dict:
        decision = ig.check_input(
            _last_human(state["messages"]),
            max_chars=guard_cfg["input"].get("max_chars", 2000),
            classify=classify if guard_cfg["input"].get("classifier_when_uncertain") else None,
        )
        update: dict = {
            "guard": {"verdict": decision.verdict, "reason": decision.reason,
                      "category": decision.category},
            "chunks": [],
            "regens": 0,
        }
        if not decision.allowed:
            canned = {
                "refuse_injection": ig.REFUSAL_INJECTION,
                "refuse_sensitive": ig.REFUSAL_SENSITIVE,
                "redirect_off_topic": ig.REDIRECT_OFF_TOPIC,
            }[decision.verdict]
            update["messages"] = [AIMessage(content=canned)]
        return update

    def agent_node(state: AgentState) -> dict:
        # Cap history so long sessions don't grow the prompt unboundedly.
        history = state["messages"][-memory_msgs:]
        response = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT), *history])
        return {"messages": [response]}

    # handle_tool_errors: a failing tool becomes an error ToolMessage the model
    # can react to, instead of crashing the turn.
    tool_node = ToolNode(ALL_TOOLS, handle_tool_errors=True)

    def clarify_node(state: AgentState) -> dict:
        # The clarify tool is terminal: speak the question back to the user.
        question = str(state["messages"][-1].content).removeprefix("CLARIFY:").strip()
        return {"messages": [AIMessage(content=question)]}

    def output_guard_node(state: AgentState) -> dict:
        answer = str(state["messages"][-1].content)
        chunks = _turn_tool_chunks(state["messages"])
        decision = check_output(
            answer,
            chunks,
            judge=judge,
            groundedness=guard_cfg["output"].get("groundedness_check", True),
        )
        if decision.ok:
            return {"chunks": chunks}
        if state.get("regens", 0) < max_regens:
            feedback = HumanMessage(
                content=REGENERATION_PROMPT.format(answer=answer, feedback=decision.feedback())
            )
            return {"messages": [feedback], "regens": state.get("regens", 0) + 1, "chunks": chunks}
        logger.warning("output guard: fallback after %d regens (%s)", max_regens, decision.reason)
        return {"messages": [AIMessage(content=SAFE_FALLBACK)], "chunks": chunks}

    # ---- edges ------------------------------------------------------------

    def route_after_input(state: AgentState) -> str:
        return "agent" if state["guard"]["verdict"] == "allow" else END

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "output_guard"

    def route_after_tools(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last.content, str) and last.content.startswith("CLARIFY:"):
            return "clarify"
        return "agent"

    def route_after_output(state: AgentState) -> str:
        # A HumanMessage here means the guard queued regeneration feedback.
        return "agent" if isinstance(state["messages"][-1], HumanMessage) else END

    g = StateGraph(AgentState)
    g.add_node("input_guard", input_guard_node)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_node("clarify", clarify_node)
    g.add_node("output_guard", output_guard_node)

    g.set_entry_point("input_guard")
    g.add_conditional_edges("input_guard", route_after_input, {"agent": "agent", END: END})
    g.add_conditional_edges(
        "agent", route_after_agent, {"tools": "tools", "output_guard": "output_guard"}
    )
    g.add_conditional_edges("tools", route_after_tools, {"agent": "agent", "clarify": "clarify"})
    g.add_edge("clarify", END)
    g.add_conditional_edges("output_guard", route_after_output, {"agent": "agent", END: END})

    return g.compile(checkpointer=checkpointer or MemorySaver())


def recursion_limit(config: dict | None = None) -> int:
    """Graph steps allowed per turn, derived from max tool iterations
    (each iteration is agent + tools = 2 steps, plus guards and slack)."""
    cfg = config or load_config("agent")
    return cfg["agent"].get("max_tool_iterations", 5) * 2 + 6


def ask(
    graph, question: str, *, thread_id: str = "cli", config: dict | None = None
) -> dict:
    """One turn. Returns {answer, guard, chunks}."""
    from langgraph.errors import GraphRecursionError

    try:
        state = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": recursion_limit(config),
            },
        )
    except GraphRecursionError:
        logger.warning("recursion limit hit; returning safe fallback")
        return {"answer": SAFE_FALLBACK, "guard": {"verdict": "allow"}, "chunks": []}
    return {
        "answer": str(state["messages"][-1].content),
        "guard": state.get("guard", {}),
        "chunks": state.get("chunks", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask AI Rababb a question.")
    parser.add_argument("question")
    parser.add_argument("-v", "--verbose", action="store_true", help="show node/tool trace")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    graph = build_graph()
    if args.verbose:
        from langgraph.errors import GraphRecursionError

        try:
            for event in graph.stream(
                {"messages": [HumanMessage(content=args.question)]},
                config={"configurable": {"thread_id": "cli"}, "recursion_limit": recursion_limit()},
            ):
                for node, update in event.items():
                    for m in update.get("messages", []):
                        kind = type(m).__name__
                        calls = getattr(m, "tool_calls", None)
                        detail = f" tool_calls={[c['name'] for c in calls]}" if calls else ""
                        print(f"[{node}] {kind}{detail}: {str(m.content)[:120]}")
        except GraphRecursionError:
            print(f"\n{SAFE_FALLBACK}")
            return
    else:
        print(ask(graph, args.question)["answer"])


if __name__ == "__main__":
    main()
