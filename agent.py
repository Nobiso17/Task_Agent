"""
Task Planning & Execution Agent — built with LangGraph + Claude
================================================================
The agent receives a high-level goal and runs through three phases:
  1. PLAN   — breaks the goal into concrete sub-tasks
  2. EXECUTE — runs each sub-task using available tools
  3. SUMMARIZE — compiles the results into a final report

Graph layout:
  planner → executor → (loop back if tasks remain) → summarizer → END
"""

from __future__ import annotations

import json
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from tools import TOOLS, run_tool

# ── Model ──────────────────────────────────────────────────────────────────────

llm = ChatAnthropic(model="claude-opus-4-5", max_tokens=2048)

# ── State ──────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    goal: str                          # original user goal
    tasks: list[dict]                  # [{"id":1, "description":"...", "done":false}]
    current_task_index: int            # which task we're on
    task_results: list[str]            # one result string per completed task
    messages: Annotated[list, add_messages]  # full conversation history
    final_report: str                  # produced by summarizer


# ── Node: Planner ──────────────────────────────────────────────────────────────

def planner(state: AgentState) -> dict:
    """Break the goal into a numbered list of concrete sub-tasks."""
    print("\n📋  PLANNER — decomposing goal…")

    tool_names = ", ".join(t["name"] for t in TOOLS)
    system = SystemMessage(content=(
        "You are a task planner. Given a goal, decompose it into 3-5 concrete, "
        "actionable sub-tasks. You have access to these tools: "
        f"{tool_names}. "
        "Reply ONLY with a JSON array like:\n"
        '[{"id":1,"description":"..."},{"id":2,"description":"..."}]'
    ))
    user = HumanMessage(content=f"Goal: {state['goal']}")

    response: AIMessage = llm.invoke([system, user])
    raw = response.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    tasks = json.loads(raw)
    for t in tasks:
        t["done"] = False

    print(f"   → {len(tasks)} tasks planned")
    for t in tasks:
        print(f"     {t['id']}. {t['description']}")

    return {
        "tasks": tasks,
        "current_task_index": 0,
        "task_results": [],
        "messages": [system, user, response],
    }


# ── Node: Executor ─────────────────────────────────────────────────────────────

EXECUTOR_SYSTEM = SystemMessage(content=(
    "You are an execution agent. You are given one task to complete. "
    "Use the available tools if helpful, then write a concise result summary.\n\n"
    "Available tools and their usage:\n"
    + "\n".join(
        f"  • {t['name']}: {t['description']}  →  call as: TOOL:{t['name']}:<input>"
        for t in TOOLS
    )
    + "\n\nIf you call a tool, write exactly one line starting with 'TOOL:' "
    "and nothing else on that line. After seeing the tool result, write your "
    "final task summary starting with 'RESULT:'."
))

def executor(state: AgentState) -> dict:
    """Execute the current task, optionally calling a tool."""
    idx = state["current_task_index"]
    task = state["tasks"][idx]
    print(f"\n⚙️   EXECUTOR — task {task['id']}: {task['description']}")

    user_msg = HumanMessage(content=f"Task: {task['description']}")
    history = [EXECUTOR_SYSTEM, user_msg]

    # Allow the LLM one round of tool use
    for _ in range(3):
        response: AIMessage = llm.invoke(history)
        text = response.content.strip()
        history.append(response)

        # Check if model wants to call a tool
        tool_line = next((l for l in text.splitlines() if l.startswith("TOOL:")), None)
        if tool_line:
            _, tool_name, tool_input = tool_line.split(":", 2)
            tool_name = tool_name.strip()
            tool_input = tool_input.strip()
            print(f"   🔧 calling tool '{tool_name}' with: {tool_input!r}")
            tool_result = run_tool(tool_name, tool_input)
            print(f"   ✅ tool result: {tool_result}")
            history.append(HumanMessage(content=f"Tool result: {tool_result}"))
            continue

        # Extract RESULT line
        result_line = next((l for l in text.splitlines() if l.startswith("RESULT:")), None)
        result = result_line[len("RESULT:"):].strip() if result_line else text

        print(f"   📝 result: {result}")

        # Mark task done
        updated_tasks = [
            {**t, "done": True} if t["id"] == task["id"] else t
            for t in state["tasks"]
        ]

        new_results = state["task_results"] + [f"Task {task['id']} — {task['description']}: {result}"]

        return {
            "tasks": updated_tasks,
            "current_task_index": idx + 1,
            "task_results": new_results,
            "messages": history[2:],  # skip system msg already in history
        }

    # Fallback
    return {
        "tasks": state["tasks"],
        "current_task_index": idx + 1,
        "task_results": state["task_results"] + [f"Task {task['id']} — no result"],
        "messages": history[2:],
    }


# ── Node: Summarizer ───────────────────────────────────────────────────────────

def summarizer(state: AgentState) -> dict:
    """Compile all task results into a final report."""
    print("\n📊  SUMMARIZER — compiling report…")

    results_text = "\n".join(state["task_results"])
    system = SystemMessage(content=(
        "You are a report writer. Given a goal and a list of completed task results, "
        "write a clear, concise final report (3-5 sentences) summarising what was "
        "accomplished and any key findings."
    ))
    user = HumanMessage(content=(
        f"Goal: {state['goal']}\n\nCompleted tasks:\n{results_text}"
    ))

    response: AIMessage = llm.invoke([system, user])
    report = response.content.strip()
    print(f"\n{'='*60}\n📄  FINAL REPORT\n{'='*60}\n{report}\n{'='*60}\n")

    return {"final_report": report, "messages": [system, user, response]}


# ── Routing ────────────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Loop back to executor while tasks remain, else move to summarizer."""
    if state["current_task_index"] < len(state["tasks"]):
        return "execute"
    return "summarize"


# ── Build Graph ────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("summarizer", summarizer)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", should_continue, {
        "execute": "executor",
        "summarize": "summarizer",
    })
    graph.add_edge("summarizer", END)

    return graph.compile()


# ── Run ────────────────────────────────────────────────────────────────────────

app = build_graph()


def run(goal: str) -> str:
    """Run the agent against a goal and return the final report."""
    initial_state: AgentState = {
        "goal": goal,
        "tasks": [],
        "current_task_index": 0,
        "task_results": [],
        "messages": [],
        "final_report": "",
    }
    final_state = app.invoke(initial_state)
    return final_state["final_report"]


if __name__ == "__main__":
    import sys
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Research best practices for Python project structure, "
        "create a project outline, and write a getting-started checklist."
    )
    run(goal)
