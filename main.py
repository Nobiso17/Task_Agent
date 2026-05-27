"""
main.py — interactive CLI for the Task Agent
=============================================
Run:
    python main.py                          # uses the built-in demo goal
    python main.py "your custom goal here"  # pass your own goal
"""

import sys
from agent import run

DEMO_GOAL = (
    "Research best practices for Python project structure, "
    "create a project outline, and write a getting-started checklist."
)

if __name__ == "__main__":
    goal = " ".join(sys.argv[1:]).strip() or DEMO_GOAL
    print(f"\n🎯  Goal: {goal}\n")
    report = run(goal)
    print("\n✅  Done.")
