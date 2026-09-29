"""Autonomous Coding-Agent Harness.

A model-agnostic autonomous coding agent that reads a repository and an
issue description, plans a fix, edits code with checkpoint/rollback safety,
verifies edits, runs tests, and only commits when tests actually pass.
"""

__version__ = "3.0.0"
