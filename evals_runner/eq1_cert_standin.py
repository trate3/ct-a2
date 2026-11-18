"""Agent and tap drivers for the certification timing: they take the real model files as artifacts and run the mock program and tap.

The protocol hashes every artifact file of a package (twice), so the hashing covers the real model bytes. The evaluation
inside the session runs the repository's mock program and mock tap on a few mock items; the timing script subtracts it.
"""
from __future__ import annotations

from ct_core.agent import Agent
from ct_core.interfaces import AgentDriver, TapDriver
from evals.basic_mocked import mocks
from evals.basic_mocked.mocks import MockProgram, MockTap


class StandInAgentDriver(AgentDriver):
    DRIVER = "eq1-standin-agent/v1"

    @classmethod
    def code_paths(cls):
        return (mocks.__file__,)

    @classmethod
    def from_package(cls, config, artifacts):
        if not isinstance(config.get("items"), list):
            raise ValueError("malformed stand-in agent package")
        return cls(config["items"])        # the model files are committed by the protocol, not loaded here

    def __init__(self, items):
        self.items = items

    def new_agent(self, randomness):
        return Agent.model_only(MockProgram(self.items))


class StandInTapDriver(TapDriver):
    DRIVER = "eq1-standin-tap/v1"

    @classmethod
    def code_paths(cls):
        return (mocks.__file__,)

    @classmethod
    def from_package(cls, config, artifacts):
        return cls()

    def new_tap(self, randomness):
        return MockTap()
