"""Regression tests for task 0006: bandit router (Thompson) + CTL feasibility filter."""
from aura_state.core.engine import AuraEngine, Node


class Start(Node):
    system_prompt = "s"

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {}


class Good(Node):
    system_prompt = "g"

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {}


class Bad(Node):
    system_prompt = "b"

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {}


def _engine(seed=0):
    e = AuraEngine(route_seed=seed)
    e.register(Start, Good, Bad)
    e._transitions["Start"] = ["Good", "Bad"]
    return e


def test_router_feasibility_fixes_0006():
    e = _engine()
    # Make even a great-looking edge infeasible; it must never be selected.
    e.adaptive_graph.record_edge_outcome("Start", "Bad", success=True)
    for _ in range(50):
        e.adaptive_graph.record_edge_outcome("Start", "Bad", success=True)
    e.set_feasibility_filter(lambda frm, to: to != "Bad")
    picks = {e._route_select("Start", {}) for _ in range(50)}
    assert "Bad" not in picks
    assert picks == {"Good"}


def test_router_thompson_convergence_fixes_0006():
    e = _engine(seed=42)
    # Good edge mostly succeeds, Bad edge mostly fails.
    for _ in range(40):
        e.adaptive_graph.record_edge_outcome("Start", "Good", success=True)
        e.adaptive_graph.record_edge_outcome("Start", "Bad", success=False)
    picks = [e._route_select("Start", {}) for _ in range(200)]
    good = picks.count("Good")
    assert good > 180  # concentrates on the clearly-better arm


def test_router_non_stationarity_fixes_0006():
    e = _engine(seed=7)
    # Phase 1: Good is best.
    for _ in range(60):
        e.adaptive_graph.record_edge_outcome("Start", "Good", success=True)
        e.adaptive_graph.record_edge_outcome("Start", "Bad", success=False)
    assert [e._route_select("Start", {}) for _ in range(50)].count("Good") > 40

    # Phase 2: Good starts failing hard, Bad starts winning. Discounting must
    # let the router shift within a bounded number of rounds.
    for _ in range(80):
        e.adaptive_graph.record_edge_outcome("Start", "Good", success=False)
        e.adaptive_graph.record_edge_outcome("Start", "Bad", success=True)
    shifted = [e._route_select("Start", {}) for _ in range(50)].count("Bad")
    assert shifted > 40


def test_router_infeasible_all_returns_end_fixes_0006():
    e = _engine()
    e.set_feasibility_filter(lambda frm, to: False)
    assert e._route_select("Start", {}) == "END"
