"""Task 0013: counterexample-guided replanning drives the plan to PROVEN,
or aborts with an explicit unresolved violation — never a silent pass."""
from aura_state import (
    AuraEngine, Node, CompiledTransition,
    reachability, taint_to_repair, ctl_to_repair,
)


class Ingest(Node):
    system_prompt = "untrusted ingest"
    untrusted_source = True

    def handle(self, user_text, extracted_data=None, memory=None):
        return "SendEmail", {}


class SendEmail(Node):
    system_prompt = "dangerous sink"
    dangerous_sink = True

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {}


def test_replan_converges_taint_fixes_0013():
    # An unsafe design (untrusted -> sink) is repaired by inserting a sanitizer
    # and re-verifies safe within the budget.
    e = AuraEngine()
    e.register(Ingest, SendEmail)
    e.connect([CompiledTransition(from_node=Ingest, to_node=SendEmail)])
    assert e.analyze_taint().verified is False   # starts violated

    result = e.repair()                          # default deterministic repair
    assert result.verified is True
    assert result.iterations >= 1
    assert e.analyze_taint().verified is True     # now provably safe
    # A sanitizer node was inserted on the path.
    assert any(getattr(n, "sanitizer", False) for n in e._nodes.values())


def test_replan_converges_ctl_reachability_fixes_0013():
    # A target that is unreachable from the entry is repaired by adding an edge.
    class Start(Node):
        system_prompt = "start"

        def handle(self, user_text, extracted_data=None, memory=None):
            return "END", {}

    class Orphan(Node):
        system_prompt = "unreachable target"

        def handle(self, user_text, extracted_data=None, memory=None):
            return "END", {}

    e = AuraEngine()
    e.register(Start, Orphan)          # no edge to Orphan -> unreachable
    props = [{"description": "Orphan reachable", "formula": reachability("Orphan")}]

    result = e.repair(properties=props, check_taint=False)
    assert result.verified is True
    assert "Orphan" in e._transitions.get("Start", [])   # edge added


def test_replan_aborts_on_unrepairable_fixes_0013():
    # A repair_fn that always declines must abort with the unresolved violation,
    # never silently report success.
    e = AuraEngine()
    e.register(Ingest, SendEmail)
    e.connect([CompiledTransition(from_node=Ingest, to_node=SendEmail)])

    def decline(engine, signal):
        return False

    result = e.repair(decline, max_iterations=3)
    assert result.verified is False
    assert result.unresolved                       # explicit, not empty
    assert any("SendEmail" in u for u in result.unresolved)


def test_adapters_map_counterexamples_fixes_0013():
    # taint adapter
    e = AuraEngine()
    e.register(Ingest, SendEmail)
    e.connect([CompiledTransition(from_node=Ingest, to_node=SendEmail)])
    v = e.analyze_taint().violations[0]
    sig = taint_to_repair(v)
    assert sig.kind == "taint" and sig.detail["sink"] == "SendEmail"
    assert sig.detail["path"] == ["Ingest", "SendEmail"]

    # ctl adapter: reachability violation carries the target
    class Start(Node):
        system_prompt = "s"

        def handle(self, user_text, extracted_data=None, memory=None):
            return "END", {}

    class Orphan(Node):
        system_prompt = "o"

        def handle(self, user_text, extracted_data=None, memory=None):
            return "END", {}

    e2 = AuraEngine()
    e2.register(Start, Orphan)
    from aura_state.verification.temporal_verifier import PropertyResult
    vr = [r for r in e2.verify([{"description": "reach", "formula": reachability("Orphan")}])
          if r.result != PropertyResult.PROVEN][0]
    csig = ctl_to_repair(vr)
    assert csig.kind == "ctl" and csig.detail["target"] == "Orphan"
