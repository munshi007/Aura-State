"""Task 0014 (prototype): static taint — untrusted data must not reach a sink."""
from aura_state import AuraEngine, Node, CompiledTransition, analyze_taint


class Ingest(Node):
    system_prompt = "ingest untrusted user/tool content"
    untrusted_source = True

    def handle(self, user_text, extracted_data=None, memory=None):
        return "SendEmail", {}


class Validate(Node):
    system_prompt = "validate/sanitize"
    sanitizer = True

    def handle(self, user_text, extracted_data=None, memory=None):
        return "SendEmail", {}


class SendEmail(Node):
    system_prompt = "irreversible side effect"
    dangerous_sink = True

    def handle(self, user_text, extracted_data=None, memory=None):
        return "END", {}


def test_untrusted_reaches_sink_is_violation_fixes_0014():
    e = AuraEngine()
    e.register(Ingest, SendEmail)
    e.connect([CompiledTransition(from_node=Ingest, to_node=SendEmail)])
    r = analyze_taint(e)
    assert r.verified is False
    assert len(r.violations) == 1
    v = r.violations[0]
    assert v.source == "Ingest" and v.sink == "SendEmail"
    assert v.path == ["Ingest", "SendEmail"]


def test_sanitizer_blocks_taint_fixes_0014():
    e = AuraEngine()
    e.register(Ingest, Validate, SendEmail)
    e.connect([
        CompiledTransition(from_node=Ingest, to_node=Validate),
        CompiledTransition(from_node=Validate, to_node=SendEmail),
    ])
    r = analyze_taint(e)
    assert r.verified is True
    assert r.violations == []


def test_no_source_no_violation_fixes_0014():
    class Plain(Node):
        system_prompt = "no taint"

        def handle(self, user_text, extracted_data=None, memory=None):
            return "SendEmail", {}

    e = AuraEngine()
    e.register(Plain, SendEmail)
    e.connect([CompiledTransition(from_node=Plain, to_node=SendEmail)])
    assert analyze_taint(e).verified is True


def test_taint_verdict_lands_in_contract_fixes_0014():
    e = AuraEngine()
    e.register(Ingest, SendEmail)
    e.connect([CompiledTransition(from_node=Ingest, to_node=SendEmail)])
    c = e.compile_contract()
    assert c.taint is not None
    assert c.taint.verdict == "VIOLATED"
    assert c.taint.violations[0].sink == "SendEmail"
    # The verdict is part of the content hash -> adding a sanitizer changes the contract.
    e2 = AuraEngine()
    e2.register(Ingest, Validate, SendEmail)
    e2.connect([
        CompiledTransition(from_node=Ingest, to_node=Validate),
        CompiledTransition(from_node=Validate, to_node=SendEmail),
    ])
    c2 = e2.compile_contract()
    assert c2.taint.verdict == "PROVEN"
    assert c.content_hash() != c2.content_hash()
