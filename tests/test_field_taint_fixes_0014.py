"""Full 0014: field-level taint. A clean field passes a sink untouched; only a
tainted field reaching the sink is a violation. Precision node-level can't express."""
from pydantic import BaseModel

from aura_state import AuraEngine, Node, CompiledTransition, analyze_field_taint


class Lead(BaseModel):
    note: str = ""        # free text from the user — untrusted
    account_id: str = ""  # looked up internally — trusted


def _wire(*nodes_and_edges):
    e = AuraEngine()
    return e


def test_only_tainted_field_triggers_fixes_0014():
    # Ingest marks `note` untrusted (not account_id). SendEmail is a sink on
    # `account_id` only -> the tainted field never reaches the sink arg -> safe.
    class Ingest(Node):
        system_prompt = "ingest"
        untrusted_fields = ["note"]

        def handle(self, u, extracted_data=None, memory=None):
            return "Send", {}

    class Send(Node):
        system_prompt = "send"
        sink_fields = ["account_id"]

        def handle(self, u, extracted_data=None, memory=None):
            return "END", {}

    e = AuraEngine()
    e.register(Ingest, Send)
    e.connect([CompiledTransition(from_node=Ingest, to_node=Send)])
    r = analyze_field_taint(e)
    assert r.verified is True          # tainted `note` doesn't reach the account_id sink

    # Node-level would have flagged this (whole node untrusted -> whole sink):
    # field-level is strictly more precise.


def test_tainted_field_reaches_matching_sink_fixes_0014():
    class Ingest(Node):
        system_prompt = "ingest"
        untrusted_fields = ["note"]

        def handle(self, u, extracted_data=None, memory=None):
            return "Send", {}

    class Send(Node):
        system_prompt = "send"
        sink_fields = ["note"]         # the dangerous action consumes `note`

        def handle(self, u, extracted_data=None, memory=None):
            return "END", {}

    e = AuraEngine()
    e.register(Ingest, Send)
    e.connect([CompiledTransition(from_node=Ingest, to_node=Send)])
    r = analyze_field_taint(e)
    assert r.verified is False
    v = r.violations[0]
    assert v.field == "note" and v.source == "Ingest" and v.sink == "Send"
    assert v.path == ["Ingest", "Send"]


def test_field_specific_sanitizer_fixes_0014():
    class Ingest(Node):
        system_prompt = "ingest"
        untrusted_fields = ["note", "subject"]

        def handle(self, u, extracted_data=None, memory=None):
            return "Clean", {}

    class Clean(Node):
        system_prompt = "clean only note"
        sanitizes_fields = ["note"]    # clears note, NOT subject

        def handle(self, u, extracted_data=None, memory=None):
            return "Send", {}

    class Send(Node):
        system_prompt = "send"
        dangerous_sink = True          # any tainted field is dangerous here

        def handle(self, u, extracted_data=None, memory=None):
            return "END", {}

    e = AuraEngine()
    e.register(Ingest, Clean, Send)
    e.connect([
        CompiledTransition(from_node=Ingest, to_node=Clean),
        CompiledTransition(from_node=Clean, to_node=Send),
    ])
    r = analyze_field_taint(e)
    # `note` was cleaned; `subject` was not -> still a violation, and it names
    # exactly the field that leaked.
    assert r.verified is False
    leaked = {v.field for v in r.violations}
    assert leaked == {"subject"}


def test_extracts_schema_fields_are_untrusted_by_default_fixes_0014():
    # untrusted_source with a schema -> every extracted field is untrusted.
    class Extract(Node):
        system_prompt = "extract"
        extracts = Lead
        untrusted_source = True

        def handle(self, u, extracted_data=None, memory=None):
            return "Send", {}

    class Send(Node):
        system_prompt = "send"
        sink_fields = ["note"]

        def handle(self, u, extracted_data=None, memory=None):
            return "END", {}

    e = AuraEngine()
    e.register(Extract, Send)
    e.connect([CompiledTransition(from_node=Extract, to_node=Send)])
    r = analyze_field_taint(e)
    assert r.verified is False
    assert r.violations[0].field == "note"


def test_field_taint_lands_in_contract_fixes_0014():
    class Ingest(Node):
        system_prompt = "ingest"
        untrusted_fields = ["note"]

        def handle(self, u, extracted_data=None, memory=None):
            return "Send", {}

    class Send(Node):
        system_prompt = "send"
        sink_fields = ["note"]

        def handle(self, u, extracted_data=None, memory=None):
            return "END", {}

    e = AuraEngine()
    e.register(Ingest, Send)
    e.connect([CompiledTransition(from_node=Ingest, to_node=Send)])
    c = e.compile_contract()
    assert c.taint.verdict == "VIOLATED"
    assert c.taint.violations[0].field == "note"
