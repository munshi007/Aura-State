"""Static detection of MCP tool-description poisoning.

The MCP `description` field is an unsanitized attack surface: a malicious or
compromised server can embed instructions in what looks like help text, and an
agent will follow them (the CSA 2026 tool-poisoning class; ~200k vulnerable
instances). Aura already *imports* the tool surface — this scans each tool's
description for the tell-tales of an injected directive, before the agent ever
connects to the server.

Heuristic and honest: it flags *suspicious* text, not a proven exploit, and can
false-positive on a legitimately imperative description — so the clear-attack
patterns are blocking (high) and the softer ones are advisory (medium). It never
executes or fetches anything.
"""
from __future__ import annotations

import re
from typing import List, Tuple

# (label, severity, regex). severity: "high" = clear injected directive (blocking);
# "medium" = suspicious, advisory.
_PATTERNS: List[Tuple[str, str, "re.Pattern[str]"]] = [
    ("injected-instruction", "high",
     re.compile(r"(?i)\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|any)\b.{0,25}\b(instruction|prompt|rule|message)s?\b")),
    ("exfil-directive", "high",
     re.compile(r"(?i)\b(send|forward|email|post|upload|exfiltrate|leak|transmit)\b.{0,50}\b(to\s+\S+@|to\s+https?://|attacker|external|to the address|webhook)\b")),
    ("hidden-system-instruction", "high",
     re.compile(r"(?i)(<\s*/?\s*system\s*>|\[system\]|note to (the )?(assistant|ai|model)|(the )?(assistant|ai|model)\s+(should|must)\s+(always|never|secretly))")),
    ("directive-to-model", "medium",
     re.compile(r"(?i)\b(you must|you should always|do not (tell|mention|reveal|inform)|instead of|before (calling|using|responding)|without telling|secretly)\b")),
    ("credential-solicitation", "medium",
     re.compile(r"(?i)\b(api[_-]?key|password|secret|access token|credential|ssh key|private key|\.env)\b")),
]


def scan_description(text: str) -> List[Tuple[str, str]]:
    """Return [(label, severity), ...] for the poisoning patterns matched, if any."""
    if not text or not isinstance(text, str):
        return []
    return [(label, sev) for label, sev, rx in _PATTERNS if rx.search(text)]
