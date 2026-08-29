# 0003: Pickle RCE in tracer/cache -> msgpack + atomic writes

**Status:** done (2026-08-24)
**Type:** security
**Tags:** `[execution]` `[rce]` `[durability]`
**Priority:** now
**Depends on:** none
**Owner:** Rohan Munshi
**Reviewer:** _unassigned_
**Prototype review:** _pending_
**Found in:** core-hardening research (2026-06-01), tracer/cache audit.

## Why

`AuraTrace` serializes every step to `.aura_trace/<session>/step_NNN_<node>.pkl`
and reads them back with `pickle.load`. `pickle.load` executes arbitrary code on
deserialization. Any trace file an attacker can write or swap (shared volume, CI
artifact, synced dir, a teammate's repro bundle) becomes RCE the moment a trace
is replayed or inspected. Traces are exactly the kind of artifact people share to
reproduce bugs — the threat is realistic. Secondary: torn writes (no atomic
rename / fsync) corrupt traces on crash.

## What's broken

`aura_state/execution/tracer.py` (and any `pickle` use in
`aura_state/memory/trajectory_cache.py`).

- `pickle.load` on trace replay (tracer.py:~62) — arbitrary code execution.
- `pickle.dump` writes `.pkl` alongside JSON — the pickle path is the dangerous
  one and is redundant with the JSON dump.
- Writes are not atomic: a crash mid-write leaves a truncated file; readers see
  partial data. No `fsync`, no temp+rename.

## Repro

```python
import pickle, os
# attacker writes a malicious step file into a session dir
class Exploit:
    def __reduce__(self):
        return (os.system, ("echo pwned > /tmp/aura_pwned",))
open(".aura_trace/sess/step_001_x.pkl","wb").write(pickle.dumps(Exploit()))
# later: any replay/inspection that pickle.loads the session dir runs it
```

## Root cause

`pickle` chosen for convenience (handles arbitrary Python objects). Trace/cache
payloads are plain data (dicts, strings, numbers, lists) — they never needed
arbitrary-object serialization, so the unsafe primitive bought nothing.

## Fix

- **Remove `pickle` entirely** from tracer and cache. Serialize with **msgpack**
  (compact, fast) or JSON. Define explicit schemas for trace step records and
  cache entries; reject unknown top-level shapes on load.
- **Atomic durable writes:** write to `path.tmp` in the same dir, `flush` +
  `os.fsync(fd)`, then `os.replace(tmp, path)`; `fsync` the directory fd after
  rename. No reader ever sees a partial file.
- **Load validation:** on read, validate against the schema; a malformed/legacy
  `.pkl` is rejected with a clear error, never deserialized.
- Migration: provide a one-shot converter for existing `.pkl` traces *only if* a
  user opts in via an explicit flag, run in a subprocess, and document the risk —
  otherwise drop legacy traces.

## Test strategy

- write a trace, read it back, assert round-trip equality (real files, no mock)
- assert no `.pkl` is produced and `pickle` is not imported in tracer/cache
- crash-safety: write to tmp, kill before rename (simulate), assert the live file is either the old complete file or absent — never truncated
- a `__reduce__` exploit object cannot round-trip (msgpack/JSON refuses it)

Add `test_tracer_no_pickle_fixes_0003`, `test_tracer_atomic_write_fixes_0003`.

## Acceptance criteria

- [ ] `pickle` not imported or called anywhere in `execution/tracer.py` or `memory/trajectory_cache.py`
- [ ] trace/cache round-trip via msgpack or JSON with explicit schema validation on load
- [ ] writes are atomic (temp -> fsync -> `os.replace` -> dir fsync); torn-write test passes
- [ ] malformed/legacy file rejected with a clear error, never deserialized as code
- [ ] regression tests `test_tracer_*_fixes_0003` added, using real files
- [ ] exploit-object repro can no longer execute on load (record in Notes)

## Notes

_record serialization choice (msgpack vs JSON), schema, and legacy-trace handling here_

## Completion (2026-08-24)
`pickle` fully removed from `execution/tracer.py` (cache never used pickle — JSON only). Traces are JSON with a `schema_version` + required-keys check on load; a hostile/legacy blob fails JSON parse or schema and raises `TraceFormatError`, never deserialized. Writes are atomic+durable: temp file → flush → `os.fsync` → `os.replace` → dir fsync. Tests: `tests/test_tracer_fixes_0003.py` (5, incl. `__reduce__` exploit rejected). Full suite 105 passed. Note: no legacy-`.pkl` migration provided — old traces are dropped (documented).
