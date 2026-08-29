"""Aura-State command line: `aura-state ui` launches the local studio."""
import argparse
import sys
import threading
import webbrowser


def _cmd_ui(args):
    try:
        import uvicorn
    except ImportError:
        print("The Aura Studio UI needs extra packages. Install them with:\n"
              "    pip install 'aura-state[ui]'", file=sys.stderr)
        return 1
    from .ui.server import create_app

    url = f"http://{args.host}:{args.port}"
    print(f"\n  Aura Studio — real verifiers, running locally, no cloud.")
    print(f"  Open  {url}  (opening your browser…)\n")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    return 0


def _load_flow(path):
    """Load a flow from a .json export, or a .py module exposing `flow`/`FLOW`
    (a dict) or `build()` returning an AuraEngine."""
    import json
    import os
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    if path.endswith(".py"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("aura_flow_under_check", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr in ("flow", "FLOW"):
            if isinstance(getattr(mod, attr, None), dict):
                return getattr(mod, attr)
        if hasattr(mod, "build"):
            eng = mod.build()
            c = eng.compile_contract().model_dump()
            nodes = [{"id": n["name"],
                      "kind": ("sanitizer" if n.get("sanitizer") else "tool" if n.get("dangerous_sink") else "extract"),
                      "capability": ("sanitizer" if n.get("sanitizer") else "sink" if n.get("dangerous_sink")
                                     else "untrusted" if n.get("untrusted_source") else "plain"),
                      "obligations": n.get("obligations", [])} for n in c["nodes"]]
            edges = [[a, b] for a, bs in c["transitions"].items() for b in bs]
            return {"name": os.path.basename(path)[:-3], "nodes": nodes, "edges": edges, "entry": c.get("entry_node")}
        raise ValueError(f"{path}: no `flow` dict or `build()` found")
    raise ValueError(f"unsupported file type: {path} (use .json or .py)")


def _cmd_check(args):
    import json
    import logging
    logging.getLogger("aura_state").setLevel(logging.ERROR)   # clean CLI output
    from .check import check_flow

    def c(code, s):
        return s if args.no_color or not sys.stdout.isatty() else f"\033[{code}m{s}\033[0m"
    marks = {"critical": c("31", "✗"), "high": c("31", "✗"), "medium": c("33", "▲"), "low": c("33", "▲")}

    reports = []
    for path in args.paths:
        try:
            report = check_flow(_load_flow(path))
        except Exception as e:
            print(f"aura-state check: could not load {path}: {e}", file=sys.stderr)
            return 2
        reports.append((path, report))

    if args.json:
        out = {"verified": all(r.verified for _, r in reports),
               "agents": [{"path": p, **r.to_dict()} for p, r in reports]}
        print(json.dumps(out, indent=2))
        return 0 if out["verified"] else 1

    failed = 0
    for path, report in reports:
        head = f"aura-state check · {report.agent} · {report.nodes} nodes"
        head += "" if len(reports) == 1 else f"  ({path})"
        print(f"\n  {head}\n")
        if not report.findings:
            print("  " + c("32", "✓ PROVEN") + " — no findings.")
        for f in report.findings:
            loc = f" [{f.node}]" if f.node else ""
            print(f"  {marks.get(f.severity, '•')} {c('2', f.check + loc)}: {f.message}")
        if not report.verified:
            failed += 1
            blocking = sum(1 for f in report.findings if f.severity in ("critical", "high"))
            print("\n  " + c("31", f"✗ NOT PROVEN — {blocking} blocking finding(s)"))
        elif report.findings:
            print("\n  " + c("33", f"⚠ {len(report.findings)} advisory finding(s)") + " — no blocking issues.")

    if len(reports) > 1:
        ok = len(reports) - failed
        print("\n  " + "─" * 40)
        print(f"  {c('32', str(ok)+' proven')}, {c('31', str(failed)+' failed')} of {len(reports)} agents.\n")
    else:
        print()
    return 1 if failed else 0


def _cmd_version(args):
    from importlib.metadata import version
    print(version("aura-state"))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aura-state", description="Build LLM agents you can prove things about.")
    sub = parser.add_subparsers(dest="command")

    p_ui = sub.add_parser("ui", help="launch the local Aura Studio in your browser")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8155)
    p_ui.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    p_ui.set_defaults(func=_cmd_ui)

    p_c = sub.add_parser("check", help="statically verify agent designs (CI-friendly; exits non-zero if not proven)")
    p_c.add_argument("paths", nargs="+", help="flow .json files (studio export) or .py modules exposing flow/FLOW or build()")
    p_c.add_argument("--json", action="store_true", help="machine-readable JSON output")
    p_c.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    p_c.set_defaults(func=_cmd_check)

    p_v = sub.add_parser("version", help="print the installed version")
    p_v.set_defaults(func=_cmd_version)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
