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

    p_v = sub.add_parser("version", help="print the installed version")
    p_v.set_defaults(func=_cmd_version)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
