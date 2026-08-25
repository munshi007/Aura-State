# Publishing `aura-state` to PyPI

The package builds cleanly and installs from a wheel in a fresh environment.
Uploading needs **your** PyPI account + API token — do it once, then
`pip install aura-state` works for everyone.

## One-time setup
1. Create a PyPI account: https://pypi.org/account/register/
2. Confirm the name **aura-state** is free: https://pypi.org/project/aura-state/
   (404 = available). If taken, change `name` in `pyproject.toml`.
3. Create an API token: PyPI → Account settings → API tokens → "Add API token"
   (scope: entire account for the first upload, or project-scoped after).

## Build (already verified working)
```bash
python -m pip install --upgrade build twine
rm -rf dist
python -m build            # -> dist/aura_state-0.2.0.tar.gz + .whl
python -m twine check dist/*
```

## (Recommended) dry run on TestPyPI first
```bash
python -m twine upload --repository testpypi dist/*
# username: __token__     password: <your TestPyPI token>
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ aura-state
```

## Publish to real PyPI
```bash
python -m twine upload dist/*
# username: __token__     password: <your PyPI token>
```
Then verify from a clean environment:
```bash
python -m venv /tmp/t && /tmp/t/bin/pip install aura-state
/tmp/t/bin/python -c "import aura_state; print('ok')"
```

## Releasing a new version later
1. Bump `version` in `pyproject.toml` (PyPI rejects re-uploading the same version).
2. `rm -rf dist && python -m build && python -m twine upload dist/*`.
3. Tag it: `git tag v0.2.0 && git push --tags`.

## Notes
- Do NOT commit `dist/` (already in `.gitignore`).
- Keep the token secret; prefer a `~/.pypirc` or the `TWINE_PASSWORD` env var
  over pasting it. Never put it in the repo.
