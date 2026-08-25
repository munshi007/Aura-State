# Publishing `aura-state` to PyPI

Two ways. **Trusted Publishing via GitHub is recommended** — no tokens, no
secrets. The manual token flow is the fallback.

---

## Recommended: publish from GitHub (Trusted Publishing, no token)

A workflow is already committed at `.github/workflows/publish.yml`. It publishes
automatically when you create a GitHub Release, authenticating via OIDC — you
never handle a token.

**One-time setup on PyPI** (before the first release):
1. NOTE: the "Connect GitHub" button under *Account → connect external accounts*
   is only for account login/recovery — it does NOT publish. Skip it.
2. Go to **https://pypi.org/manage/account/publishing/**
3. Under **"Add a new pending publisher"**, fill in exactly:
   - **PyPI Project Name:** `aura-state`
   - **Owner:** `munshi007`
   - **Repository name:** `Aura-State`
   - **Workflow name:** `publish.yml`
   - **Environment name:** *(leave blank)*
4. Click **Add**.

**Then publish** (each release):
1. Make sure the repo is public and `publish.yml` is pushed to `main`.
2. On GitHub: **Releases → Draft a new release**.
3. **Choose a tag →** type `v0.2.0` → "Create new tag on publish".
4. Title `v0.2.0`, click **Publish release**.
5. The Actions tab shows "Publish to PyPI" running; ~1 min later
   `pip install aura-state` is live. (Watch it: repo → Actions.)

For the next version: bump `version` in `pyproject.toml`, push, then draft a new
release with the new tag (e.g. `v0.2.1`).

---

## Fallback: manual upload with an API token

The package builds cleanly and installs from a wheel in a fresh environment.

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
