# docs/

Static assets for the README write-up.

## `panel.png` — agent panel screenshot

Referenced from the main [README](../README.md#local-agent-panel). The committed
capture is **offline-deterministic**: the panel runs against the test
`FakeProvider` (hence the `scripted` chip in the envelope rail) driving a real
`read_file → write_file → run_command` (pytest) turn on a temp copy of
`tests/fixtures/tiny_repo`. The UI, the workspace edit, and the pytest run are
real; only the tool-choice sequence is scripted, so the image is reproducible.

### Refresh it (live, ~30 seconds)

```bash
agent-harness serve                      # http://127.0.0.1:8000/
```

1. Open **http://127.0.0.1:8000/** in a browser.
2. (Optional) set `workspace_root` to an absolute path and pick a provider.
3. Click **New session**, send a bugfix task
   (e.g. *"Fix the failing divide test in test_calc.py"*).
4. Once the tool cards and envelope rail are populated, screenshot the window and
   save it here as `panel.png`.

A live capture needs a tool-calling provider — a cloud key (`provider: anthropic`)
is the most reliable; `gemma4` may stop early on 8 GB VRAM (see [demo.md](../demo.md)).
The committed version was produced headlessly via Playwright + the `FakeProvider`,
so no GPU or API key is required to regenerate an equivalent shot.
