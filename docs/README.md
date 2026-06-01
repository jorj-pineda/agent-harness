# docs/

Static assets for the README write-up.

## `panel.png` — agent panel screenshot

Referenced from the main [README](../README.md#local-agent-panel). Capture it once a
turn has populated the tool rail (~30 seconds):

```bash
agent-harness serve                      # http://127.0.0.1:8000/
```

1. Open **http://127.0.0.1:8000/** in a browser.
2. (Optional) set `workspace_root` to an absolute path and pick a provider.
3. Click **New session**, send a bugfix task
   (e.g. *"Fix the failing divide test in test_calc.py"*).
4. Once the tool cards and envelope rail are populated, screenshot the window and
   save it here as `panel.png`.

A populated capture needs a tool-calling provider — a cloud key (`provider: anthropic`)
is the most reliable; `gemma4` may stop early on 8 GB VRAM (see [demo.md](../demo.md)).
