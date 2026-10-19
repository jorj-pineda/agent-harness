# Pinned Duodoro library snapshot

These eleven files are exact bytes from the authorized Duodoro revision
`5324cabba4ad3c8ad2e1cd1480218cf087d9b741`; `source.json` records SHA-256 values.
They provide portable, dependency-free library criterion tests without requiring
Jorge's checkout in CI. They are not a full application fixture, a held-out set,
or reference solutions. Type-only imports are erased by the selected Node 24
runtime. The CommonJS parser's actual shared companion-name dependency is included.

Tests assemble disposable Git fixtures, bind task preparation to that temporary
commit and execute Node checks with the owned external pytest bootstrap. Separate
preparation validation uses the complete pinned source, never the working tree.
No models, dependency installation or services are needed by these regressions.
