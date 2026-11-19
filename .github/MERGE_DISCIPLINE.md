# Merge discipline

The desired policy requires a pull request, an up-to-date branch, resolved review conversations,
and successful **Baseline gate** and **Support scenarios (offline)** checks from
the GitHub Actions app (ID 15368, verified from this repository's check runs).
Restrictions apply to administrators too; force pushes and deletion are disabled.

The human approval count is zero because Jorge is the sole maintainer. This does
not enforce Sol review: sol-loop remains an explicit pre-push workflow, with its
findings and validation recorded in the PR. Do not describe it as a GitHub check.

## Apply and verify

[branch-protection.json](branch-protection.json) is the reviewable desired policy.
Committing it alone does not enable GitHub protection. An administrator applies it
explicitly, then verifies GitHub's response:

```sh
gh api --method PUT repos/jorj-pineda/agent-harness/branches/main/protection \
  --input .github/branch-protection.json
gh api repos/jorj-pineda/agent-harness/branches/main/protection
```

This command replaces the branch's protection configuration. Inspect existing
protection and rulesets before applying it to preserve intentional future rules.
Do not use `--admin` to bypass checks. GitHub administrators can still edit the
policy; this is merge enforcement, not immutable governance.

## Scripted reports

Both CI matrices run `python -m evals.run --check`. A mismatch against explicit
boolean `expected.should_escalate` gold, missing/duplicate scenario IDs or provider
names, empty selection, missing/duplicate/unexpected matrix rows, or a non-scripted
result makes the command fail. The report is written before returning failure,
with the selected scenario filename and a PASS/FAIL section naming failures.
`--check --live` is rejected before live settings or provider access. Without
`--check`, report-only exit behavior remains available.

CI uploads separate coding and support Markdown artifacts for 14 days, including
when a matrix check fails. Argument errors or an earlier test/runtime failure can prevent report
creation; artifact upload warns about a missing file rather than inventing one.
Full offline tests, lint, formatting and configured Mypy checks still gate CI.

Overlap scores, trace flags and scripted escalation gold are not independent
patch acceptance or model-quality evidence. The gate does not require those
descriptive scores to equal one. Real coding evaluations and release criteria
remain separate. This workflow adds no deployment or target-model inference.

Policy fields follow [GitHub's branch-protection API](https://docs.github.com/en/rest/branches/branch-protection).
