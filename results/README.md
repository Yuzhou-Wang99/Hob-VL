# Saved evaluation results

This archive contains 36 completed runs: eight model identities, nine inference
configurations, and four task/format conditions. There are 126,000 final answers
and 92 earlier API-error attempts, all superseded by completed retries. Invalid
model outputs remain in the archive and count as incorrect under the saved
scoring policies. Pilot and intentionally interrupted evaluations are excluded.

- `runs/`: full saved predictions, manifests, and summaries.
- `archive_manifest.json`: coverage, attempt counts, and release-file checksums.
- `configurations.json`: requested API settings and recorded initialization dates.
- `generated/`: CSV tables, invalid-output details, and a JSON summary.
- `property_analysis/`: descriptive logical-property analyses and the heatmap.

Start with [the accuracy overview](generated/README.md),
[the property report](property_analysis/README.md), and
[the reproduction instructions](../REPRODUCING.md).

Only IDs, paths, account-specific endpoint information, and corresponding file
hashes/fingerprints were adapted for the release. No final response, reasoning
text, target, score, or usage was altered. The `archive` block in each manifest
records provenance. Archived runs are protected from inference resumption;
write new evaluations to a separate directory.
