# Before publishing this repository

This directory is suitable for a repository commit after inspection. A ZIP
file alone does not establish a published commit, release tag or archival DOI.

1. Review all scripts, result files and paths for private material or secrets.
2. Decide the software license. None has been assigned without author approval.
3. Confirm that the manuscript's research-assistance disclosure remains
   accurate, and review the released code and its substantive claims.
4. Supply the full `python -m pip freeze` from the original Mac virtualenv,
   and confirm platform/environment details; `requirements-reported.txt` is partial.
5. Validate the 2017 complete-Friday CSV against the upstream provider where
   possible, and independently compare the current 2018 S3 object if feasible.
6. Confirm whether any model outputs or datasets may legally be redistributed;
   default is to share scripts and aggregate result tables only.
7. Run smoke tests and the full dataset checksum checks on the author's Mac.
8. Create a public repository, review Git contents and commit an immutable
   release; then archive that release to a persistent repository (e.g. Zenodo)
   if appropriate and include its actual URL/DOI in the paper.
9. Link the main manuscript availability statement only after verifying that
   external readers can access the published release.

If a script is only a diagnostic and not an end-to-end pipeline, state that
explicitly. Do not claim the package reproduces raw packet-to-flow extraction.

The curated `reports/score_diagnostics/metadata.json` has two Mac-specific
absolute file paths converted to project-relative paths for privacy; retain
the original unaltered file in the private evidence archive.
