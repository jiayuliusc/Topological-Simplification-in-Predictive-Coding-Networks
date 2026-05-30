# Public Release Checklist

Use this checklist before publishing the repository alongside the arXiv paper.

## Required Before Public Release

- Add the final paper title, author list, and arXiv citation.
- Update `CITATION.cff` with the final author list, DOI/arXiv identifier,
  release date, and preferred citation message.
- Add a license chosen by the authors or institution.
- Add links and checksums for any public datasets or generated artifacts the
  authors choose to share.
- Verify the documented dependency extras are clear enough for users to adapt
  with their own environment manager, cluster modules, Codex, Cursor, or similar
  tools.
- Verify the core COM comparison and figure scripts expose clear `--help`
  output and use documented input/output paths.
- Confirm no private filesystem paths, usernames, or cluster-only assumptions
  appear in tracked source files.
- Confirm no large binary artifacts are tracked by git.

## Recommended

- Add a small smoke-test dataset or fixture for CI.
- Add unit tests for `compute_com_from_diagrams`, diagram normalization, and
  unequal-n bootstrap behavior.
- Add a `CITATION.cff` after the arXiv identifier is available.
- Add an environment lock file only if the authors want strict one-machine
  reproduction. Otherwise, keep dependency guidance at the interface level.
- Add a short methods note mapping scripts to paper figures.
