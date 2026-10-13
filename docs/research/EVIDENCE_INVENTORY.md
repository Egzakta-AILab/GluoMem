# Research evidence inventory

Recorded 2026-10-13 against commit `ab3ecdd56ea87f1916f3fd515b3a35d8911d2233`
(GitHub `main`, version 0.0.1). This inventory lists what research material exists,
what was archived, and what is missing. Nothing was deleted. No secrets or private
content are listed.

## Method

- Tracked files: `git ls-files` at `ab3ecdd` (164 files).
- Ignored and untracked files in the checkout: `git status --short --ignored` was
  empty, so there were no local-only artifacts to preserve.
- Missing run directories: every `experiments/<timestamp>_*` path cited by the
  results documents, searched in the checkout and with Spotlight (`mdfind`) on the
  maintainer machine. None was found.
- Hashes are SHA-256 of the file content at `ab3ecdd`. Recover any original with
  `git show ab3ecdd:<original path>`.

## Archived research documents

These moved to `docs/research/`. Each received an evidence-status banner at the top;
the text below the banner is unchanged, so the original remains recoverable from Git.

| Original path | Archived path | Bytes at `ab3ecdd` | SHA-256 at `ab3ecdd` |
|---|---|---|---|
| `PAPER.md` | [PAPER.md](./PAPER.md) | 12915 | `bc55210887967016a9a0681e3e7fd8cc09647dc7e7b2d23fdfce6dfcac0fea64` |
| `RESULTS.md` | [RESULTS.md](./RESULTS.md) | 18805 | `d7f166f02f0384d644905607cbe220a3541f0964cf0cebced36d6d691fd8cf76` |
| `LOCOMO_RESULTS.md` | [LOCOMO_RESULTS.md](./LOCOMO_RESULTS.md) | 7096 | `a310a7aa0a5a78e73aec894988920b8b9208f1927024f236f0a214d13c8d6db5` |
| `LONGMEMEVAL_RESULTS.md` | [LONGMEMEVAL_RESULTS.md](./LONGMEMEVAL_RESULTS.md) | 14992 | `b92b21b693755fb3a8be30878c56e060c7cb3307063aa4e7b9d2c21fe01a4ba3` |

## Kept in place

| Path | Bytes at `ab3ecdd` | SHA-256 at `ab3ecdd` | Change in this revision |
|---|---|---|---|
| `SPEC.md` | 13466 | `828d995d665c611238ecafe3606f53272230a13030623ecbcb9b8df178c40419` | Status notes added |
| `DESIGN.md` | 18409 | `fc834961fb28d9d5f84fa99ab62126de7b4f17ba8cdec34c499323f4da2402cb` | Status notes added |
| `README.md` | 4637 | `79cb7f3e8bf980888d96dbc06d2c404f58e87c39d548bb8953761587a8370823` | Rewritten to match the implementation |
| `conf/config.yaml` | 11550 | `8e7f80b5babc6a25bea707137610ea07e1f0c13d839f7068322f64d566ed6f44` | None |
| `benchmarks_data/locomo_sample_seed42.json` | 7021 | `b96e6642e151062eb27c2def3d4cab8f7e105f208737996fcec90578867ab08e` | None |
| `benchmarks_data/longmemeval_oracle_all500.json` | 8990 | `3f88211815b9599a5e325c7989ce3123cbcf712f0c0836a89382f17e670b2d6c` | None |
| `benchmarks_data/longmemeval_oracle_pref30.json` | 773 | `2f97b4bff22db0665541d76e751503e71a25f0defc8a9aadbf15437081ef0533` | None |
| `benchmarks_data/longmemeval_s_pref15.json` | 560 | `2370488b1a8aeb330b7c8b9a6101acb2df05a1f2aaaa53eae5d1c08357ebd43d` | None |
| `benchmarks_data/longmemeval_s_sample_seed42.json` | 2169 | `6c26d3fa2c761ad9a48756c0cd41204c68452f2f3455123953c650c827c8bb12` | None |
| `tests/fixtures/locomo_tiny.json` | 2251 | `e593ff44c623ce1e8f26b5a8a9afe95d6b8d92ba9a4357a4f7defacdf893c6fc` | None |
| `tests/fixtures/locomo_tiny_2conv.json` | 1901 | `d4515dc6688c777b6ca388a200478e04ab9ed6366ffca8444e8ac60ffce64c78` | None |
| `tests/fixtures/longmemeval_tiny.json` | 2061 | `e0241fa1e6fe3055fe9c63bcaf73f1559c7a6ac8cc08bb290ed32be22ccac1c3` | None |

The `benchmarks_data/*.json` files are sampling manifests: dataset name, dataset
SHA-256, seed and question identifiers. They contain no benchmark conversation or
answer text. The `tests/fixtures/*.json` files are authored synthetic conversations
shaped like the public formats.

## Missing evidence

The results documents cite these run directories as committed. None is in the
repository history or on the maintainer machine. Their predictions, judgments and
manifests cannot be rescored, so the reported numbers are historical claims only.

| Cited by | Run directories |
|---|---|
| `LOCOMO_RESULTS.md`, `RESULTS.md` | `20260720T104120_locomo_gluomem`, `20260720T110152_locomo_gluomem`, `20260720T102815_locomo_gluomem`, `20260720T142910_locomo_gluomem`, `20260721T141438_locomo_gluomem`, `20260706T154347_locomo_full_context`, `20260720T105659_locomo_vector_rag`, `20260720T104634_locomo_mem0`, `20260720T105150_locomo_bm25`, `20260720T091322_locomo_gluomem`, `20260720T091744_locomo_gluomem`, `20260720T092140_locomo_gluomem`, `20260706T164602_locomo_gluomem` |
| `LONGMEMEVAL_RESULTS.md` | `20260721T224430_longmemeval_oracle_gluomem`, `20260722T072611_longmemeval_oracle_full_context`, `20260722T074437_longmemeval_oracle_bm25`, `20260722T075757_longmemeval_oracle_vector_rag`, `20260722T085051_longmemeval_oracle_mem0`, `20260725T140047_longmemeval_s_gluomem`, `20260724T183931_longmemeval_s_full_context`, `20260724T184450_longmemeval_s_bm25`, `20260724T185606_longmemeval_s_vector_rag`, `20260724T185803_longmemeval_s_mem0`, `20260724T183654_longmemeval_s_gluomem`, `20260727T081854_longmemeval_s_gluomem`, `20260727T081924_longmemeval_s_gluomem`, `20260727T083719_longmemeval_oracle_gluomem`, `20260727T083832_longmemeval_oracle_gluomem`, `20260727T142452_longmemeval_oracle_gluomem`, `20260727T143314_longmemeval_oracle_gluomem`, `20260728T005258_locomo_gluomem`, `20260728T010111_locomo_gluomem`, `20260728T010558_longmemeval_oracle_gluomem`, `20260728T011914_longmemeval_oracle_gluomem`, `20260728T013324_longmemeval_oracle_gluomem`, `20260728T014700_longmemeval_oracle_gluomem`, `20260729T033957_longmemeval_s_gluomem`, `20260729T034202_longmemeval_s_gluomem`, `20260729T075512_longmemeval_oracle_gluomem`, `20260729T080735_longmemeval_s_gluomem` |
| `PAPER.md` | `experiments/ablations.yaml` (planned, never present) |

Restoring these requires the original author's copies. Until then, a new baseline
must be produced from published, rescorable run directories before any external
quality claim.

## Outside the repository

Two DSPy response caches exist in the maintainer's home directory. They were not
moved, changed or committed; they are listed so their existence is known. They can
hold model completions for benchmark inputs, so they must not be published without
review.

| Path | Files | Size | SHA-256 of sorted per-file hashes |
|---|---|---|---|
| `~/.gluomem/dspy_cache` | 16 | 512 KiB | `4edcb27a7733b9f8d7f89d8d95653e4ab593e886638ad3fb3a3b27892cd57d64` |
| `~/.mneme/dspy_cache` (pre-rename project name) | 16 | 552 KiB | `b4272c108be2edf2907d9756b8559e4b480e28ab86f036a3352cce9ad21648ad` |

## Removed

Nothing.
