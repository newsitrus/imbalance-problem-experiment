# Skill: parse-keyword-dir

TRIGGER: User invokes `/parse-keyword-dir` or asks to parse/process all papers in a keyword folder.
DESCRIPTION: Takes a keyword folder under `papers/`, audits every paper's status using `/audit-papers`, then runs `/analyze-paper` on papers that have not been parsed or are only partially parsed.

## Dependencies

- `.claude/skills/audit-papers/SKILL.md` — used in Phase 1 to audit paper folders and create/validate `_status.md` files.
- `.claude/skills/analyze-paper/SKILL.md` — used in Phase 3 to analyze figures and produce `complete_content.md`.

## Instructions

1. **Accept keyword folder path** from the user. This is a keyword-level directory under `papers/` (e.g., `papers/Imbalance data/`). Verify the folder exists.

2. **Phase 1 — Audit**

   Run the `/audit-papers` skill (as defined in `.claude/skills/audit-papers/SKILL.md`) scoped to the given keyword folder. This will:
   - Find all paper folders (subdirectories containing `parsed_pdf/`).
   - Create or validate `_status.md` for each paper folder.
   - Print the audit summary table.

   Follow every instruction in `.claude/skills/audit-papers/SKILL.md` exactly — do NOT reimplement or skip any of its steps.

3. **Phase 2 — Determine parse status for each paper**

   After the audit completes, classify each paper folder's parse status:

   | Status | Condition |
   |--------|-----------|
   | `no_content` | No `parsed_pdf/content.md` exists — skip this paper entirely |
   | `not_parsed` | `content.md` exists but `parsed_pdf/complete_content.md` does NOT exist |
   | `partially_parsed` | `complete_content.md` exists but the count of `<!-- FIGURE-DATA:` opening tags is less than the count of `![](images/` references in it |
   | `fully_parsed` | `complete_content.md` exists and every `![](images/` reference has a corresponding `<!-- FIGURE-DATA:` block |

   Print a status table after classification:

   ```
   Parse status for: <keyword folder name>

   | # | Paper | Images | Referenced | Analyzed | Status |
   |---|-------|--------|------------|----------|--------|
   | 1 | <short name> | 43 | 11 | 11 | fully_parsed |
   | 2 | <short name> | 20 | 8 | 0 | not_parsed |
   | 3 | <short name> | 15 | 5 | 3 | partially_parsed |
   ```

   - `Images`: total image files in `parsed_pdf/images/`
   - `Referenced`: images referenced in `content.md`
   - `Analyzed`: count of `<!-- FIGURE-DATA:` opening tags in `complete_content.md` (0 if file doesn't exist)

4. **Phase 3 — Analyze unparsed papers**

   For each paper with status `not_parsed` or `partially_parsed`, run the `/analyze-paper` skill (as defined in `.claude/skills/analyze-paper/SKILL.md`) on that paper. This will:
   - Create or resume `complete_content.md`.
   - Classify, analyze, and inject FIGURE-DATA blocks for each unprocessed image.

   Follow every instruction in `.claude/skills/analyze-paper/SKILL.md` exactly — do NOT reimplement or skip any of its steps. For `partially_parsed` papers, pass the resume intent (skip figures that already have `<!-- FIGURE-DATA` blocks) rather than asking the user.

   After each paper is fully analyzed, print its completion status before moving to the next.

5. **Phase 4 — Final summary**

   After all papers are processed, print:

   ```
   Keyword folder parsing complete: <keyword folder name>
   Total papers: <N>

   | # | Paper | Status | Action |
   |---|-------|--------|--------|
   | 1 | <name> | fully_parsed | skipped |
   | 2 | <name> | not_parsed -> fully_parsed | analyzed (8 figures) |
   | 3 | <name> | partially_parsed -> fully_parsed | resumed (5 remaining figures) |
   | 4 | <name> | no_content | skipped (no content.md) |
   ```
