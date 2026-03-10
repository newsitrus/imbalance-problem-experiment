# Skill: audit-papers

TRIGGER: User invokes `/audit-papers` or asks to audit paper folders / check paper status.
DESCRIPTION: Audits all paper folders under "imbalance data problem", creates or validates `_status.md` tracking files for image analysis state.

## Instructions

1. **Find all paper folders** by searching for directories that contain a `parsed_pdf/` subdirectory anywhere under `imbalance data problem/`. Each such directory is a "paper folder". If an argument is provided, only audit the matching paper folder.

2. **For each paper folder**, perform the audit:

### A. Gather facts (do NOT guess — only report what exists on disk)

   - List all `.jpg`/`.png` files in `parsed_pdf/images/`.
   - Grep `parsed_pdf/content.md` for `![](images/<hash>` to get the list of **referenced** images.

### B. If `_status.md` does NOT exist — create it

   Use this exact template:

   ```markdown
   # <Paper Folder Name>

   ## Status
   - total_images: <count of files in parsed_pdf/images/>
   - referenced_in_content: <count of images referenced in content.md>

   ## Images

   | # | Hash | In content.md |
   |---|------|---------------|
   | 1 | abcd1234... | yes |
   | 2 | ef567890... | no |
   ```

   Rules for populating the table:
   - List **every** image file from `parsed_pdf/images/`, sorted alphabetically by hash.
   - `Hash`: first 12 characters of the filename (without extension), enough to identify.
   - `In content.md`: `yes` if the hash appears in a `![](images/` reference in `content.md`, otherwise `no`.

### C. If `_status.md` already exists — validate and fix

   Read the existing `_status.md`, then check:

   1. **Missing images**: images on disk not listed in the table. Add them.
   2. **Stale images**: images listed in the table but no longer on disk. Remove them.
   3. **Wrong states**: verify `In content.md` column matches actual `content.md` references. Fix any mismatches.
   4. **Update counts** in the Status section to match the corrected table.

   If changes were made, report what was fixed. If everything is correct, report "status OK".

3. **Summary**: After processing all paper folders, print a summary table:

   ```
   Audit complete. <N> paper folders checked.

   | Paper | Images | Referenced | Status File |
   |-------|--------|------------|-------------|
   | <name> | 44 | 11 | created |
   | <name> | 43 | 8 | OK |
   ```

   `Status File` column: `created` if new, `fixed` if corrected, `OK` if valid.
