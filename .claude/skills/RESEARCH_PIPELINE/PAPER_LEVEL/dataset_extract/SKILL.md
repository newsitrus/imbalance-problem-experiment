# Skill: dataset-extract

TRIGGER: User invokes `/dataset-extract` or asks to extract dataset information from a paper.
DESCRIPTION: Takes a paper name, locates it in the topic folder structure, reads its parsed content, extracts dataset information according to the attributes defined in `prompt.txt`, and exports the results to an Excel file using the Excel MCP tool.

## Folder structure

The topic folder root is discovered by finding symlinks or directories at the project root that point to a topic-level research folder. The canonical structure is:

```
<topic_root>/
  <keyword>/                                          (e.g., "Imbalance data")
    papers/<source>/<paper_name>/                      (source = conference | journal | institutional repository)
      parsed_pdf/
        complete_content.md                            (REQUIRED — includes figure analysis)
        content.md                                     (raw parsed content, used by /analyze-paper to produce complete_content.md)
    dataset + method + evaluation metric taxonomy/
      <paper_name>/
        datasets_information.xlsx                      (OUTPUT)
```

## Instructions

1. **Accept paper name** from the user. This is the name of a paper folder (e.g., "A dual evolutionary bagging for class imbalance learning").

2. **Locate the paper's parsed content**

   Search for the paper folder across all topic roots, keyword folders, and source types:
   - Find all symlinks and directories at the project root that could be topic roots (e.g., `imbalance data problem` symlink → `/mnt/imbalance-data-problem/`).
   - Under each topic root, search `<keyword>/papers/<source>/<paper_name>/parsed_pdf/` for each keyword and source combination.
   - Source types to check: `conference`, `journal`, `institutional repository`.
   - Check for `complete_content.md`. This is the **only** accepted input — do NOT fall back to `content.md`.
   - If `complete_content.md` does not exist but `content.md` does, run the `/analyze-paper` skill (as defined in `.claude/skills/analyze-paper/SKILL.md`) on that paper first to generate `complete_content.md`, then continue.
   - If neither `complete_content.md` nor `content.md` exists, report an error and stop.
   - Record the **keyword** and **topic root** where the paper was found — these are needed for the output path.

3. **Read the extraction prompt**

   Read the file `.claude/skills/RESEARCH_PIPELINE/PAPER_LEVEL/dataset_extract/prompt.txt`.
   Parse the attribute list between `<ATTRIBUTES>` and `</ATTRIBUTES>` tags. These define the columns for the output Excel file.

4. **Read the paper content**

   Read the located `complete_content.md`. This is the full text of the paper with figure analysis included.

5. **Extract dataset information**

   Using the prompt instructions and the attribute list from `prompt.txt`, analyze the paper content and extract ALL datasets mentioned. For each dataset, populate every attribute. Follow the rules in `prompt.txt` precisely:
   - One row per dataset (or per dataset configuration/sub-problem).
   - Use "NA" for unknown values.
   - "Was used in the paper" and "Was mentioned in the paper" are mutually exclusive.
   - Fill "Was used by which paper" with the paper's full title (derived from the paper folder name or paper content).

6. **Create the output directory**

   The output goes to:
   ```
   <topic_root>/<keyword>/dataset + method + evaluation metric taxonomy/<paper_name>/
   ```
   Create this directory if it does not exist.

7. **Export to Excel**

   Use the Excel MCP tool (`mcp__excel__create_workbook`, `mcp__excel__write_data_to_excel`) to:
   - Create `datasets_information.xlsx` in the output directory.
   - Use sheet name `Dataset Information`.
   - Write headers in row 1 matching the attribute names from `prompt.txt`:
     ```
     Name | # of instances | # of features | # of classes | Class values | IR (Imbalance Ratio) | Is public | Public resource link | Application domain | Field | Was used by which paper | Was used in the paper (trained on) | Was mentioned in the paper (mentioned only) | Data preprocessing | Feature preprocessing
     ```
   - Write one row per extracted dataset starting from row 2.
   - If the file already exists, overwrite it (the user is re-extracting).

8. **Report results**

   Print a summary:
   ```
   Dataset extraction complete: <paper_name>
   Source: <keyword>/papers/<source>/<paper_name>
   Output: <output_path>/datasets_information.xlsx
   Datasets found: <N>

   | # | Dataset Name | Instances | Used | Mentioned |
   |---|--------------|-----------|------|-----------|
   | 1 | <name>       | <count>   | Yes  | No        |
   | 2 | <name>       | <count>   | No   | Yes       |
   ```

## Error handling

- If the paper is not found in any keyword/source combination, list the paths searched and ask the user to verify the paper name.
- If `complete_content.md` is missing but `content.md` exists, automatically run `/analyze-paper` on the paper to generate `complete_content.md` before proceeding.
- If neither `complete_content.md` nor `content.md` exists, report that the paper has not been parsed yet and suggest running `/parse-keyword-dir` first.
- If the Excel MCP tool is unavailable, fall back to printing the data as a markdown table.
