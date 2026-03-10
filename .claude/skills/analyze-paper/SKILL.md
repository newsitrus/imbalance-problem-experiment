# Skill: analyze-paper

TRIGGER: User invokes `/analyze-paper` or asks to analyze all figures in a paper.
DESCRIPTION: Analyzes every figure in a paper's content.md — classifies each image, runs the appropriate analysis skill, and injects results into a complete_content.md copy.

## Instructions

1. Accept the paper name/path from the user. Derive the path to its `parsed_pdf/content.md`.

2. **Create working copy:** Copy `content.md` to `complete_content.md` in the same directory. If `complete_content.md` already exists, ask the user whether to overwrite or resume (skip figures that already have `<!-- FIGURE-DATA` blocks).

3. **Find all images:** Grep for all `![](images/` references in `complete_content.md` to get the list of image hashes with their line numbers.

4. **For each image (process sequentially in the main agent, no subagents):**

   a. **Locate the image file** using Glob with the hash pattern.

   b. **Classify the figure type** — Call `mcp__MiniMax__understand_image` directly (NOT via subagent) with a prompt to classify as "plot", "diagram", or "general".

   c. **Get image context** — Run image-context logic:
      - Grep for the image hash in content.md to get the caption
      - Extract figure number from caption
      - Grep for mentions of the figure in the paper

   d. **Run the appropriate MiniMax analysis** based on classified type:
      - `plot` → Use the plot-to-table prompt (data extraction + analysis)
      - `diagram` → Use the diagram-to-graph prompt (JSON graph + analysis)
      - `general` → Use the analyze-figure prompt (descriptive analysis)

   e. **Inject the result** into `complete_content.md`:
      - Find the line with the image hash
      - Find the caption line after it
      - Use Edit to insert the analysis block after the caption using the format:
        ```
        <!-- FIGURE-DATA: Fig. X | type: [plot/diagram/general] -->
        > **[Extracted Data/JSON/Analysis]**
        >
        > <content>
        <!-- /FIGURE-DATA -->
        ```
      - If block already exists, replace it

5. **After all figures are processed**, report a summary:
   ```
   Processed X figures:
   - Y plots
   - Z diagrams
   - W general figures
   Output: <path to complete_content.md>
   ```
