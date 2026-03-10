# Skill: analyze-figure

TRIGGER: User invokes `/analyze-figure` or asks to analyze a scientific figure that is not a plot or diagram.
DESCRIPTION: Provides a 500–1000 word scientific analysis of a figure (e.g., sample images, heatmaps, attention maps, confusion matrices, generated outputs, qualitative comparisons) using the MiniMax vision tool, enriched with paper context.

## Instructions

1. Accept the image filename (full or partial hash) or URL from the user.
2. Run the `image-context` skill to retrieve the figure's **caption** and **mentions** from the paper's `content.md`. If no `content.md` exists, skip this step.
3. Call `mcp__MiniMax__understand_image` with:
   - `image_source`: the user-provided path or URL
   - `prompt`: insert the caption and mentions into the prompt as context:

```
You are given a scientific figure with the following context from the paper:

Caption: <caption from image-context>
Paper mentions: <mentions from image-context>

Using the above context, provide a detailed scientific analysis of this figure in 500–1000 words covering:

1) Description: What the figure shows — identify all visual elements, labels, sub-figures, color coding, and layout.
2) Purpose: Why this figure exists in the paper — what research question or claim does it support.
3) Key observations: What patterns, comparisons, or results are visible. Be specific with references to sub-figures or regions.
4) Scientific interpretation: What the observations mean in the context of the paper's methodology and findings.
5) Strengths and limitations: What the figure demonstrates well and what it may not capture.
6) Takeaway: The main conclusion supported by this figure.

If there is any part that you cannot determine, state NA. Do not guess.
```

4. Present the analysis to the user.
