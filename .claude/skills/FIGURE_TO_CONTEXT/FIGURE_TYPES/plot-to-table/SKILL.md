# Skill: plot-to-table

TRIGGER: User invokes `/plot-to-table` or asks to extract data from a scientific plot/chart.
DESCRIPTION: Converts a scientific plot image into structured markdown tables with analysis using the MiniMax vision tool, enriched with paper context.

## Instructions

1. Ask the user for the image path or URL if not provided.
2. Run the `image-context` skill to retrieve the figure's **caption** and **mentions** from the paper's `content.md`. If no `content.md` exists (e.g., the image is a standalone URL), skip this step.
3. Call `mcp__MiniMax__understand_image` with:
   - `image_source`: the user-provided path or URL
   - `prompt`: insert the caption and mentions into the prompt as context:

```
You are given a scientific plot image with the following context from the paper:

**Caption:** <caption from image-context>
**Paper mentions:** <mentions from image-context>

Using the above context to inform your understanding (e.g., series names, method names, dataset details), analyze this image. For EACH plot: 1) Identify its title, chart type, axis labels with units, and legend. Use the paper context to resolve any ambiguous legend entries or series labels. 2) Extract all data points per series/category into a separate markdown table with columns matching the axes. If exact values are unclear, mark estimates with ~. Note any error bars or confidence intervals. 3) Provide a (~500 to ~1000 words) scientific analysis covering: key trends, comparisons across series/categories, outliers, and the main takeaway. If multiple plots exist, number them (Plot 1, Plot 2, etc.) and analyze each independently, then conclude with a ~50-word cross-plot summary highlighting relationships or overarching findings. If there is any part that you do not know, put NA, do not guess.
```

4. Present the results to the user.
