# Skill: inject-figure-data

TRIGGER: User invokes `/inject-figure-data` or asks to inject figure analysis back into a paper.
DESCRIPTION: Injects analysis from any figure type (plot, diagram, or general) into content.md or complete_content.md below the figure caption.

## Instructions

1. Accept the image hash, figure type (plot/diagram/general), analysis content, and target file path.
2. Locate the image reference in the target file using Grep on the image hash.
3. Extract the figure number from the caption line (e.g., `Fig. X`).
4. Inject the result directly after the figure caption line using this format:

For **plot** type:
```
<!-- FIGURE-DATA: Fig. X | type: plot -->
> **[Extracted Data]**
>
> <tables from analysis>
>
> **[Analysis]**
>
> <scientific analysis>
<!-- /FIGURE-DATA -->
```

For **diagram** type:
```
<!-- FIGURE-DATA: Fig. X | type: diagram -->
> **[JSON Graph]**
>
> <json graph from analysis>
>
> **[Analysis]**
>
> <scientific analysis>
<!-- /FIGURE-DATA -->
```

For **general** type:
```
<!-- FIGURE-DATA: Fig. X | type: general -->
> **[Analysis]**
>
> <scientific analysis>
<!-- /FIGURE-DATA -->
```

5. If a `<!-- FIGURE-DATA: Fig. X` block already exists for this figure, replace it with the new result.
6. Use Edit to insert the block — do not rewrite the entire file.
