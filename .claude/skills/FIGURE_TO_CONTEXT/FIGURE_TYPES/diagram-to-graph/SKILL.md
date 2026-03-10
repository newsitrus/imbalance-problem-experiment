# Skill: diagram-to-graph

TRIGGER: User invokes `/diagram-to-graph` or asks to convert a diagram/flowchart into a JSON graph.
DESCRIPTION: Converts a scientific diagram image into a structured JSON graph using the MiniMax vision tool, enriched with paper context.

## Instructions

1. Accept the image filename (full or partial hash) or URL from the user.
2. Run the `image-context` skill to retrieve the figure's **caption** and **mentions** from the paper's `content.md`. If no `content.md` exists, skip this step.
3. Call `mcp__MiniMax__understand_image` with:
   - `image_source`: the user-provided path or URL
   - `prompt`: insert the caption and mentions into the prompt as context:

```
You are given a scientific diagram with the following context from the paper:

**Caption:** <caption from image-context>
**Paper mentions:** <mentions from image-context>

Using the above context to resolve ambiguous labels or component names, do the following:

1) Convert this diagram into a JSON graph. For each element, create a node with: id (snake_case), label (exact text from diagram), type (one of: process, data, decision, input, output, component, group). For each arrow/connection, create an edge with: source, target, and optional label. If elements are grouped/nested, add a "parent" field referencing the group node id. Capture all text, annotations, and directional flows accurately. If there is any part that you do not know, put NA, do not guess.

2) After the JSON, provide a ~500 to ~1000 words scientific analysis covering: the overall architecture and purpose of the diagram, how components interact and data flows through the system, the role of each major group/stage, design choices and their scientific rationale based on the paper context, and the key takeaway of the approach.

Output the JSON first, then the analysis.
```

4. Present the JSON and analysis to the user.
