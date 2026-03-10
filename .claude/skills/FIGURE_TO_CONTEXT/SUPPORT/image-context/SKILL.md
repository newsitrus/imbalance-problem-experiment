# Skill: image-context

TRIGGER: User invokes `/image-context` or asks for caption/context of an image from a parsed paper.
DESCRIPTION: Retrieves the caption and all mentions/discussions of a specific image throughout a content.md file.

## Instructions

1. Accept the image filename (full or partial hash) from the user.
2. Use Glob to locate the image file: `**/*<hash>*`.
3. From the matched path, derive the sibling `content.md` at `<parent>/parsed_pdf/content.md`.
4. Use Grep to find the image hash in `content.md` with `-A 2` to extract the caption line (e.g., `Fig. X.`).
5. Extract the figure number (e.g., `Fig. 10`) from the caption.
6. Use Grep to search the entire `content.md` for all mentions of that figure reference (e.g., `Fig. 10`, `Figure 10`, `fig. 10`) to get line numbers.
7. For each mention, use Read with offset/limit to capture the **full paragraph** containing the reference — expand until you hit a blank line or heading on both sides.
8. Output format:

```
**Caption:** <full figure caption>

**Mentions:**
- <each paragraph or sentence that references this figure, with its location in the paper>
```
