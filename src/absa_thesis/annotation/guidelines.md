# Annotation Guidelines (Aspect + Polarity)

Unit: sentence-level.

Labels:
- `aspect_category`: choose one from the schema list (battery, screen, performance, updates, price, design, usability, support, privacy, ads). If none apply, mark as `other` and leave a note.
- `polarity`: {positive, negative, neutral}

Rules of thumb:
- If multiple aspects are present, split into separate sentences where possible. Otherwise choose the dominant aspect (by sentiment weight or focus).
- Ignore star ratings. Label based on the text.

Edge cases:
- Mixed sentiment in one sentence → label as `neutral` unless one polarity is clearly dominant.
- Vague praise/complaint with no specific aspect → `other`.
