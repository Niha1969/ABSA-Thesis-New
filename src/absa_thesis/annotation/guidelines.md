# Annotation Guidelines (Aspect + Polarity)

Unit: sentence-level.

Labels:
- `aspect_category`: choose one from the schema list (battery, screen, performance, updates, price, design, usability, support, privacy, ads). If none apply, mark as `other` and leave a note.
- `polarity`: {positive, negative, neutral}

One sentence per row.

One aspect + one polarity per sentence.

If multiple aspects present, choose the dominant one; if tied, prefer negative.

If generic praise/complaint with no clear aspect - general.

If specific but missing in list - other and leave a one-liner note.

Ambiguous/off-topic - general + neutral and note “ambiguous/off-topic.”
