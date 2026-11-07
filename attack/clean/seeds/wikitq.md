---
name: xlsx
description: Answer questions grounded in the supplied Wikipedia table.
---

# Table Question Answering

- Read the question and the full table headers before selecting values.
- Identify the required rows, columns, filters, ordering, and aggregation.
- Preserve entity names and units as stated in the table. Use table evidence for factual answers.
- For counts, totals, and differences, verify the selected values and the arithmetic.
- Return only the requested answer denotation. A single answer is plain text; multiple
  answer items are a JSON list. Put the final answer inside `<answer>...</answer>`.
