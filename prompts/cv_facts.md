# CV facts (once per CV)

Model: DeepSeek V4.1 Flash, thinking off, JSON output. The model only extracts; Python computes years and seniority from the dates (the model slips on date arithmetic with thinking off).

## System

You extract facts from a CV. Copy what the CV says; do not infer or compute anything. If something is not stated, use null. Reply with JSON only.

## User

CV:
"""
{cv_text}
"""

Return:

```json
{
  "location": "city, country as written, or null",
  "open_to_relocation": "yes | no | null (only if the CV says so)",
  "work_authorization": "as written (e.g. 'EU citizen', 'needs visa sponsorship'), or null",
  "languages": [{"language": "English", "level": "native | fluent | professional | basic | as written"}],
  "roles": [
    {
      "title": "as written",
      "company": "as written",
      "start": "YYYY-MM (use -01 if only a year is given)",
      "end": "YYYY-MM or 'present'",
      "employment": "full-time | part-time | freelance | internship | null",
      "function": "engineering | data | design | product | marketing | sales | operations | other",
      "location": "as written or null"
    }
  ],
  "education": [{"degree": "as written", "field": "as written", "end": "YYYY or null"}]
}
```

## Python, after extraction

- `total_years`: union of full-time and freelance role intervals (overlaps counted once), internships excluded, in years to one decimal.
- `years_by_function`: the same union, per role `function`. Hybrid titles take the function of the work described.
- `level`: highest level reached, from titles: intern < junior < mid < senior < lead/staff < principal/head/director. Founder, co-founder, CTO and CEO count as lead (a small-company founder title does not screen as principal). Titles without a level word count as mid after 2 years, junior before.
- `today`: the labelling date, so "present" resolves the same way for every pair.
