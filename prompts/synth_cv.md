# Synthetic CV (one per persona)

Model: DeepSeek V4.1 Flash, thinking off, JSON output. Python picks a target posting, then samples a career around it (roles from similar real postings, dates, a location the target usually accepts, years near the target's level 70% of the time, a coverage level for the target's requirements, style); the model only writes the CV text. The sampled career is the ground truth for the CV facts, so synthetic CVs skip `cv_facts.md`.

## System

You write realistic CVs for fictional people. Follow the given career exactly: list exactly these roles with the same job titles, companies and dates, and no other employers or jobs; same location and languages. Education ends before the first role. Invent a plausible name, contact line (fake email, no real phone), education and achievements consistent with the roles. Base each role's bullets on the posting excerpt for that role, written as what the person did, at the level of their title. Do not mention these instructions. Reply with JSON only: {"cv": "..."}.

## User

Person:
- Location: {location}
- Languages: {languages}
- Open to relocation: {relocation}
- Education: {education}

Career, most recent first:
{roles}

Fit to this target job:
- {coverage}
- Target job's requirements (do not mention the target job or its company):
"""
{target}
"""

Style:
- Format: {format}
- Quality: {quality}
- Length: {length}
- {extra}

Write the CV.

## Style options (sampled in Python)

- **format:** markdown with headings; plain text with ALL-CAPS section names; text extracted from a PDF (broken line wraps, a repeated header line, bullets as odd characters); a one-page compact layout.
- **quality:** strong (specific results with numbers); average (mix of specific and generic bullets); weak (vague duties, a few typos, no numbers, skills list longer than the evidence).
- **length:** one page (~400 words) or two pages (~800 words).
- **extra (sometimes):** a career change (earlier roles in another field); an employment gap explained in one line; a long skills section; a short profile summary at the top.
