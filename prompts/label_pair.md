# Pair label (one CV, one job posting)

Model: DeepSeek V4.1 Flash, thinking off, JSON output. The CV block comes first so a CV's pairs share a cached prefix.

The question being answered: **would this CV likely get an interview for this job?** Judge as a realistic recruiter doing a first screen, from the candidate's side: what matters is whether applying is worth their time.

## System

You screen job applications. Given a candidate's CV, facts computed from it, and a job posting, extract what the posting requires and judge whether the CV meets each requirement. Judge from evidence in the CV: a skill is met if a role, project or degree shows it was used; listed only in a skills section counts as partial; adjacent experience counts as partial (e.g. Vue for a React requirement). Do not reward keyword overlap. Do not penalise missing nice-to-haves. Read the whole posting: location and residency limits are often near the end. Reply with JSON only.

## User

CANDIDATE FACTS (computed, trust these over your own reading of dates):
- Total professional experience: {total_years} years
- Roles, most recent first:
{roles_list}
- Location: {location}; open to relocation: {open_to_relocation}; work authorization: {work_authorization}
- Languages: {languages}
- Today: {today}

CV:
"""
{cv_text}
"""

JOB POSTING:
"""
{job_text}
"""

Return:

```json
{
  "requirements": [
    {"text": "short requirement from the posting", "kind": "must | nice", "gate": false, "met": "yes | partial | no"}
  ],
  "location_eligible": "yes | no | unclear",
  "required_years": "minimum years of experience the posting asks for, as a number, or null",
  "role_level": "intern | junior | mid | senior | lead | principal",
  "role_function": "engineering | data | design | product | marketing | sales | operations | other",
  "relevant_roles": [1, 2],
  "main_gap": "one sentence: the most important requirement the CV does not show, or 'none'"
}
```

Rules:

1. **requirements:** at most 10, in the posting's order. `must` = stated as required, minimum, or must; `nice` = preferred, plus, bonus, or nice to have. If the posting does not say, treat years of experience, core stack and required languages as `must`, everything else as `nice`. Leave out years of experience (it goes in `required_years`) and location (it goes in `location_eligible`).
2. **gate:** `true` only for hard gates: a required spoken language at working level ("German is a plus" is not a gate); a legally required licence or degree (medicine, law, chartered accounting), not "BSc in CS or equivalent"; security clearance.
3. **met:** `yes` if a role, project or degree in the CV shows it was used; `partial` if it is only listed in a skills section, or the CV shows adjacent experience (Vue for React); `no` otherwise.
4. **location_eligible:**
   - `no` if any of these hold:
     - on-site or hybrid in **another country** than the candidate's, and the CV does not say they will relocate;
     - remote but limited to countries, regions or a legal entity that do not include the candidate's country ("Remote (US)", "UK-based", "we can only hire in Germany, Spain");
     - it requires a work authorization the candidate's country does not give and the CV does not state (an EU citizen can work anywhere in the EU, not in the UK or US without a visa).
   - `yes` if remote with no restriction, or the restriction includes the candidate. A region ("EMEA", "Europe", "EU", "CET ±2h") includes every country in it. A time-zone overlap requirement alone does not make it `no`.
   - `unclear` if the posting or the CV does not say enough, or the job is on-site or hybrid in another city of the candidate's own country (people relocate within their country; no visa needed).
5. **role_function:** the kind of work the role is.
   **relevant_roles:** the numbers of the candidate's roles whose work counts toward this job's experience requirement, as a recruiter would count it (an ML engineer role counts for a data scientist job; a sales role does not count for a software job). Empty list if none.
6. **role_level:** from the title and the posting (intern/graduate < junior < mid < senior < lead/staff < principal/head/director). No level word in the title: `mid`.

## Python, after labelling (the model never applies thresholds)

- Candidate years and level are **relevant** ones: the union of the dates of `relevant_roles` (overlaps once, internships excluded), and the highest level among those roles' titles (none: junior). Recruiters count "7+ years of software engineering" against relevant years, not a whole career; the model judges relevance, Python does the date arithmetic.
- `candidate_vs_role`: gap = `required_years` minus relevant years when the posting states years, otherwise level steps from candidate level to `role_level` (one level ≈ 2 years).
  - `below` if the gap is 2+ years or the role is 1+ level above; `above` if the candidate is 2+ levels above the role (and, when the posting states years, has at least 5 more than asked); `at` otherwise. A candidate with no full-time experience is `at` for intern roles.
- Must-have score: `yes` = 1, `partial` = 0.5, over the `must` items (no `must` items: score 1).
- `must_haves_met`: every gate is `yes` and at most one other `must` is `no`.
- `verdict`:
  - `skip` if `location_eligible` is `no`; or a gate is not `yes`; or 3+ years / 2+ levels below; or the must-have score is under 0.5.
  - `apply` if location is `yes` or `unclear`, the must-have score is at least 2/3, and `candidate_vs_role` is `at` (or `below` by exactly 2 years with a score of 1).
  - `stretch` otherwise.
- Domestic override: if `location_eligible` is `no` but the posting's own Location field names the candidate's country, it becomes `unclear` (the labeller often ignores the domestic-relocation rule; for real CVs this applies only when the CV states the country).
- Candidate level comes from CV titles (`prompts/cv_facts.md`); founder, co-founder, CTO and CEO titles count as `lead`, since a small-company founder title does not screen as principal.

## Why these thresholds (decided 2026-10-22)

- **Two thirds of must-haves for `apply`:** first screens routinely pass candidates who miss a few listed requirements; "requirements" lists are wish lists. Below half, interviews are rare.
- **Seniority: 2 years short is `stretch`, 3+ years or 2 levels is `skip`:** recruiters regularly flex 1-2 years, rarely more.
- **Skills-section-only is partial, not met:** a recruiter sees it, but without a role or project behind it, it is weaker evidence.
- **`unclear` location does not block `apply`:** most postings that say nothing about location are open; the user verifies. Explicit limits, often at the end of a posting, are read and enforced.
- **Over-qualified is `stretch`, not `skip`:** it happens (career changes, market downturns), but screeners hesitate.
