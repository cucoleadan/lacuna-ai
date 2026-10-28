// jobKey: the URLs of one job map to one key.   node test/jobkey.mjs
import assert from "node:assert/strict";
import { jobKey } from "../src/core.js";

const same = {
  "https://jobs.ashbyhq.com/sisense/7c68ab81-771c-479b-aace-1426d208309f": [
    "https://jobs.ashbyhq.com/sisense/7c68ab81-771c-479b-aace-1426d208309f/application",
    "https://jobs.ashbyhq.com/sisense/7c68ab81-771c-479b-aace-1426d208309f/application?utm_source=x#form",
  ],
  "https://jobs.lever.co/acme/1234-abcd": ["https://jobs.lever.co/acme/1234-abcd/apply", "https://jobs.lever.co/acme/1234-abcd/?lever-source=LinkedIn"],
  "https://acme.wd1.myworkdayjobs.com/en-US/careers/job/Lisbon/Engineer_JR123": [
    "https://acme.wd1.myworkdayjobs.com/en-US/careers/job/Lisbon/Engineer_JR123/apply/applyManually",
  ],
  "https://apply.workable.com/acme/j/ABC123": ["https://apply.workable.com/acme/j/ABC123/apply/"],
  "https://www.linkedin.com/jobs/view/3912345678/": [
    "https://www.linkedin.com/jobs/search/?currentJobId=3912345678&keywords=ai",
    "https://www.linkedin.com/jobs/collections/recommended/?currentJobId=3912345678",
    "https://www.linkedin.com/jobs/view/senior-ai-engineer-at-acme-3912345678?trk=x",
  ],
};
for (const [key, urls] of Object.entries(same)) {
  assert.equal(jobKey(key), key);
  for (const u of urls) assert.equal(jobKey(u), key, u);
}
// different jobs stay different; ids in the query are kept
assert.notEqual(jobKey("https://jobs.ashbyhq.com/sisense/aaa"), jobKey("https://jobs.ashbyhq.com/sisense/bbb"));
assert.equal(jobKey("https://acme.com/careers?gh_jid=42&utm_medium=x"), "https://acme.com/careers?gh_jid=42");
console.log("jobKey ok");
