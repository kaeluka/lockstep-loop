# One task, many kinds of subject

[Documentation](README.md) · [Visual guide](concepts.md)

lockstep-loop is useful when many subjects can be investigated through the **same
prompt and tool interface**. Subjects may arrive together or one at a time: a
recorded agent can reuse an investigation from an earlier run.

An environment is the caller's binding between a subject, a stable evidence
snapshot, and its tool implementations. The core only sees opaque environment
IDs for routing. A tool such as `read_section("methods")` operates on the current
environment's evidence; the model need not name the study, database, or storage
bucket that holds it.

The following are application designs, not built-in integrations. Tools, document
extraction, model adapters, evaluation criteria, and persistence are caller-owned.

## 1. Research-paper screening

**Question:** “Does this study satisfy our review's inclusion criteria? Identify
missing evidence before deciding.”

- **Environment:** an archived study bundle: paper, supplements, and corrections.
- **Tools:** `study_metadata()`, `read_section(section)`, `read_supplement(name)`.
- **Shared work:** studies with identical metadata observations can share early
  reasoning about eligibility. The agent requests detailed methods only where
  needed; full-text differences naturally cause splits.
- **Specialist:** `assess_design(criterion="randomized")` can investigate different
  methods sections and return the same structured finding. Equal findings need
  not split the parent, even though the specialist's histories remain distinct.
- **Later:** a corrected supplement is a new environment snapshot. Replay checks
  the evidence again rather than assuming the old eligibility judgment still holds.

Archive the evidence first. A live search engine's changing result order is not
a stable tool observation. Automated screening should expose uncertainty and
remain subject to the review's validation process.

## 2. Product requirement checks

**Question:** “Does this device meet our outdoor operating requirements?”

- **Environment:** one product's versioned datasheet and certification evidence.
- **Tools:** `read_ratings()`, `read_test_report(test)`, `read_limitations()`.
- **Shared work:** products with the same relevant ratings share an assessment;
  a temperature limit or enclosure rating difference opens another path.
- **Feedback:** “You checked the enclosure rating but not the operating-temperature
  limits. Check both before concluding.” The same answer may be returned after
  the additional investigation, now backed by better evidence.
- **Later:** a revised datasheet can reuse the prior reasoning if the observations
  still match. A changed specification triggers reasoning at the changed checkpoint.

The [Python](../python/examples/product_screening.py) and
[Rust](../rust/examples/product_screening.rs) demos implement a deliberately small
version of this scenario. Their model replies are scripted to make call counts
and branching reproducible; they are not real product-certification advice.

## 3. Supplier evidence review

**Question:** “Is the submitted evidence sufficient for our assessment checklist?”

- **Environment:** a frozen supplier packet containing questionnaires, certificates,
  and audit documents, together with any fixed assessment inputs such as an as-of date.
- **Tools:** `read_checklist_answers()`, `list_evidence()`, `read_document(kind)`.
- **Shared work:** identical checklist observations can share a path. A missing
  document or different exception opens another. Reading every full document at
  the start would usually fragment the conversation immediately.
- **Specialist:** `assess_evidence(requirement="retention")` can examine different
  wording and return a stable finding, while retaining the complete evidence in
  its child agent.
- **Feedback:** “Check the exclusions as well as the certificate.” Do not assert
  “this supplier has an expired certificate” as transferable feedback: other
  environments following the same path inherit that continuation.

Time-dependent checks must use a pinned assessment date, not the wall clock.
Express changed checklist requirements in the prompt or observed evidence. Pure
tools rerun, so changed outputs branch without a manual version bump. Review
findings support a decision; they are not a compliance guarantee.

## 4. Contract and policy comparison

**Question:** “Flag renewal terms that differ from our review policy.”

- **Environment:** one signed contract, its incorporated schedules, and amendments.
- **Tools:** `read_clause(topic)`, `list_amendments()`, `read_amendment(section)`.
- **Shared work:** identical template clauses support common reasoning. Negotiated
  language causes a split; it should not be erased merely to improve cache hits.
- **Specialist:** `extract_obligation(topic="notice_period")` can return a structured
  finding while keeping citations and raw text available in the child record.
- **Feedback:** “You did not inspect the amendment that can override the renewal
  clause.” This concerns the investigation, not its previous answer's string value.

A new document bundle is a new environment. A changed review policy is instead a
prompt/evidence change or caller-owned policy selection, not a manual cache key.
Human review remains important for interpretations and consequential decisions.

## 5. Dataset intake and recurring reports

**Question:** “Is this dataset release suitable for our reporting pipeline, and
what caveats must accompany it?”

- **Environment:** an immutable dataset release plus its schema and data dictionary.
- **Tools:** `read_schema()`, `column_statistics(column)`, `read_dictionary(field)`.
- **Shared work:** releases with matching schemas and statistics can share an
  investigation, even if their underlying rows are different. Equality is about
  what the agent observed—not a claim that the datasets themselves are equal.
- **Later:** next month's release replays the recorded checks. If units change
  from dollars to cents, the changed dictionary observation opens a new path.
- **Feedback:** “Also check unit definitions; a stable schema does not imply
  stable meanings.” Add a durable investigation step rather than silently
  changing a previously recorded observation.

The tools must query pinned data and use deterministic aggregates/sampling.
Checks need to observe the properties that matter: a cache cannot detect a defect
in a column the investigation never queried.

## 6. Source-code analysis

**Question:** “Identify unsafe path handling in this package.”

- **Environment:** a repository at a resolved commit, with any relevant configuration.
- **Tools:** `list_files(directory)`, `read_file(path)`, `find_references(symbol)`.
- **Shared work:** unchanged files and configuration preserve common observations
  across forks or revisions. Changes to observed code cause branches.
- **Specialist:** `list_imports(file_path="foo.py")` builds a deterministic child
  prompt, so parent prompt edits need not discard the child's investigation.
- **Coverage:** caller code can enumerate files and require one investigation per
  file, rather than trusting a model to decide what to explore.

This is one instance of the same pattern—not a privileged environment type or a
built-in repository integration. Resolve `HEAD`, install tools, and freeze inputs
outside the core.

## Designing for useful sharing

1. **Keep the task stable.** Put subject selection in the environment, not an
   environment-specific sentence in every initial prompt. Structured subagent
   arguments can deterministically parameterize a task: a different criterion or
   file path then intentionally creates a different child prompt/trie.
2. **Ask for relevant evidence first.** A structured factual query can avoid
   reading a whole document unnecessarily. Detailed evidence is still available
   when needed. Avoid timestamps, random ordering, and storage-location noise.
3. **Contain unavoidable variation.** A specialist may split on full documents
   while returning the same finding to its parent. This costs child work: measure
   total tokens and tool executions, not just the parent's branch count.
4. **Keep the evidence.** Full tool outputs remain recorded. Canonical formatting
   is not permission to discard meaningful differences. Custom `group_by` is an
   explicit, potentially lossy policy—not semantic equivalence established by the core.
5. **Evaluate the investigation.** Matching observations justify replay of that
   recorded path; they do not prove that the agent asked every necessary question.
   Use fixtures, coverage requirements, and feedback to improve it.

## When this is a poor fit

- The subjects need fundamentally different tasks or tool contracts.
- Every early observation is unique and no useful child finding contains that
  variation. Correctness still matters, but sharing may save little.
- Tools mutate state, depend on nondeterministic hidden context, or read changing
  sources without snapshotting. Recording cannot make these safe to replay.
- The desired result is an action, such as sending a payment or changing a record.
  Keep that side effect outside the read-only investigation loop.
- You need cross-agent dependency push updates or a workflow scheduler. Caller
  code can compose investigations today; automatic propagation is still
  [future work](https://github.com/kaeluka/lockstep-loop/issues/1).

For a concrete introduction to the machinery, continue with the
[visual guide](concepts.md). For exact APIs and limitations, see the
[technical reference](history-reference.md).
