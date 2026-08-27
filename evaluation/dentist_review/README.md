# Independent Clinical Review Protocol — AI-Generated Pre-Clinical Reports

## Purpose

This is a structured evaluation instrument for an independent dentist to assess the
clinical accuracy of the Stage 3 LLM-generated pre-clinical reports (Gemini 2.5 Flash),
as distinct from the automatic text-similarity metrics (Spontaneous Recall, BERTScore)
already reported in the paper. Automatic metrics measure overlap between two pieces of
text; they cannot tell whether the AI report is clinically correct. This protocol asks
a dentist to make that judgment directly, by reading each generated report against the
actual radiograph.

## Sample

15 of the 50 cases, stratified by Stage 2 Spontaneous Recall (SR) to cover a range of
automatic-metric outcomes rather than only the easy cases:

| Bucket | SR range | n | Case IDs |
|---|---|---|---|
| Zero | SR = 0 (YOLO detected nothing matching the dentist's mention) | 3 | 23, 25, 48 |
| Low/mid | 0 < SR < 0.8 | 1 | 03 |
| Not evaluable | No testable class mentioned in the reference description (SR undefined) | 5 | 02, 09, 21, 34, 47 |
| High | SR ≥ 0.8 | 6 | 00, 01, 14, 28, 29, 30 |

The "not evaluable" and "zero" cases matter most: these are exactly the cases where the
paper's own automatic metrics provide no signal (or a failing signal) about the
generated report's quality, so they are the strongest test of whether the report itself
is trustworthy.

## What to do

For each of the 15 rows in `evaluation_sheet.csv`:

1. Open the corresponding image (`image_filename` column) from the original OPG set.
2. Read the AI-generated report (`ai_generated_report` column) side by side with the
   image. The `dentist_reference_description` column is the original clinical
   description written independently for this project — use it only as background
   context (e.g., to know what was previously noted), not as an answer key: judge the
   AI report against the actual radiograph, not against that text.
3. Fill in the six response columns:
   - **correct_findings (Y/N/Partial)** — do the findings stated in the report match
     what you see in the image? "Partial" if some findings are right and others aren't.
   - **missed_findings** — anything clinically relevant visible in the image that the
     report does not mention. Write "none" if nothing was missed.
   - **hallucinated_findings** — anything the report states that is not actually
     present in the image. Write "none" if there are no hallucinations.
   - **clinical_appropriateness_1to5** — are the recommended actions in the report
     reasonable given the actual findings? (1 = inappropriate/unsafe, 5 = fully
     appropriate)
   - **overall_usability_1to5** — would this report be a reasonable starting point for
     a real pre-clinical note, after a dentist reviews and corrects it? (1 = not
     usable, 5 = usable with minimal editing)
   - **comments** — anything else worth noting (optional).

Estimated time: 5-8 minutes per case, roughly 1.5-2 hours for all 15.

## Returning results

Send back the completed `evaluation_sheet.csv` (or the equivalent spreadsheet). No
statistical processing is needed on your end — just the per-case judgments.

## Note on scope

This is not a requirement for the current submission; it is documented in the paper's
Future Work as ongoing independent validation. Completing even a subset of these 15
cases is useful — partial results can still be reported.
