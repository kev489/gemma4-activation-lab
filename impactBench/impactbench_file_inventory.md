# ImpactBench File Inventory

This document lists the ImpactBench-related files currently present in this workspace and describes their contents.

## Source

- Site: https://impactbench.media.mit.edu/
- Transcript endpoint pattern: `https://impactbench.from.pub/scenarios/{benchmark}/{model}/{scenario_id}.json`

## Counts

- Models represented: 14
- Scenario rows per model: 2,160
- Total model-scenario transcript payloads fetched: 30,240
- Total failed transcript fetches: 0
- Unique scenario IDs in the public scenario index: 240
- Metric keys in the public scenario index: 360

## Files

| File | Size | Contents |
| --- | ---: | --- |
| `impactbench_file_inventory.md` | 8.0K | This inventory file. |
| `SHA256SUMS.txt` | 4.0K | SHA-256 checksums for the archive files and this inventory file. |
| `impactbench_transcripts/all_models/manifest.json` | 8.0K | Summary of the all-model pull, including model names, expected payload count, fetched count, failed count, and output paths. |

## Transcript Archives

The `impactbench_transcripts/all_models/` folder contains gzip-compressed JSONL transcript payloads for all 14 models. Each `*_transcripts.jsonl.gz` file contains 2,160 records, one JSON object per scenario row.

| File | Size | Records |
| --- | ---: | ---: |
| `impactbench_transcripts/all_models/claude-haiku-4-5_transcripts.jsonl.gz` | 9.1M | 2,160 |
| `impactbench_transcripts/all_models/claude-opus-4-6_transcripts.jsonl.gz` | 12M | 2,160 |
| `impactbench_transcripts/all_models/claude-sonnet-4-6_transcripts.jsonl.gz` | 8.4M | 2,160 |
| `impactbench_transcripts/all_models/deepseek-v3_transcripts.jsonl.gz` | 21M | 2,160 |
| `impactbench_transcripts/all_models/gemini-2-5-pro_transcripts.jsonl.gz` | 31M | 2,160 |
| `impactbench_transcripts/all_models/gemma-4-31b_transcripts.jsonl.gz` | 14M | 2,160 |
| `impactbench_transcripts/all_models/gpt-4o_transcripts.jsonl.gz` | 10M | 2,160 |
| `impactbench_transcripts/all_models/gpt-5_transcripts.jsonl.gz` | 13M | 2,160 |
| `impactbench_transcripts/all_models/gpt-5-1_transcripts.jsonl.gz` | 27M | 2,160 |
| `impactbench_transcripts/all_models/grok-4-1_transcripts.jsonl.gz` | 27M | 2,160 |
| `impactbench_transcripts/all_models/grok-4-1-reasoning_transcripts.jsonl.gz` | 15M | 2,160 |
| `impactbench_transcripts/all_models/llama-4_transcripts.jsonl.gz` | 9.0M | 2,160 |
| `impactbench_transcripts/all_models/mistral-small-3_transcripts.jsonl.gz` | 14M | 2,160 |
| `impactbench_transcripts/all_models/qwen3-80b_transcripts.jsonl.gz` | 24M | 2,160 |

Compressed size of the transcript archive files plus manifest and inventory: 234M.

Uncompressed size reported by `gzip -l` for the all-model transcript JSONL files: 750,334,619 bytes.

## Removed Redundant Files

The following previously generated files were removed because their contents are duplicated by or derivable from the all-model transcript archives and manifest:

- `impactbench_questions.csv`
- `impactbench_questions.json`
- `impactbench_full_scenarios.csv`
- `impactbench_full_scenarios_with_gpt5_transcripts.json`
- `impactbench_gpt5_transcript_messages.csv`
- `impactbench_all_transcript_urls.json`
- `impactbench_transcripts/gpt-5_transcripts.jsonl`
- `impactbench_transcripts/all_models/*_failures.json`

## Record Shape

Each transcript JSONL record contains these top-level fields:

- `benchmark`
- `metric_id`
- `metric_name`
- `metric_criterion`
- `behavior_type`
- `measurement`
- `harmful`
- `locations`
- `age`
- `scenario_id`
- `scenario_title`
- `transcript_model`
- `transcript_url`
- `scenario`
- `samples`
- `verdict`

The nested `scenario` object contains:

- `id`
- `metric_id`
- `metric`
- `title`
- `description`
- `user_persona`
- `user_goal`
- `latent_adversarial_goal`
- `landmarks`
- `demographic`
- `base_scenario_id`

The `samples` field contains transcript message arrays with `role` and `content`.

The `verdict` field contains the model-specific evaluation result and justification.
