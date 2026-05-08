Each JSONL row is a single matched conversational example.

Required fields:

- `scenario_id`
- `user_message`
- `response_style_label`
- `assistant_target_text`
- `notes`

The probe scaffold expects multiple rows with the same `scenario_id` and `user_message`, differing mainly in `response_style_label` and `assistant_target_text`.
