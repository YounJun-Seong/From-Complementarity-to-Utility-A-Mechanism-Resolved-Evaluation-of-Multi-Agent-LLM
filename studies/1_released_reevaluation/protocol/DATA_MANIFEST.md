# Data manifest (sha256)

| file | lines | sha256 | source |
|---|---|---|---|
| `external/MedAgents/datasets/MedQA/test.jsonl` | 1273 | `cfa75cbcc53c29b0…` | MedAgents release datasets/MedQA/test.jsonl (5-option US) |
| `external/data/medqa/test.jsonl` | 1273 | `cfa75cbcc53c29b0…` | MedAgents release datasets/MedQA/test.jsonl (5-option US) |
| `external/data/medqa/train.jsonl` | 10178 | `898a23e63a99073b…` | HF bigbio/med_qa med_qa_en_source (5-option US, original MedQA release) |
| `external/MedLA/data/MIRAGE/rawdata/medqa/data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl` | 1273 | `c3b905ccfa66152d…` | HF GBaker/MedQA-USMLE-4-options phrases_no_exclude_test.jsonl (= MedQA data_clean US 4_options, MIRAGE) |
| `data/medqa_raw/5opt_train.parquet` | 36639 | `e4158a94eacd9e96…` | HF bigbio/med_qa med_qa_en_source (5-option US, original MedQA release) |
| `data/medqa_raw/5opt_test.parquet` | 4795 | `3233cb669a7d5650…` | HF bigbio/med_qa med_qa_en_source (5-option US, original MedQA release) |

- 5-option test (bigbio) matches MedAgents test question-for-question and in order (1273/1273).
- 4-option test matches the same 1273 questions in the same order (option E removed).
- external/data/medqa/train.jsonl = bigbio 5-option train (10178) converted to {question, answer, options{A..E}, meta_info, answer_idx}.
