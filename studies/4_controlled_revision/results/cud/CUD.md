| System (backbone, data) | G_app | G_base | G_int | C_oracle | C_realized | CUG |
|---|---|---|---|---|---|---|
| MedAgents_GPT | +10.7 [+8.2, +13.1] | -0.5 [-2.7, +1.5] | — | — | — | — |
| MDAgents_GPT | — | -12.0 [-15.6, -8.5] | — | — | — | — |
| MedLA_GPT | +0.7 [-1.3, +2.6] | -4.8 [-6.5, -3.1] | — | — | — | — |
| MedLA_Llama | +2.7 [-0.4, +5.7] | -11.0 [-13.8, -8.1] | — | — | — | — |
| MedAgents_Qwen14B_first200 | — | -4.0 | -3.2 [-7.5, +1.0] | — | — | — |
| MedAgents_Llama8B_first200 | — | -9.5 [-15.0, -4.0] | -13.3 [-18.7, -8.0] | — | — | — |
| Phase II: MedAgents vs router (Qwen-14B, MedQA PILOT/HELD-OUT) | — | +0.7 [-3.7, +5.1] | -3.9 [-7.7, -0.2] | +3.6 [+1.4, +6.0] (placebo +0.2) | +0.3 [-0.6, +1.5] | +3.3 |
| Phase III: TDC (Qwen-14B, MedMCQA PILOT, n=111 triggered) | — | — | -8.1 [-17.1, +0.0] | +6.3 naive (HET placebo +7.2) | -8.1 [-17.1, +0.0] | +14.4 |

Phase III revisions (n=220 representative revisions): switch 0.495, BRR 0.182, HRR 0.127, NetRevisionUtility +5.5 pp; P(W→C | W) 0.265, P(C→W | C) 0.406

| state | n | rescue | harm | BRR | HRR |
|---|---|---|---|---|---|
| weak | 59 | 4 | 4 | 0.203 | 0.110 |
| split | 23 | 0 | 5 | 0.174 | 0.152 |
| fragmented | 29 | 3 | 7 | 0.143 | 0.143 |

gold among the top-2 answers on triggered questions: {'X': 45, 'Y': 25, 'neither': 41}
