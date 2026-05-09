
# Demo: CPSC Recall Change Monitoring System

## Demo Goal
The demo shows how the system automatically compares two CPSC recall snapshots, detects important recall changes, retrieves supporting evidence, generates a risk-oriented summary, and evaluates the output.

## Input
- Previous snapshot: `cpsc_recalls_20260304.csv`
- Current snapshot: `cpsc_recalls_20260430.csv`
- Previous rows: 9631
- Current rows: 9722
- Detected change units: 412

## Demo Flow
1. Load previous and current CPSC recall snapshots.
2. Detect added, removed, and modified recall records.
3. Convert changes into recall-level retrieval documents.
4. Run four generation methods:
   - Prompt-only
   - RAG
   - Agentic RAG
   - Contrastive Adaptive RAG
5. Evaluate each method using recall coverage, precision, F1, grounding rate, and change-unit coverage.
6. Select the strongest method and generate a final risk summary.

## Demo Focus Recall
- Focus recall: `26-418`

## Best Method
- Best method by automatic evaluation: `contrastive_adaptive_rag`

## Main Evaluation Table
| method                   | gold_mode   |   final_recall_coverage |   final_recall_precision |   final_recall_f1 |   final_cu_coverage |   final_cu_weighted_coverage |   final_grounding_rate |   enforce_delta_recall_coverage |   retrieval_p@5 |   retrieval_r@5 |   retrieval_p@10 |   retrieval_r@10 |
|:-------------------------|:------------|------------------------:|-------------------------:|------------------:|--------------------:|-----------------------------:|-----------------------:|--------------------------------:|----------------:|----------------:|-----------------:|-----------------:|
| prompt_only              | human_final |                  0.5    |                   0.8    |            0.6154 |              0.5    |                       0.5    |                      1 |                          0      |             nan |        nan      |            nan   |         nan      |
| rag                      | human_final |                  0.6875 |                   0.5789 |            0.6286 |              0.6875 |                       0.6875 |                      1 |                          0.0625 |               1 |          0.3125 |              0.9 |           0.5625 |
| agentic_rag              | human_final |                  0.875  |                   0.6087 |            0.7179 |              0.875  |                       0.875  |                      1 |                          0.125  |               1 |          0.3125 |              0.9 |           0.5625 |
| contrastive_adaptive_rag | human_final |                  0.9375 |                   0.6522 |            0.7692 |              0.9375 |                       0.9375 |                      1 |                          0.25   |               1 |          0.3125 |              0.9 |           0.5625 |

## Final Takeaway
The demo demonstrates that the system is not only generating a natural-language summary, but also tracing the summary back to detected changes and retrieved evidence. This makes the output more auditable than a pure prompt-only summarization baseline.
