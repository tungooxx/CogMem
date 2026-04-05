# CogMem v1 Memory Bank — Example Episodes

**Model:** qwen2.5:3b | **Dataset:** BigCodeBench Full (1140 tasks) | **Pass rate:** 254/1140 (22.3%)

## Passed Examples (Q = 1.0)

### BigCodeBench/0 — "Calculates the average of the sums of..."
- **Q-value:** 1.0 (PASS)
- **Description:** Calculates the average of the sums of absolute values in each row of a matrix
- **Status:** Code runs correctly, passes all test cases
- **Use in training:** SFT learns this pattern (Q-weighted: 3 copies)

### BigCodeBench/1 — "Generate a random string of the speci..."
- **Q-value:** 1.0 (PASS)
- **Description:** Generate a random string of the specified length
- **Status:** Correct implementation
- **Use in training:** SFT training example

### BigCodeBench/7 — "Find the best-selling product from a ..."
- **Q-value:** 1.0 (PASS)
- **Description:** Find the best-selling product from a nested list of sales data
- **Status:** Correct pandas/numpy usage
- **Use in training:** SFT training example

### BigCodeBench/9 — "Create a Pandas DataFrame from a list..."
- **Q-value:** 1.0 (PASS)
- **Description:** Create a Pandas DataFrame from a list of dictionaries
- **Status:** Correct DataFrame construction
- **Use in training:** SFT training example

### BigCodeBench/1139 — "Train a simple linear regression mode..."
- **Q-value:** 1.0 (PASS)
- **Description:** Train a simple linear regression model
- **Status:** Correct sklearn usage
- **Use in training:** SFT training example

---

## Failed Examples (Q = -1.0)

### BigCodeBench/2 — "Create a dictionary in which keys are..."
- **Q-value:** -1.0 (FAIL)
- **Error type:** TestFailure
- **Description:** Create a dictionary where keys are random strings
- **What went wrong:** Code runs but produces wrong output
- **Use in training:** DPO "rejected" example (contrast with similar passing tasks)

### BigCodeBench/8 — "Convert elements in 'T1' to integers..."
- **Q-value:** -1.0 (FAIL)
- **Error type:** TypeError
- **Description:** Convert elements in 'T1' to integers
- **What went wrong:** Wrong type conversion, called wrong method
- **Use in training:** DPO "rejected" (BigCodeBench/10 is same task, PASS → forms preference pair)

### BigCodeBench/13 — "Download all files from a specific di..."
- **Q-value:** -1.0 (FAIL)
- **Error type:** Timeout
- **Description:** Download all files from a specific directory on an FTP server
- **What went wrong:** Requires network access (FTP), impossible in sandbox
- **Use in training:** Skip — impossible task, not a model failure

### BigCodeBench/21 — "Obtain system details, including oper..."
- **Q-value:** -1.0 (FAIL)
- **Error type:** TypeError
- **Description:** Obtain system details including operating system info
- **What went wrong:** Called wrong system API method
- **Use in training:** DPO "rejected" example

### BigCodeBench/1138 — "Sorts a numeric 2D numpy array in asc..."
- **Q-value:** -1.0 (FAIL)
- **Error type:** TestFailure
- **Description:** Sorts a numeric 2D numpy array in ascending order
- **What went wrong:** Logic error in sorting implementation
- **Use in training:** DPO "rejected" example

---

## DPO Preference Pair Examples

DPO needs (chosen, rejected) pairs from the same or similar tasks:

### Pair 1: BigCodeBench/8 vs BigCodeBench/10
- **Task:** Convert elements in 'T1' to integers
- **Chosen (Q=1.0):** BigCodeBench/10 — correct type conversion
- **Rejected (Q=-1.0):** BigCodeBench/8 — TypeError, wrong method
- **What DPO learns:** Correct int() conversion pattern

### Pair 2: BigCodeBench/3 vs BigCodeBench/2
- **Task:** Create a dictionary with specific keys
- **Chosen (Q=1.0):** BigCodeBench/3 — correct dict comprehension
- **Rejected (Q=-1.0):** BigCodeBench/2 — wrong output format
- **What DPO learns:** Correct dictionary construction pattern

### Pair 3: BigCodeBench/10 vs BigCodeBench/11 (both PASS, but different Q after Cycle 2)
- After Cycle 2, these may have different Q-values (e.g., 0.9 vs 0.6)
- DPO uses the Q-gap to create preference pairs even among successes
- This teaches subtle quality differences, not just pass/fail

---

## Q-Value Zones (After Cycle 2+)

After multiple cycles, Q-values distribute into three zones:

| Zone | Q Range | Count (est.) | Training Use |
|---|---|---|---|
| **High** | >= 0.7 | ~200 | SFT (memorize proven patterns) |
| **Middle** | 0.3 - 0.7 | ~150 | DPO pairs (learn from inconsistency) |
| **Low** | < 0.3 | ~790 | Skip (impossible or too hard) |

Currently (Cycle 1): High=254, Middle=0, Low=886 (binary Q-values)
After Cycle 2: expect Middle zone to grow as some tasks flip between pass/fail
