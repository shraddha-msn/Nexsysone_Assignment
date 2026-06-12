# Telecom Tower Lease Vetting Agent

An autonomous AI agent that evaluates incoming lease applications from mobile operators against tower capacity constraints and regional municipality policies.

## Overview

When a telecom operator submits a plain-English request to mount equipment on a tower, this agent:

1. Extracts the operator name, equipment weight, mounting height, and tower ID from the request
2. Calls `check_tower_capacity` to verify the tower can bear the additional load
3. Calls `check_regional_policy` to verify the equipment meets zone-specific municipality rules
4. Returns a structured JSON verdict — `APPROVED` or `REJECTED` — with full reasoning

The agent is built on the **agentic tool-calling pattern**: the LLM decides which tools to call, in what order, and with what arguments — driven entirely by the plain-text input. Tool results are fed back into the conversation so the model can synthesize a grounded final judgment rather than guessing.

---

## Project Structure

```
.
├── agent.py                  # Agent logic, tool implementations, agentic loop
├── towers_inventory.json     # Towers across UAE emirates with capacity data
├── regional_policies.txt     # Municipality rules per zone (height, weight, special conditions)
├── output.txt                # Sample output from running all 3 test cases
└── README.md
```

---

## Setup

**Requirements:** Python 3.9+

Install the dependency:

```bash
pip install groq
```

Get a free Groq API key from [console.groq.com](https://console.groq.com) — no billing required.

Set the key as an environment variable:

```bash
# macOS / Linux
export GROQ_API_KEY="your-key-here"

# Windows PowerShell
$env:GROQ_API_KEY = "your-key-here"
```

---

## Running the Agent

**Run the built-in test cases:**

```bash
python agent.py
```

**Run a custom lease request:**

```bash
python agent.py "Operator Etisalat wants to mount a 20kg antenna at 35 meters on Tower TWR-114."
```

---

## Example Output

**Input:**
```
Operator Du wants to mount a 15kg 5G antenna at a height of 40 meters on Tower TWR-101.
```

**Output:**
```json
{
  "status": "APPROVED",
  "operator": "Du",
  "tower_id": "TWR-101",
  "region": "DXB-North",
  "reason": "Both capacity and regional policy checks passed. Projected tower load (475kg) is within the 500kg limit, and the equipment height (40m) and weight (15kg) comply with DXB-North zone rules.",
  "details": {
    "equipment_weight_kg": 15,
    "mounting_height_m": 40,
    "tower_capacity_check": {
      "feasible": true,
      "current_weight_kg": 460,
      "max_allowed_weight_kg": 500,
      "projected_total_weight_kg": 475
    },
    "regional_policy_check": {
      "compliant": true,
      "zone_max_height_m": 45,
      "zone_max_tenant_weight_kg": 100
    }
  }
}
```

---

## Test Cases

Three scenarios are included in `main()`:

| # | Request | Expected | Reason |
|---|---------|----------|--------|
| 1 | Du, 15kg, 40m, TWR-101 | APPROVED | 460+15=475kg < 500kg limit; 40m < 45m DXB-North limit |
| 2 | Etisalat, 30kg, 20m, TWR-102 | REJECTED | 30kg exceeds SHJ-Coastal per-tenant limit of 25kg |
| 3 | Vodafone, 10kg, 30m, TWR-999 | REJECTED | Tower TWR-999 does not exist in inventory |

---

## Design Decisions

**Why an agentic loop instead of a single function call?**
The LLM needs two separate tool calls — capacity check first (to discover the region), then policy check (using that region). A single-shot call cannot chain dependent lookups. The loop lets the model act on intermediate results, which mirrors how real agentic workflows handle multi-step reasoning.

**Why Groq (llama-3.3-70b-versatile)?**
Fast inference, free tier with no billing required, and reliable native function-calling support sufficient for this use case.

**Why ground the LLM with tools instead of letting it reason from raw file contents?**
Passing raw JSON/text into the prompt and asking the model to do arithmetic is error-prone. Structured tool functions enforce correct logic and make the checks auditable — the tool result is deterministic regardless of how the model phrases the question.
