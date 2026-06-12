"""
Telecom Tower Lease Vetting Agent

Uses the Groq API with tool-calling to autonomously vet incoming
lease applications against tower capacity constraints and regional policies.
"""

import json
import os
import re
import sys
from pathlib import Path

from groq import Groq

# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent


def _load_towers() -> dict[str, dict]:
    path = BASE_DIR / "towers_inventory.json"
    towers = json.loads(path.read_text(encoding="utf-8"))
    return {t["tower_id"]: t for t in towers}


def _load_policies() -> list[str]:
    path = BASE_DIR / "regional_policies.txt"
    return path.read_text(encoding="utf-8").strip().splitlines()


TOWERS: dict[str, dict] = _load_towers()
POLICIES: list[str] = _load_policies()

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def check_tower_capacity(tower_id: str, equipment_weight_kg: float) -> dict:
    tower = TOWERS.get(tower_id)
    if tower is None:
        return {
            "found": False,
            "tower_id": tower_id,
            "message": f"Tower '{tower_id}' does not exist in the inventory.",
        }

    if tower.get("status") != "active":
        return {
            "found": True,
            "tower_id": tower_id,
            "status": tower["status"],
            "feasible": False,
            "message": f"Tower '{tower_id}' is currently '{tower['status']}' and cannot accept new equipment.",
        }

    current = tower["current_weight_kg"]
    max_allowed = tower["max_allowed_weight_kg"]
    projected = current + equipment_weight_kg
    available = max_allowed - current
    feasible = projected <= max_allowed

    return {
        "found": True,
        "tower_id": tower_id,
        "region": tower["region"],
        "max_height_m": tower["max_height_m"],
        "status": tower["status"],
        "current_weight_kg": current,
        "max_allowed_weight_kg": max_allowed,
        "available_capacity_kg": round(available, 2),
        "requested_weight_kg": equipment_weight_kg,
        "projected_total_weight_kg": round(projected, 2),
        "feasible": feasible,
        "message": (
            f"Weight check PASSED: projected total {projected}kg <= {max_allowed}kg limit."
            if feasible
            else f"Weight check FAILED: projected total {projected}kg exceeds {max_allowed}kg limit by {round(projected - max_allowed, 2)}kg."
        ),
    }


def check_regional_policy(
    region: str, equipment_height_m: float, equipment_weight_kg: float
) -> dict:
    zone_key = region.replace("-", " ")
    relevant = [
        line for line in POLICIES
        if zone_key.lower() in line.lower() or region.lower() in line.lower()
    ]

    if not relevant:
        return {
            "region": region,
            "found": False,
            "message": f"No policy found for region '{region}'. Manual review required.",
        }

    combined_policy = " ".join(relevant)

    height_match = re.search(r"height[^.]*?(\d+)\s*meters?", combined_policy, re.IGNORECASE)
    max_height = float(height_match.group(1)) if height_match else None

    weight_match = re.search(
        r"(?:no single tenant asset may exceed|maximum single[- ]tenant equipment weight[^.]*?)\s*(\d+)\s*kg",
        combined_policy,
        re.IGNORECASE,
    )
    max_weight = float(weight_match.group(1)) if weight_match else None

    special_conditions = []
    if "corrosion-resistant" in combined_policy.lower():
        special_conditions.append("Corrosion-resistant hardware mandatory.")
    if "heritage" in combined_policy.lower():
        special_conditions.append("Heritage area: municipality approval required.")
    if "anti-ice" in combined_policy.lower():
        special_conditions.append("Anti-ice coating required.")

    height_ok = (equipment_height_m <= max_height) if max_height is not None else True
    weight_ok = (equipment_weight_kg <= max_weight) if max_weight is not None else True
    compliant = height_ok and weight_ok

    violations = []
    if not height_ok:
        violations.append(f"Height {equipment_height_m}m exceeds zone limit of {max_height}m.")
    if not weight_ok:
        violations.append(f"Equipment weight {equipment_weight_kg}kg exceeds per-tenant limit of {max_weight}kg.")

    return {
        "region": region,
        "found": True,
        "policy_text": " | ".join(relevant),
        "zone_max_height_m": max_height,
        "zone_max_tenant_weight_kg": max_weight,
        "requested_height_m": equipment_height_m,
        "requested_weight_kg": equipment_weight_kg,
        "height_compliant": height_ok,
        "weight_compliant": weight_ok,
        "compliant": compliant,
        "special_conditions": special_conditions,
        "violations": violations,
        "message": (
            "Policy check PASSED: all regional constraints met."
            if compliant
            else "Policy check FAILED: " + " ".join(violations)
        ),
    }


# ---------------------------------------------------------------------------
# Tool registry for Groq
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "check_tower_capacity": check_tower_capacity,
    "check_regional_policy": check_regional_policy,
}

GROQ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_tower_capacity",
            "description": (
                "Check whether a specific tower has sufficient structural weight capacity "
                "to accommodate new equipment. Returns current load, maximum capacity, "
                "available headroom, and a feasibility verdict."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tower_id": {
                        "type": "string",
                        "description": "The tower identifier, e.g. 'TWR-101'.",
                    },
                    "equipment_weight_kg": {
                        "type": "number",
                        "description": "Weight of the equipment the operator wants to mount, in kilograms.",
                    },
                },
                "required": ["tower_id", "equipment_weight_kg"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_regional_policy",
            "description": (
                "Check whether proposed equipment meets the regional municipality rules "
                "for the tower's zone. Validates height limits, per-tenant weight limits, "
                "and surfaces any special installation requirements."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": "The region/zone of the tower, e.g. 'DXB-North'.",
                    },
                    "equipment_height_m": {
                        "type": "number",
                        "description": "Mounting height of the equipment on the structure, in meters.",
                    },
                    "equipment_weight_kg": {
                        "type": "number",
                        "description": "Weight of the equipment in kilograms.",
                    },
                },
                "required": ["region", "equipment_height_m", "equipment_weight_kg"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous vetting agent for a telecom tower lease management platform.

Your job is to evaluate incoming lease applications from mobile operators and produce a structured verdict.

When given a lease request:
1. Extract: operator name, equipment weight (kg), mounting height (m), and tower ID.
2. Call check_tower_capacity with the tower ID and equipment weight.
3. Read the "region" field from the check_tower_capacity result. Use THAT region value — do not guess or infer the region from any other source — to call check_regional_policy with the region, height, and weight.
4. If the tower was not found (found=false), skip step 3 and reject immediately.
5. Synthesize both results into a final verdict.

Your final response MUST be a single valid JSON object with absolutely no text before or after it — no explanations, no markdown, no code fences, no prose:
{
  "status": "APPROVED" or "REJECTED",
  "operator": "<operator name>",
  "tower_id": "<tower id>",
  "region": "<region>",
  "reason": "<concise explanation of the decision>",
  "details": {
    "equipment_weight_kg": <number>,
    "mounting_height_m": <number>,
    "tower_capacity_check": <the full result from check_tower_capacity>,
    "regional_policy_check": <the full result from check_regional_policy, or null if skipped>
  }
}

Approve only if BOTH checks pass. Reject if either check fails, citing the specific constraint violated."""

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_vetting_agent(lease_request: str) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=api_key)
    model = "llama-3.3-70b-versatile"

    print(f"\n[Agent] Processing: {lease_request!r}")
    print(f"[Agent] Model: {model}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": lease_request},
    ]

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=GROQ_TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        print(f"[Agent] finish_reason={finish_reason}")

        # Append assistant message to history
        messages.append(msg)

        if not msg.tool_calls:
            # No tool calls — extract final text and parse JSON
            final_text = msg.content or ""
            # Strip markdown fences, then extract the outermost JSON object
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", final_text).strip()
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            try:
                judgment = json.loads(match.group() if match else clean)
            except (json.JSONDecodeError, AttributeError):
                judgment = {
                    "status": "ERROR",
                    "reason": "Agent returned non-JSON response.",
                    "raw_response": final_text,
                }
            return judgment

        # Execute each tool call
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            print(f"[Tool] {fn_name}({json.dumps(fn_args)})")

            fn = TOOL_FUNCTIONS.get(fn_name)
            if fn is None:
                result = {"error": f"Unknown tool: {fn_name}"}
            else:
                result = fn(**fn_args)

            print(f"[Tool] Result: {json.dumps(result, indent=2)}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    test_cases = [
        # Standard success case — should be APPROVED
        "Operator Du wants to mount a 15kg 5G antenna at a height of 40 meters on Tower TWR-101.",
        # Should REJECT — exceeds per-tenant weight limit for SHJ-Coastal (max 25kg)
        "Operator Etisalat wants to install a 30kg repeater at 20 meters on Tower TWR-102.",
        # Should REJECT — tower does not exist
        "Operator Vodafone wants to mount a 10kg antenna at 30 meters on Tower TWR-999.",
    ]

    requests_to_run = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else test_cases

    for req in requests_to_run:
        print("\n" + "=" * 70)
        judgment = run_vetting_agent(req)
        print("\n[VERDICT]")
        print(json.dumps(judgment, indent=2))
        print("=" * 70)


if __name__ == "__main__":
    main()
