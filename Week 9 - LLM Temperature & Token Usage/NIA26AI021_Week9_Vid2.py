import os
import json

from dotenv import load_dotenv
from groq import Groq


# Load API key
load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

print("API client ready")


# ============================================================
# PROMPTS
# ============================================================

bad_prompt = """
Summarize this support ticket and suggest a response.
"""


good_prompt = """
You are a Tier 2 customer support specialist at TechCorp,
a B2B SaaS company.

Task:
Analyze the support ticket and generate a structured response.

Context:
- Critical issues have a 4-hour SLA.
- Refunds are available for outages exceeding 24 hours.
- Escalation path: Tier 1 -> Tier 2 -> Engineering -> CTO.
- Billing errors should be flagged for the finance team.

Constraints:
- Be empathetic and professional.
- If data loss is mentioned, mark the priority as critical.
- Never promise a specific resolution time.
- Include relevant knowledge base article IDs when applicable.

Return only valid JSON.

Output format:
{
    "category": "billing/technical/account/feature_request/other",
    "priority": "low/medium/high/critical",
    "sentiment": "positive/neutral/negative/furious",
    "needs_engineering": true/false,
    "suggested_response": "draft email reply",
    "kb_articles": ["article_id"],
    "auto_respond": true/false
}

Example:
Input:
"I was charged twice for my subscription."

Output:
{
    "category": "billing",
    "priority": "high",
    "sentiment": "negative",
    "needs_engineering": false,
    "suggested_response": "We are sorry about the billing issue and will investigate it.",
    "kb_articles": ["KB-101"],
    "auto_respond": false
}
"""


# ============================================================
# TEST INPUTS
# ============================================================

tickets = [
    "My billing shows $499 but my plan is $299. This is the third time this year.",
    "Our dashboard has been unavailable for the last six hours and our team cannot access customer reports.",
    "I accidentally deleted an important customer project and now all the project data appears to be gone.",
    "I forgot my password and cannot access my company account.",
    "It would be great if you could add a dark mode option to the dashboard."
]


# ============================================================
# MODEL CALL
# ============================================================

def classify_ticket(prompt, ticket, structured=False):

    request = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": ticket
            }
        ],
        "temperature": 0.2
    }

    if structured:
        request["response_format"] = {
            "type": "json_object"
        }

    response = client.chat.completions.create(**request)

    return response.choices[0].message.content.strip()


# ============================================================
# BAD PROMPT
# ============================================================

print("\n" + "=" * 60)
print("BAD PROMPT")
print("=" * 60)

bad_results = []

for i, ticket in enumerate(tickets, start=1):

    result = classify_ticket(
        bad_prompt,
        ticket
    )

    bad_results.append(result)

    print(f"\nTicket {i}")
    print(result)


# ============================================================
# GOOD PROMPT
# ============================================================

print("\n" + "=" * 60)
print("GOOD PROMPT")
print("=" * 60)

good_results = []

for i, ticket in enumerate(tickets, start=1):

    result = classify_ticket(
        good_prompt,
        ticket,
        structured=True
    )

    good_results.append(result)

    print(f"\nTicket {i}")
    print(result)


# ============================================================
# JSON VERIFICATION
# ============================================================

required_fields = {
    "category",
    "priority",
    "sentiment",
    "needs_engineering",
    "suggested_response",
    "kb_articles",
    "auto_respond"
}

valid_json = 0

for result in good_results:

    try:
        parsed = json.loads(result)

        if required_fields.issubset(parsed.keys()):
            valid_json += 1

    except json.JSONDecodeError:
        pass


print("\n" + "=" * 60)
print("JSON VERIFICATION")
print("=" * 60)

print(f"Valid JSON: {valid_json}/5")


# ============================================================
# BAD PROMPT FORMAT CHECK
# ============================================================

bad_json = 0

for result in bad_results:

    try:
        json.loads(result)
        bad_json += 1

    except json.JSONDecodeError:
        pass


# ============================================================
# THREE FEW-SHOT EXAMPLES
# ============================================================

few_shot_prompt = """
You are a Tier 2 customer support specialist at TechCorp,
a B2B SaaS company.

Task:
Analyze the support ticket and generate a structured response.

Context:
- Critical issues have a 4-hour SLA.
- Refunds are available for outages exceeding 24 hours.
- Escalation path: Tier 1 -> Tier 2 -> Engineering -> CTO.
- Billing errors should be flagged for the finance team.

Constraints:
- Be empathetic and professional.
- If data loss is mentioned, mark the priority as critical.
- Never promise a specific resolution time.
- Include relevant knowledge base article IDs when applicable.

Return only valid JSON.

Output format:
{
    "category": "billing/technical/account/feature_request/other",
    "priority": "low/medium/high/critical",
    "sentiment": "positive/neutral/negative/furious",
    "needs_engineering": true/false,
    "suggested_response": "draft email reply",
    "kb_articles": ["article_id"],
    "auto_respond": true/false
}

Examples:

Input:
"My invoice is higher than expected."

Output:
{
    "category": "billing",
    "priority": "high",
    "sentiment": "negative",
    "needs_engineering": false,
    "suggested_response": "We will investigate the billing difference and work with our finance team to resolve it.",
    "kb_articles": ["KB-101"],
    "auto_respond": false
}

Input:
"The application crashes whenever I open reports."

Output:
{
    "category": "technical",
    "priority": "high",
    "sentiment": "negative",
    "needs_engineering": true,
    "suggested_response": "We have escalated the issue to our engineering team for investigation.",
    "kb_articles": ["KB-205"],
    "auto_respond": false
}

Input:
"I forgot my password and cannot log in."

Output:
{
    "category": "account",
    "priority": "medium",
    "sentiment": "neutral",
    "needs_engineering": false,
    "suggested_response": "Please use the password reset option to regain access to your account.",
    "kb_articles": ["KB-301"],
    "auto_respond": true
}
"""


# ============================================================
# FEW-SHOT TEST
# ============================================================

print("\n" + "=" * 60)
print("FEW-SHOT PROMPT")
print("=" * 60)

few_shot_results = []
few_shot_valid = 0

for i, ticket in enumerate(tickets, start=1):

    result = classify_ticket(
        few_shot_prompt,
        ticket,
        structured=True
    )

    few_shot_results.append(result)

    print(f"\nTicket {i}")
    print(result)

    try:
        parsed = json.loads(result)

        if required_fields.issubset(parsed.keys()):
            few_shot_valid += 1

    except json.JSONDecodeError:
        pass


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(f"\nBad prompt valid JSON: {bad_json}/5")
print(f"Good prompt valid JSON: {valid_json}/5")
print(f"Few-shot valid JSON: {few_shot_valid}/5")

print("\nConclusion:")

print(
    "The good prompt is more suitable for automation because it "
    "defines the role, task, context, constraints, output format, "
    "and example while requiring structured JSON."
)

if few_shot_valid > valid_json:
    print(
        "The few-shot examples improved consistency in this experiment."
    )
elif few_shot_valid == valid_json:
    print(
        "The few-shot examples did not improve the JSON success rate "
        "because the good prompt already achieved the same result."
    )
else:
    print(
        "The few-shot examples produced a lower JSON success rate "
        "in this experiment."
    )