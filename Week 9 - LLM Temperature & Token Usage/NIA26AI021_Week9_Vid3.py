import os
import json

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# SETUP
# ============================================================

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

print("API client ready")


# ============================================================
# FUNCTION 1 - LOOK UP CUSTOMER
# ============================================================

def lookup_customer(email):
    customers = {
        "john@email.com": {
            "name": "John Doe",
            "plan": "Pro",
            "tenure": "3 years"
        },
        "jane@email.com": {
            "name": "Jane Smith",
            "plan": "Enterprise",
            "tenure": "5 years"
        },
        "fail@email.com": {
            "name": "Test Customer",
            "plan": "Pro",
            "tenure": "2 years"
        }
    }

    if email in customers:
        return customers[email]

    return {
        "error": "Customer not found"
    }


# ============================================================
# FUNCTION 2 - CHECK ACCOUNT SECURITY
# ============================================================

def check_account_security(email):
    security_status = {
        "john@email.com": {
            "status": "suspicious activity detected",
            "action": "immediate security review recommended"
        },
        "jane@email.com": {
            "status": "no suspicious activity detected",
            "action": "no immediate security action required"
        },
        "fail@email.com": {
            "status": "suspicious activity detected",
            "action": "immediate security review recommended"
        }
    }

    if email in security_status:
        return security_status[email]

    return {
        "error": "Customer security record not found"
    }


# ============================================================
# FUNCTION 3 - CREATE SUPPORT TICKET
# ============================================================

def create_support_ticket(
    customer_email,
    category,
    priority,
    subject,
    description
):
    if customer_email == "fail@email.com":
        return {
            "error": "Ticket creation failed"
        }

    return {
        "ticket_id": "TKT-76423",
        "status": "created",
        "assigned_to": "Tier 1 Support",
        "estimated_response": "1 hour"
    }


# ============================================================
# TOOL DEFINITIONS
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": "Look up a customer by email address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "Customer email address"
                    }
                },
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_account_security",
            "description": "Check whether a customer's account has suspicious security activity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "Customer email address"
                    }
                },
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": "Create a support ticket for a customer issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_email": {
                        "type": "string"
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "billing",
                            "technical",
                            "account",
                            "general"
                        ]
                    },
                    "priority": {
                        "type": "string",
                        "enum": [
                            "low",
                            "medium",
                            "high",
                            "critical"
                        ]
                    },
                    "subject": {
                        "type": "string"
                    },
                    "description": {
                        "type": "string"
                    }
                },
                "required": [
                    "customer_email",
                    "category",
                    "priority",
                    "subject",
                    "description"
                ]
            }
        }
    }
]


# ============================================================
# FUNCTION MAP
# ============================================================

available_functions = {
    "lookup_customer": lookup_customer,
    "check_account_security": check_account_security,
    "create_support_ticket": create_support_ticket
}


# ============================================================
# AGENT LOOP
# ============================================================

def run_agent(user_message):

    messages = [
        {
            "role": "system",
            "content": """
You are a customer support AI agent.

Use the available tools when necessary.

Workflow for account or security issues:
1. Look up the customer first.
2. If the customer is found, check account security when relevant.
3. If security review is needed, create a support ticket.
4. Use the result of each tool call to decide what to do next.

You may make multiple tool calls in sequence.

If a tool returns an error:
- Do not invent information.
- Explain the problem clearly.
- Handle the error gracefully.

Only provide a final response after the required tool calls are completed.
"""
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    for step in range(1, 6):

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=500
        )

        assistant_message = response.choices[0].message

        messages.append(assistant_message)

        # Final response
        if not assistant_message.tool_calls:
            print("\nFinal response:")
            print(assistant_message.content)
            return assistant_message.content

        # Execute tool calls
        for tool_call in assistant_message.tool_calls:

            function_name = tool_call.function.name

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )
            except json.JSONDecodeError:
                arguments = {}
                result = {
                    "error": "Invalid tool arguments"
                }
            else:
                function = available_functions.get(function_name)

                if function is None:
                    result = {
                        "error": f"Unknown function: {function_name}"
                    }
                else:
                    try:
                        result = function(**arguments)
                    except Exception as error:
                        result = {
                            "error": str(error)
                        }

            print(f"\nStep {step}: {function_name}")
            print("Arguments:", arguments)
            print("Result:", result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                }
            )

    print("\nFinal response:")
    print("The agent could not complete the request.")
    return "The agent could not complete the request."


# ============================================================
# TEST 1 - MULTIPLE TOOL CALLS IN SEQUENCE
# ============================================================

print("\n" + "=" * 60)
print("TEST 1 - MULTI-STEP REQUEST")
print("=" * 60)

run_agent(
    "Hi, I'm john@email.com and I can't access my account. "
    "I think it has been hacked."
)


# ============================================================
# TEST 2 - CUSTOMER NOT FOUND
# ============================================================

print("\n" + "=" * 60)
print("TEST 2 - CUSTOMER NOT FOUND")
print("=" * 60)

run_agent(
    "Hi, I'm unknown@email.com and I can't access my account."
)


# ============================================================
# TEST 3 - TICKET CREATION FAILURE
# ============================================================

print("\n" + "=" * 60)
print("TEST 3 - TICKET CREATION FAILURE")
print("=" * 60)

run_agent(
    "Hi, I'm fail@email.com and I can't access my account. "
    "I think it has been hacked."
)