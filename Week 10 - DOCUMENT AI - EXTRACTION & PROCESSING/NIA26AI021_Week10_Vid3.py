import os
import json
from dotenv import load_dotenv
from groq import Groq


# Load API key
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# --------------------------------------------------
# TOOL FUNCTIONS
# --------------------------------------------------

def lookup_customer(email):
    customers = {
        "john@email.com": {
            "name": "John Doe",
            "email": "john@email.com",
            "plan": "Pro",
            "account_age": "3 years"
        },
        "jane@email.com": {
            "name": "Jane Smith",
            "email": "jane@email.com",
            "plan": "Enterprise",
            "account_age": "5 years"
        },
        "fail@email.com": {
            "name": "Test Customer",
            "email": "fail@email.com",
            "plan": "Pro",
            "account_age": "2 years"
        }
    }

    customer = customers.get(email)

    if customer is None:
        return {
            "error": "Customer not found"
        }

    return customer


def check_account_security(email):
    security_status = {
        "john@email.com": {
            "status": "suspicious activity detected",
            "action": "immediate security review required"
        },
        "jane@email.com": {
            "status": "no suspicious activity",
            "action": "no immediate security action required"
        },
        "fail@email.com": {
            "status": "suspicious activity detected",
            "action": "immediate security review required"
        }
    }

    return security_status.get(
        email,
        {"error": "Security information not found"}
    )


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


# Added tool 1
def get_customer_orders(email):
    orders = {
        "john@email.com": [
            {
                "order_id": "ORD-1001",
                "product": "Business Software Plan",
                "status": "active"
            }
        ],
        "jane@email.com": [
            {
                "order_id": "ORD-1002",
                "product": "Enterprise Software Plan",
                "status": "active"
            }
        ],
        "fail@email.com": []
    }

    if email not in orders:
        return {
            "error": "Customer orders not found"
        }

    return {
        "orders": orders[email]
    }


# Added tool 2
def send_support_email(email, subject, message):
    if email == "fail@email.com":
        return {
            "error": "Email could not be sent"
        }

    return {
        "status": "email sent",
        "recipient": email,
        "subject": subject
    }


# --------------------------------------------------
# TOOL SCHEMAS
# --------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": "Look up a customer using their email address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string"
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
            "description": "Check the security status of a customer account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string"
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
                        "type": "string"
                    },
                    "priority": {
                        "type": "string"
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_orders",
            "description": "Retrieve the customer's orders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string"
                    }
                },
                "required": ["email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_support_email",
            "description": "Send a support email to the customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string"
                    },
                    "subject": {
                        "type": "string"
                    },
                    "message": {
                        "type": "string"
                    }
                },
                "required": [
                    "email",
                    "subject",
                    "message"
                ]
            }
        }
    }
]


# --------------------------------------------------
# AVAILABLE FUNCTIONS
# --------------------------------------------------

available_functions = {
    "lookup_customer": lookup_customer,
    "check_account_security": check_account_security,
    "create_support_ticket": create_support_ticket,
    "get_customer_orders": get_customer_orders,
    "send_support_email": send_support_email
}


# --------------------------------------------------
# AGENT LOOP
# --------------------------------------------------

def run_agent(user_message):
    messages = [
        {
            "role": "system",
            "content": """
You are a customer support AI agent.

Use the available tools when necessary.

For account or security issues:
1. Look up the customer first.
2. Check account security.
3. Retrieve customer orders when relevant.
4. Create a support ticket when necessary.
5. Send a support email when appropriate.

You may make multiple tool calls in sequence.

If a tool returns an error:
- Do not invent information.
- Explain the problem clearly.
- Handle the error gracefully.
- Still provide a useful final response.
"""
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    for step in range(8):
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=700
        )

        assistant_message = response.choices[0].message

        messages.append(assistant_message)

        if not assistant_message.tool_calls:
            return assistant_message.content

        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_arguments = json.loads(
                tool_call.function.arguments
            )

            print(f"Tool called: {function_name}")

            if function_name not in available_functions:
                result = {
                    "error": "Unknown function"
                }
            else:
                try:
                    function_to_call = available_functions[
                        function_name
                    ]

                    result = function_to_call(
                        **function_arguments
                    )

                except Exception as error:
                    result = {
                        "error": str(error)
                    }

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                }
            )

    return "The agent reached the maximum number of steps."


# --------------------------------------------------
# TEST CASES
# --------------------------------------------------

print("\n" + "=" * 60)
print("TEST 1: MULTI-STEP CUSTOMER SUPPORT REQUEST")
print("=" * 60)

result_1 = run_agent(
    """
John Doe cannot access his account and believes someone may have
accessed it. Look up his account, check his security status, retrieve
his orders, create a high-priority support ticket if necessary, and
send him a support email explaining the next steps.
"""
)

print("\nFINAL RESULT:")
print(result_1)


print("\n" + "=" * 60)
print("TEST 2: CUSTOMER NOT FOUND")
print("=" * 60)

result_2 = run_agent(
    """
Look up the customer unknown@email.com, check the account security,
and help with the account access problem.
"""
)

print("\nFINAL RESULT:")
print(result_2)


print("\n" + "=" * 60)
print("TEST 3: TOOL FAILURE")
print("=" * 60)

result_3 = run_agent(
    """
The customer fail@email.com believes the account was hacked.
Look up the customer, check the security status, retrieve the orders,
create a support ticket, and send a support email.
"""
)

print("\nFINAL RESULT:")
print(result_3)


print("\n" + "=" * 60)
print("VID 3 COMPLETE")
print("=" * 60)