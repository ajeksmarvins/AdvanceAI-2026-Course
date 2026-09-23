import os
import json
from dotenv import load_dotenv
from groq import Groq


# =========================
# SETUP
# =========================

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Setting completed")


# =========================
# MULTI-STEP PIPELINE
# =========================

def process_customer_feedback(feedback_text):

    # Step 1: Summarize
    print("\nProcessing customer feedback")
    print("-" * 40)

    summary_response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "Summarize this customer feedback in 2 sentences."
            },
            {
                "role": "user",
                "content": feedback_text
            }
        ],
        temperature=0.3
    )

    summary = summary_response.choices[0].message.content

    print("Summary:", summary)


    # Step 2: Classify sentiment
    print("\nClassifying sentiment")

    sentiment_response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the customer feedback sentiment. "
                    "Return JSON with a sentiment field as "
                    "positive, negative, or neutral."
                )
            },
            {
                "role": "user",
                "content": summary
            }
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    sentiment_json = json.loads(
        sentiment_response.choices[0].message.content
    )

    sentiment = sentiment_json.get("sentiment")

    print("Sentiment:", sentiment)


    # Step 3: Extract action items
    print("\nExtracting action items")

    action_response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract action items from the customer feedback. "
                    "Return JSON with actions as a list of strings."
                )
            },
            {
                "role": "user",
                "content": summary
            }
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    action_json = json.loads(
        action_response.choices[0].message.content
    )

    actions = action_json.get("actions", [])

    print("Action Items:", actions)


    # Conditional logic
    response_text = None

    if sentiment == "negative":

        print("\nDrafting response")

        response_result = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a customer service manager. "
                        "Draft a professional, empathetic response."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Customer feedback summary: "
                        + summary
                    )
                }
            ],
            temperature=0.5
        )

        response_text = response_result.choices[0].message.content

        print("Drafted Response:", response_text)

    else:

        print("No response needed for non-negative feedback")


    print("\nPipeline Complete")

    return {
        "summary": summary,
        "sentiment": sentiment,
        "actions": actions,
        "draft_response": response_text
    }


# =========================
# TEST CASE 1
# =========================

feedback_1 = """
I have been using your product for a few months now, and I am quite
disappointed. The recent update made the dashboard slower and I now
have difficulty finding my reports.
"""

print("\n" + "=" * 60)
print("TEST CASE 1")
print("=" * 60)

result_1 = process_customer_feedback(feedback_1)

print("\nFINAL RESULT:")
print(json.dumps(result_1, indent=4))


# =========================
# TEST CASE 2
# =========================

feedback_2 = """
The new update looks great and the interface is much easier to use.
I especially like the new dashboard and the improved navigation.
"""

print("\n" + "=" * 60)
print("TEST CASE 2")
print("=" * 60)

result_2 = process_customer_feedback(feedback_2)

print("\nFINAL RESULT:")
print(json.dumps(result_2, indent=4))


# =========================
# TEST CASE 3
# =========================

feedback_3 = """
The product is okay overall, but I would like to see better reporting
options and more customization in the dashboard.
"""

print("\n" + "=" * 60)
print("TEST CASE 3")
print("=" * 60)

result_3 = process_customer_feedback(feedback_3)

print("\nFINAL RESULT:")
print(json.dumps(result_3, indent=4))


# =========================
# SAVE RESULTS
# =========================

results = {
    "test_case_1": result_1,
    "test_case_2": result_2,
    "test_case_3": result_3
}

with open("week10_vid2_results.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nResults saved to: week10_vid2_results.json")
print("Vid 2 complete.")