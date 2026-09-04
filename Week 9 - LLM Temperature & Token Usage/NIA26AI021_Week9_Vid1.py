import os
import csv

from dotenv import load_dotenv
from groq import Groq


# Load API key
load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


# Prompt and temperatures
prompt = "Write a 2-sentence product description for wireless headphones."

temperatures = [0.0, 0.3, 0.5, 0.7, 1.0]

temperature_results = []
token_results = []


# ============================================================
# TEMPERATURE COMPARISON
# ============================================================

print("API client ready")

print("\n" + "=" * 60)
print("TEMPERATURE COMPARISON")
print("=" * 60)


for temperature in temperatures:

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        max_tokens=50
    )

    message = response.choices[0].message
    usage = response.usage

    temperature_results.append({
        "temperature": temperature,
        "response": message.content
    })

    token_results.append({
        "call_type": "temperature_comparison",
        "temperature": temperature,
        "repeat": "",
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    })

    print(f"\nTemperature: {temperature}")
    print("Response:")
    print(message.content)

    print("\nToken Usage:")
    print("Prompt tokens:", usage.prompt_tokens)
    print("Completion tokens:", usage.completion_tokens)
    print("Total tokens:", usage.total_tokens)


# ============================================================
# ZERO TEMPERATURE REPEAT TEST
# ============================================================

print("\n" + "=" * 60)
print("ZERO TEMPERATURE REPEAT TEST")
print("=" * 60)

zero_temperature_results = []


for repeat in range(1, 4):

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
        max_tokens=50
    )

    message = response.choices[0].message
    usage = response.usage

    zero_temperature_results.append(message.content)

    token_results.append({
        "call_type": "zero_temperature_repeat",
        "temperature": 0.0,
        "repeat": repeat,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    })

    print(f"\nRepeat {repeat}:")
    print(message.content)

    print("\nToken Usage:")
    print("Prompt tokens:", usage.prompt_tokens)
    print("Completion tokens:", usage.completion_tokens)
    print("Total tokens:", usage.total_tokens)


# ============================================================
# OUTPUT COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("OUTPUT COMPARISON")
print("=" * 60)

for result in temperature_results:
    print(f"\nTemperature {result['temperature']}:")
    print(result["response"])


print("\nComparison:")

for i in range(1, len(temperature_results)):

    previous = temperature_results[i - 1]["response"]
    current = temperature_results[i]["response"]

    if previous == current:
        print(
            f"{temperature_results[i]['temperature']} "
            "-> Same as previous response"
        )
    else:
        print(
            f"{temperature_results[i]['temperature']} "
            "-> Different from previous response"
        )


# ============================================================
# ZERO TEMPERATURE CONSISTENCY
# ============================================================

print("\n" + "=" * 60)
print("ZERO TEMPERATURE CONSISTENCY")
print("=" * 60)

if (
    zero_temperature_results[0]
    == zero_temperature_results[1]
    == zero_temperature_results[2]
):
    print("All three responses are exactly the same.")
else:
    print("The three responses are different.")


# ============================================================
# TOKEN USAGE
# ============================================================

first_five_tokens = sum(
    result["total_tokens"]
    for result in token_results
    if result["call_type"] == "temperature_comparison"
)

zero_temperature_tokens = sum(
    result["total_tokens"]
    for result in token_results
    if result["call_type"] == "zero_temperature_repeat"
)

total_tokens = first_five_tokens + zero_temperature_tokens

print("\n" + "=" * 60)
print("TOKEN USAGE")
print("=" * 60)

print("First 5 calls:", first_five_tokens)
print("Three 0.0 calls:", zero_temperature_tokens)
print("Total:", total_tokens)


# ============================================================
# SAVE TEMPERATURE COMPARISON CSV
# ============================================================

with open(
    "temperature_comparison.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "temperature",
            "response"
        ]
    )

    writer.writeheader()
    writer.writerows(temperature_results)


# ============================================================
# SAVE TOKEN USAGE CSV
# ============================================================

with open(
    "token_usage.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "call_type",
            "temperature",
            "repeat",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens"
        ]
    )

    writer.writeheader()
    writer.writerows(token_results)