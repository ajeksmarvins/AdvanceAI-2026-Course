import os
import json
import base64
import pdfplumber
from dotenv import load_dotenv
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from PIL import Image, ImageDraw, ImageFont


# =========================================================
# SETUP
# =========================================================

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("API client ready")


# =========================================================
# CREATE DOCUMENT 2 - RECEIPT
# =========================================================

def create_receipt():

    doc = SimpleDocTemplate("sample_receipt.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("SALES RECEIPT", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph("BrightMart Electronics", styles["Heading2"])
    )
    elements.append(
        Paragraph("45 Market Road", styles["Normal"])
    )
    elements.append(
        Paragraph("Lagos, Nigeria", styles["Normal"])
    )

    elements.append(Spacer(1, 12))

    receipt_data = [
        ["Receipt Number", "RCP-2026-00451"],
        ["Date", "September 10, 2026"],
        ["Payment Method", "Card"],
        ["Customer", "Marvin Technologies"],
        ["Subtotal", "$225.00"],
        ["Tax", "$18.00"],
        ["Total Paid", "$243.00"]
    ]

    table = Table(receipt_data, colWidths=[160, 250])

    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey)
    ]))

    elements.append(table)

    doc.build(elements)

    print("Sample receipt created: sample_receipt.pdf")


# =========================================================
# CREATE DOCUMENT 3 - PURCHASE ORDER
# =========================================================

def create_purchase_order():

    doc = SimpleDocTemplate(
        "sample_purchase_order.pdf",
        pagesize=letter
    )

    styles = getSampleStyleSheet()
    elements = []

    elements.append(
        Paragraph("PURCHASE ORDER", styles["Title"])
    )

    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph("NovaTech Supplies", styles["Heading2"])
    )
    elements.append(
        Paragraph("88 Industrial Avenue", styles["Normal"])
    )
    elements.append(
        Paragraph("Abuja, Nigeria", styles["Normal"])
    )

    elements.append(Spacer(1, 12))

    order_data = [
        ["PO Number", "PO-2026-00887"],
        ["Order Date", "September 11, 2026"],
        ["Delivery Date", "September 20, 2026"],
        ["Payment Terms", "Net 15"],
        ["Supplier", "DataCore Systems Ltd."],
        ["Subtotal", "$1,130.00"]
    ]

    table = Table(order_data, colWidths=[160, 250])

    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey)
    ]))

    elements.append(table)

    doc.build(elements)

    print("Sample purchase order created: sample_purchase_order.pdf")


# =========================================================
# CREATE SCANNED-STYLE IMAGE FOR BONUS
# =========================================================

def create_invoice_image():

    img = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("arial.ttf", 42)
        heading_font = ImageFont.truetype("arial.ttf", 28)
        normal_font = ImageFont.truetype("arial.ttf", 22)
    except:
        title_font = None
        heading_font = None
        normal_font = None

    draw.text((350, 60), "INVOICE", fill="black", font=title_font)

    y = 150

    lines = [
        "TechCorp Solutions",
        "123 Business Park, Suite 100",
        "San Francisco, CA 94105",
        "",
        "Invoice Number: INV-2025-00123",
        "Invoice Date: March 15, 2025",
        "Due Date: April 14, 2025",
        "Payment Terms: Net 30",
        "",
        "Bill To:",
        "Acme Corporation",
        "456 Industrial Ave",
        "Chicago, IL 60601",
        "",
        "Laptop Pro 16-inch      2      $1,299.99      $2,599.98",
        "Wireless Mouse          5         $29.99         $149.95",
        "USB-C Docking Station   3        $199.99         $599.97",
        "Extended Warranty       2        $249.99         $499.98",
        "",
        "Subtotal: $3,849.88",
        "Tax (8.5%): $327.24",
        "TOTAL: $4,177.12"
    ]

    for line in lines:
        draw.text((80, y), line, fill="black", font=normal_font)
        y += 48

    img.save("invoice_image.png")

    print("Invoice image created: invoice_image.png")


# =========================================================
# CREATE THE DOCUMENTS
# =========================================================

create_receipt()
create_purchase_order()
create_invoice_image()


# =========================================================
# DOCUMENT LIST
# =========================================================

documents = [
    ("Invoice", "sample_invoice.pdf"),
    ("Receipt", "sample_receipt.pdf"),
    ("Purchase Order", "sample_purchase_order.pdf")
]


# =========================================================
# EXTRACT TEXT FROM PDF
# =========================================================

def extract_text(file_path):

    text = ""

    with pdfplumber.open(file_path) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text.strip()


# =========================================================
# EXTRACT STRUCTURED DATA WITH AI
# =========================================================

def extract_data(text):

    prompt = f"""
Extract information from this business document.

Return ONLY valid JSON with these fields:

document_type
document_number
date
issuer_name
customer_or_supplier
subtotal
tax
total
currency

Use null if a field is not available.

Do not invent information.

Document:
{text}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
        max_tokens=800,
        response_format={"type": "json_object"}
    )

    return json.loads(
        response.choices[0].message.content
    )


# =========================================================
# EXPECTED VALUES
# =========================================================

ground_truth = {

    "Invoice": {
        "document_type": "invoice",
        "document_number": "INV-2025-00123",
        "date": "March 15, 2025",
        "issuer_name": "TechCorp Solutions",
        "customer_or_supplier": "Acme Corporation",
        "subtotal": "$3,849.88",
        "tax": "$327.24",
        "total": "$4,177.12",
        "currency": "USD"
    },

    "Receipt": {
        "document_type": "sales receipt",
        "document_number": "RCP-2026-00451",
        "date": "September 10, 2026",
        "issuer_name": "BrightMart Electronics",
        "customer_or_supplier": "Marvin Technologies",
        "subtotal": "$225.00",
        "tax": "$18.00",
        "total": "$243.00",
        "currency": "USD"
    },

    "Purchase Order": {
        "document_type": "purchase order",
        "document_number": "PO-2026-00887",
        "date": "September 11, 2026",
        "issuer_name": "NovaTech Supplies",
        "customer_or_supplier": "DataCore Systems Ltd.",
        "subtotal": "$1,130.00",
        "tax": None,
        "total": None,
        "currency": "USD"
    }
}


# =========================================================
# NORMALIZE VALUES FOR COMPARISON
# =========================================================

def normalize(value):

    if value is None:
        return None

    value = str(value).strip().lower()
    value = value.replace("$", "")
    value = value.replace(",", "")

    try:
        return float(value)
    except ValueError:
        return value


# =========================================================
# PROCESS THREE DOCUMENTS
# =========================================================

all_results = []

for name, file_path in documents:

    print("\n" + "=" * 50)
    print(name.upper())
    print("=" * 50)

    text = extract_text(file_path)

    print("\nExtracted text:")
    print(text)

    data = extract_data(text)

    print("\nStructured JSON:")
    print(json.dumps(data, indent=4))

    all_results.append({
        "document": name,
        "data": data
    })


# =========================================================
# ACCURACY COMPARISON
# =========================================================

print("\n" + "=" * 50)
print("EXTRACTION ACCURACY")
print("=" * 50)

fields = [
    "document_type",
    "document_number",
    "date",
    "issuer_name",
    "customer_or_supplier",
    "subtotal",
    "tax",
    "total",
    "currency"
]

accuracy_results = []

for result in all_results:

    name = result["document"]
    extracted = result["data"]
    expected = ground_truth[name]

    correct = 0

    for field in fields:

        actual = normalize(extracted.get(field))
        expected_value = normalize(expected.get(field))

        if actual == expected_value:
            correct += 1

    accuracy = (correct / len(fields)) * 100

    print(
        f"{name}: "
        f"{correct}/{len(fields)} correct "
        f"({accuracy:.2f}%)"
    )

    accuracy_results.append({
        "document": name,
        "correct": correct,
        "total_fields": len(fields),
        "accuracy": round(accuracy, 2)
    })


# =========================================================
# BONUS - VISION EXTRACTION
# =========================================================

def encode_image(image_path):

    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode()


def vision_extract(image_path):

    image_data = encode_image(image_path)

    prompt = """
Extract the following information from this document image.

Return ONLY valid JSON with these fields:

document_type
document_number
date
issuer_name
customer_or_supplier
subtotal
tax
total
currency

Use null when a field is not visible.
"""

    response = client.chat.completions.create(
    model="qwen/qwen3.6-27b",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                }
            ]
        }
    ],
    temperature=0.0,
    max_completion_tokens=500,
    response_format={"type": "json_object"}
)

    return json.loads(
        response.choices[0].message.content
    )


print("\n" + "=" * 50)
print("BONUS - VISION EXTRACTION")
print("=" * 50)

vision_result = vision_extract("invoice_image.png")

print("\nVision model result:")
print(json.dumps(vision_result, indent=4))


# =========================================================
# SAVE RESULTS
# =========================================================

with open("week10_vid1_results.json", "w") as file:
    json.dump(all_results, file, indent=4)

with open("week10_vid1_accuracy.json", "w") as file:
    json.dump(accuracy_results, file, indent=4)

with open("week10_vid1_vision_result.json", "w") as file:
    json.dump(vision_result, file, indent=4)


print("\nResults saved.")
print("Vid 1 complete.")