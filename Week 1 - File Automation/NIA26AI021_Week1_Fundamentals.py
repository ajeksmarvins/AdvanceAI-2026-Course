# List of sample files
files = [
    "report.pdf",
    "invoice.pdf",
    "book.pdf",
    "photo.jpg",
    "logo.jpg",
    "sales.xlsx",
    "budget.xlsx",
    "accounts.xlsx",
    "presentation.xlsx",
    "archive.zip"
]
# Dictionary to map file extensions to folder names
folders = {
    ".pdf": "PDFs",
    ".jpg": "Images",
    ".xlsx": "Spreadsheets",
    ".docx": "Documents",
    ".txt": "Text Files"
}
# Categorize each file
for file in files:
    extension = "." + file.split(".")[-1]

    if extension in folders:
        print(f"{file} → {folders[extension]}")
    else:
        print(f"{file} → Others")
        # Count each file type
counts = {
    "PDFs": 0,
    "Images": 0,
    "Spreadsheets": 0,
    "Documents": 0,
    "Text Files": 0,
    "Others": 0
}

for file in files:
    extension = "." + file.split(".")[-1]

    if extension in folders:
        counts[folders[extension]] += 1
    else:
        counts["Others"] += 1

print("\nSummary")
print(f"Found {counts['PDFs']} PDFs")
print(f"Found {counts['Images']} Images")
print(f"Found {counts['Spreadsheets']} Spreadsheets")
print(f"Found {counts['Documents']} Documents")
print(f"Found {counts['Text Files']} Text Files")
print(f"Found {counts['Others']} Others")
print("\nLarge Files")

file_sizes = {
    "report.pdf": 25,
    "invoice.pdf": 62,
    "book.pdf": 18,
    "photo.jpg": 12,
    "logo.jpg": 55,
    "sales.xlsx": 48,
    "budget.xlsx": 75,
    "accounts.xlsx": 35,
    "presentation.xlsx": 30,
    "archive.zip": 60
}

for file in files:
    if file_sizes[file] > 50:
        print(f"{file} ({file_sizes[file]} MB) - Large - needs compression")