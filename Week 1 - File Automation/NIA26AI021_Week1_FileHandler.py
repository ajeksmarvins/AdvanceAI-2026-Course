import os
import shutil
import time

# Folder to organize
source_folder = "practice_organizer"

# File categories
folders = {
    ".pdf": "PDFs",
    ".jpg": "Images",
    ".jpeg": "Images",
    ".png": "Images",
    ".xlsx": "Spreadsheets",
    ".docx": "Documents",
    ".txt": "Text Files",
    ".csv": "Spreadsheets",
}

summary = {}

# Scan through all files
for filename in os.listdir(source_folder):

    file_path = os.path.join(source_folder, filename)

    # Skip folders
    if os.path.isdir(file_path):
        continue

    # Get file extension
    extension = os.path.splitext(filename)[1].lower()

    # Check if file was modified in the last 7 days
    modified_time = os.path.getmtime(file_path)
    seven_days = 7 * 24 * 60 * 60

    if time.time() - modified_time <= seven_days:
        destination = "Recent"
    else:
        destination = folders.get(extension, "Others")

    destination_folder = os.path.join(source_folder, destination)

    # Create folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    new_path = os.path.join(destination_folder, filename)

    # Rename if file already exists
    counter = 1
    while os.path.exists(new_path):
        name, ext = os.path.splitext(filename)
        new_path = os.path.join(
            destination_folder,
            f"{name}_{counter}{ext}"
        )
        counter += 1

    # Move the file
    shutil.move(file_path, new_path)

    # Count files
    summary[destination] = summary.get(destination, 0) + 1

print("\nSummary")
for folder, count in summary.items():
    print(f"{folder}: {count} files")