import os
import shutil
import time
import logging

logging.basicConfig(
    filename="file_organizer.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

source_folder = "practice_organizer"

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

def organize_files():

    summary = {
        "files_found": 0,
        "moved": 0,
        "skipped": 0,
        "errors": 0
    }

    start_time = time.time()

    try:

        if not os.path.exists(source_folder):
            logging.error("Folder does not exist.")
            return False

        for filename in os.listdir(source_folder):

            file_path = os.path.join(source_folder, filename)

            # Skip folders
            if os.path.isdir(file_path):
                summary["skipped"] += 1
                logging.info(f"Skipped folder: {filename}")
                continue

            # Skip hidden files
            if filename.startswith("."):
                summary["skipped"] += 1
                logging.info(f"Skipped hidden file: {filename}")
                continue

            summary["files_found"] += 1

            try:

                extension = os.path.splitext(filename)[1].lower()

                modified_time = os.path.getmtime(file_path)
                seven_days = 7 * 24 * 60 * 60

                if time.time() - modified_time <= seven_days:
                    destination = "Recent"
                else:
                    destination = folders.get(extension, "Others")

                destination_folder = os.path.join(source_folder, destination)
                os.makedirs(destination_folder, exist_ok=True)

                new_path = os.path.join(destination_folder, filename)

                counter = 1
                while os.path.exists(new_path):
                    name, ext = os.path.splitext(filename)
                    new_path = os.path.join(
                        destination_folder,
                        f"{name}_{counter}{ext}"
                    )
                    counter += 1

                shutil.move(file_path, new_path)

                summary["moved"] += 1
                logging.info(f"Moved: {filename} -> {destination}")

            except PermissionError:
                summary["errors"] += 1
                logging.error(f"Permission denied: {filename}")

            except Exception as e:
                summary["errors"] += 1
                logging.error(f"Error processing {filename}: {e}")

        duration = round(time.time() - start_time, 2)

        print("\nSummary")
        print(f"Files Found: {summary['files_found']}")
        print(f"Moved: {summary['moved']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Errors: {summary['errors']}")
        print(f"Duration: {duration} seconds")

        return True

    except Exception as e:
        logging.error(f"Program failed: {e}")
        return False


status = organize_files()

print(f"\nSuccess: {status}")