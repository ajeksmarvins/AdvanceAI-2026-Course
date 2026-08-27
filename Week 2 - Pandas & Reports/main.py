import subprocess

print("=" * 50)
print("WEEK 2 AUTOMATED REPORT PIPELINE")
print("=" * 50)

print("\nStep 1: Data Cleaning...")
subprocess.run(["python", "NIA26AI021_Week2_DataCleaning.py"])

print("\nStep 2: Report Generation...")
subprocess.run(["python", "NIA26AI021_Week2_ReportGenerator.py"])

print("\nStep 3: Email Delivery...")
subprocess.run(["python", "NIA26AI021_Week2_EmailSender.py"])

print("\n🎉 All tasks completed successfully!")