import re

# Read the file
with open("output.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

# Regex to match lines starting with RECORD followed by numbers/text
pattern = r"^RECORD\s+\d+.*$"

# Keep lines that DO NOT match the record header pattern
cleaned_lines = [line for line in lines if not re.match(pattern, line.strip())]

# Write back to a new file
with open("output.txt", "w", encoding="utf-8") as file:
    file.writelines(cleaned_lines)

print("Record header lines removed successfully!")