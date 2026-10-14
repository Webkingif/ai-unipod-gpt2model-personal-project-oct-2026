import re

# Read the file
with open("output.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Replace 2 or more consecutive newlines with a single newline
cleaned_text = re.sub(r"\n\s*\n+", "\n", text)

# Save to a new file
with open("output.txt", "w", encoding="utf-8") as file:
    file.write(cleaned_text)

print("Extra blank lines reduced to one!")