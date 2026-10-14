import re

# Read the file
with open("old corpus.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Regular expression to match URLs (http, https, www, etc.)
url_pattern = r"https?://\S+|www\.\S+"

# Replace URLs with an empty string (or space)
cleaned_text = re.sub(url_pattern, "", text)

# Save to a new file
with open("output.txt", "w", encoding="utf-8") as file:
    file.write(cleaned_text)

print("Links removed successfully!")
