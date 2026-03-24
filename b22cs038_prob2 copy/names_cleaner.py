file_path = "Indian_names.txt"

# Read the file first
with open(file_path, "r") as f:
    names = f.readlines()

# Remove duplicates while keeping order
seen = set()
unique_names = []

for name in names:
    name = name.strip()
    if name and name not in seen:
        seen.add(name)
        unique_names.append(name)

# Overwrite the file
with open(file_path, "w") as f:   # "w" clears the file
    for name in unique_names:
        f.write(name + "\n")

print("File cleared and rewritten with distinct names.")