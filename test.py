import os

# Directory containing the images
directory = "dataset/NeuPAT/audio/animation"  # Replace with the path to your images

# Iterate over files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        # Extract the base name without extension and leading zeros
        base = int(filename.split(".")[0])

        # Create a new filename with leading zeros
        new_filename = f"{base:03}.png"

        # Paths for old and new filenames
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(old_file, new_file)

print("Renaming completed.")
