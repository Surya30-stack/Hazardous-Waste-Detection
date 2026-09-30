from pathlib import Path

image_path = Path(
    "Datasets/test/images/Screenshot-2025-02-16-221353_png_png.rf.c58c8ce6b04434427fbf3f1861a9bf87.jpg"
)

label_path = Path(
    "Datasets/test/labels/"
    "Screenshot-2025-02-16-221353_png_png.rf.c58c8ce6b04434427fbf3f1861a9bf87.txt"
)

print("Image exists:", image_path.exists())
print("Label exists:", label_path.exists())

print("\nLabel path:")
print(label_path)

if label_path.exists():

    print("\n--- RAW LABEL CONTENT ---")

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        print(repr(line.strip()))

    print("\n--- PARSED VALUES ---")

    for line in lines:

        values = line.strip().split()

        print("Values:", values)
        print("Number of values:", len(values))

else:
    print("\nLABEL FILE NOT FOUND")