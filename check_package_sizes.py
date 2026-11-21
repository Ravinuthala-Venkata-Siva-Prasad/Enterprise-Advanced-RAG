
import importlib.metadata as metadata

rows = []

for dist in metadata.distributions():
    files = dist.files or []
    total_size = 0

    for file in files:
        path = dist.locate_file(file)

        if path.is_file():
            total_size += path.stat().st_size

    rows.append((total_size / (1024 ** 3), dist.metadata["Name"]))

for size, name in sorted(rows, reverse=True)[:25]:
    print(f"{size:.3f} GB  {name}")
