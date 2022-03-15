import csv


with open("data.csv") as file:
    lines = file.readlines()

rows = [
    [
        "Area",
        "Perimeter",
        "Major_Axis_Length",
        "Minor_Axis_Length",
        "Solidity",
        "Roundness",
    ]
]
ROW_COUNT = 30

for i in range(0, min(len(lines), ROW_COUNT * 6), 6):
    rows.append(list(map(float, lines[i : i + 6])))

with open("data_packed.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(rows)
