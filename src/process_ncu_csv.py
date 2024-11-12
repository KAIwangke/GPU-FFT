import sys
import csv

def process_ncu_csv(input_file, output_file):
    # Read the CSV data
    metrics = {}
    with open(input_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        metric_names = reader.fieldnames

    # Initialize sums
    sums = {name: 0.0 for name in metric_names}
    count = len(rows)

    # Sum up the metrics
    for row in rows:
        for name in metric_names:
            try:
                sums[name] += float(row[name])
            except ValueError:
                pass  # Skip non-numeric fields

    # Compute averages
    averages = {name: sums[name]/count for name in metric_names}

    # Write the averages to the output file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metric_names)
        writer.writeheader()
        writer.writerow(averages)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python process_ncu_csv.py <input_csv> <output_csv>")
        sys.exit(1)

    process_ncu_csv(sys.argv[1], sys.argv[2])
