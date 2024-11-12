import sys
import csv
import re

def process_ncu_csv(input_file, output_file):
    # Read the CSV data
    metrics = {}
    data_started = False
    metric_names = []
    rows = []

    with open(input_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not data_started:
                # Look for the line that starts with "ID,"
                if row and row[0].strip() == "ID":
                    data_started = True
                    metric_names = row[1:]  # Exclude the "ID" column
            else:
                # Collect metric values
                if row and re.match(r'^\d+$', row[0]):  # Check if the ID is a number
                    rows.append([float(value) if value else 0.0 for value in row[1:]])

    if not rows:
        print(f"No data found in {input_file}")
        return

    # Transpose the data to compute averages per metric
    columns = list(zip(*rows))
    averages = [sum(column) / len(column) for column in columns]

    # Write the averages to the output file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(metric_names)
        writer.writerow(averages)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python process_ncu_csv.py <input_csv> <output_csv>")
        sys.exit(1)

    process_ncu_csv(sys.argv[1], sys.argv[2])
