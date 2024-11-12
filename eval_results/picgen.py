import matplotlib.pyplot as plt
import pandas as pd

def load_and_process_data(file_path):
    data = pd.read_csv(file_path)
    data['Metric Value'] = pd.to_numeric(data['Metric Value'], errors='coerce')
    return data

def filter_data(data, metrics):
    return data[data['Metric Name'].isin(metrics) & data['Metric Value'].notna()]

def plot_and_save_metric(data, metric):
    metric_data = data[data['Metric Name'] == metric]
    if not metric_data.empty:
        plt.figure(figsize=(10, 6))
        metric_summary = metric_data.groupby('Stage')['Metric Value'].mean()
        metric_summary.sort_index().plot(kind='bar', title=f'Mean {metric}')
        plt.ylabel('Average Value')
        plt.xticks(rotation=45)
        image_path = f'./{metric.replace("/", "_")}_bar_chart.png'
        plt.savefig(image_path)
        plt.close()  # Close the plot to free up memory

def main():
    file_path = 'combined_data.csv'  # Update this path to your CSV file location
    metrics_to_plot = [
        'L1/TEX Cache Throughput', 'L2 Cache Throughput', 'DRAM Throughput',
        'Compute (SM) Throughput', 'Memory Throughput'
    ]

    data = load_and_process_data(file_path)
    filtered_data = filter_data(data, metrics_to_plot)

    for metric in metrics_to_plot:
        plot_and_save_metric(filtered_data, metric)

if __name__ == '__main__':
    main()
