import pandas as pd
import os
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
results_album = os.path.join(project_root, 'assets', 'results_album.xlsx')
results_artist = os.path.join(project_root, 'assets', 'results_artist.xlsx')

def save_metrics(type, model, accuracy, precision, recall, f1, test_size, neighbours):
    file_path = '';
    if type == 'album':
        file_path = results_album
    elif type == 'artist':
        file_path = results_artist

    # Create DataFrame for the current results
    df_results = pd.DataFrame({
        'Model': [model.upper()],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1],
        'Test Size': [test_size],
        'Neighbours': [neighbours]
    })

    # If the file already exists, append the results, else create a new file
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        df = pd.concat([df, df_results], ignore_index=True)
    else:
        df = df_results

    # Save the DataFrame to Excel
    df.to_excel(file_path, index=False, float_format='%.6f')

def generate_scatter_plot(df, x_col, y_col, title, file):
    plt.scatter(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True)
    plt.savefig(file)

def generate_bar_chart(df, x_col, y_col, title, file):
    df.plot(x=x_col, y=y_col, kind='bar', legend=False)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.savefig(file)

def generate_heatmap(df, columns, title, file):
    corr_matrix = df[columns].corr()
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(columns)), columns, rotation=45)
    plt.yticks(range(len(columns)), columns)
    plt.title(title)
    plt.savefig(file)

def generate_line_graph(df, x_col, y_col, title, file):
    df.plot(x=x_col, y=y_col, marker='o')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True)
    plt.savefig(file)

def generate_charts(type):
    file_path = '';
    if type == 'album':
        file_path = results_album
    elif type == 'artist':
        file_path = results_artist

    df = pd.read_excel(file_path)

    plt.figure(figsize=(10, 6))
    generate_scatter_plot(df, 'Neighbours', 'Accuracy', 'Accuracy vs. Neighbours', 'accuracy_vs_neighbours.png')
    generate_bar_chart(df, 'Model', 'F1-score', 'Model-wise Accuracy', 'model_wise_f1.png')
    generate_heatmap(df, ['Accuracy', 'Precision', 'Recall', 'F1-score'], 'Performance Metrics Correlation', 'performance_metrics_correlation.png')
    generate_line_graph(df, 'Test Size', 'Accuracy', 'Accuracy vs. Test Size', 'accuracy_vs_test_size.png')

generate_charts('album')