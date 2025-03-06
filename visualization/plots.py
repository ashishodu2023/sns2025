import matplotlib.pyplot as plt
import seaborn as sns

def plot_and_save_anomalies(df, threshold, dist_filename="dist_plot.png", time_filename="time_plot.png"):
    """
    Plots and saves:
      1) Distribution (histogram) of Reconstruction_Error with threshold line.
      2) Time-series of Reconstruction_Error vs. Timestamp, highlighting anomalies.
    Assumes df has columns: 'Timestamp', 'Reconstruction_Error', 'Anomaly'.
    """

    # 1) Distribution Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="Reconstruction_Error", bins=30, kde=True, color="blue")
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold}")
    plt.title("Distribution of Reconstruction Errors")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    # Save distribution plot as PNG
    plt.savefig(dist_filename, dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    # 2) Time-Series Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df["Timestamp"], df["Reconstruction_Error"], label="Reconstruction Error", color="blue")
    
    # Highlight anomalies in red
    anomalies = df[df["Anomaly"] == True]
    plt.scatter(
        anomalies["Timestamp"],
        anomalies["Reconstruction_Error"],
        color="red",
        label="Anomaly",
        s=50
    )

    # Optional threshold line
    plt.axhline(threshold, color="green", linestyle="--", label="Threshold")
    
    plt.title("Reconstruction Error Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.grid(True)
    # Save time-series plot as PNG
    plt.savefig(time_filename, dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory


