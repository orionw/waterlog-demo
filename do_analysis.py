import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import random
import logging

def setup_plotting():
    """Configure plot styling."""
    sns.set_theme(style="whitegrid")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (12, 6)
    logging.info("Plot styling configured")

def load_entropy_data(file_path):
    """Load the entropy analysis CSV file."""
    logging.info(f"Loading entropy data from {file_path}")
    df = pd.read_csv(file_path)
    logging.info(f"Loaded {len(df)} records")
    return df

def calculate_low_entropy_token_percentage(df, threshold=1.0):
    """Calculate percentage of tokens with entropy below threshold."""
    low_entropy_count = (df['entropy'] < threshold).sum()
    total_tokens = len(df)
    percentage = (low_entropy_count / total_tokens) * 100
    logging.info(f"Percentage of tokens with entropy < {threshold}: {percentage:.2f}%")
    return percentage

def calculate_instance_entropy_stats(df):
    """Calculate percentage of instances with average entropy < sequence length."""
    # Group by conversation to get instances
    instance_stats = df.groupby('conversation_id').agg({
        'entropy': ['sum', 'count']
    })
    
    # Flatten column names
    instance_stats.columns = ['sum_entropy', 'sequence_length']
    
    # Calculate instances where average entropy is less than sequence length
    low_entropy_instances = (instance_stats['sum_entropy'] < instance_stats['sequence_length']).sum()
    total_instances = len(instance_stats)
    
    percentage = (low_entropy_instances / total_instances) * 100
    logging.info(f"Percentage of instances with avg entropy < seq length: {percentage:.2f}%")
    return percentage

def calculate_sliding_window_stats(df, window_sizes=[3, 5, 7, 10]):
    """Calculate sliding window entropy statistics."""
    results = {}
    
    for window_size in window_sizes:
        # Calculate rolling means for each instance
        window_means = df.groupby('conversation_id')['entropy'].rolling(
            window=window_size, min_periods=window_size
        ).mean()
        
        # Calculate percentage of windows with average entropy < 1
        low_entropy_windows = (window_means < 1).sum()
        total_windows = len(window_means.dropna())
        
        # Store results
        results[window_size] = {
            'percent_low_entropy': (low_entropy_windows / total_windows) * 100 if total_windows > 0 else 0,
            'average_entropy': window_means.mean()
        }
        
        logging.info(f"Window size {window_size}: {results[window_size]['percent_low_entropy']:.2f}% low entropy")
    
    return results

def save_token_positions(instance_data, output_path):
    """Save token positions to a CSV file."""
    # Create a simplified DataFrame with position, token and entropy
    position_data = instance_data[['position', 'token', 'entropy']].copy()
    # Save to CSV
    position_data.to_csv(output_path, index=False)
    logging.info(f"Token positions saved to {output_path}")

def plot_cumulative_distribution(values, title, output_path):
    """Create a cumulative distribution plot of entropy values."""
    plt.figure(figsize=(15, 8))
    
    # Sort values and calculate cumulative percentages
    sorted_values = np.sort(values)
    percentiles = np.arange(len(sorted_values)) / float(len(sorted_values) - 1)
    
    # Create the cumulative plot
    plt.plot(sorted_values, percentiles, linewidth=2)
    plt.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='1-bit Threshold')
    
    plt.title(f'Cumulative Distribution - {title}')
    plt.xlabel('Entropy (bits)')
    plt.ylabel('Cumulative Proportion')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"Cumulative distribution plot saved to {output_path}")

def plot_instance_entropy(df, conv_id, output_dir, window_sizes=[3, 5, 7, 10, 15, 20]):
    """Plot entropy and sliding windows for a specific conversation."""
    # Filter data for this conversation
    instance_data = df[df['conversation_id'] == conv_id].copy()
    
    if len(instance_data) == 0:
        logging.warning(f"No data found for conversation {conv_id}")
        return
    
    # Drop any NaN values
    instance_data = instance_data.dropna(subset=['entropy'])
    if len(instance_data) == 0:
        logging.warning(f"No valid entropy data for conversation {conv_id}")
        return
        
    # Create base instance directory
    instance_dir = output_dir / f'conv{conv_id}'
    instance_dir.mkdir(exist_ok=True)
    
    # Save token positions to CSV
    token_file_path = instance_dir / 'token_positions.txt'
    save_token_positions(instance_data, token_file_path)
    
    # Plot raw entropy
    plt.figure(figsize=(15, 8))
    sns.lineplot(
        data=instance_data,
        x='position',
        y='entropy',
        label='Token Entropy',
        alpha=0.8,
        linewidth=2
    )
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1-bit Threshold')
    plt.title(f'Raw Entropy - Conversation {conv_id}')
    plt.xlabel('Token Position')
    plt.ylabel('Entropy (bits)')
    plt.legend()
    plt.tight_layout()
    plot_path = instance_dir / f'raw_entropy.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create cumulative distribution plot for raw entropy
    plot_cumulative_distribution(
        instance_data['entropy'].values,
        f'Raw Entropy - Conversation {conv_id}',
        instance_dir / 'raw_entropy_cumulative.png'
    )
    
    # Plot each window size separately
    for window_size in window_sizes:
        plt.figure(figsize=(15, 8))
        
        window_means = instance_data['entropy'].rolling(
            window=window_size, min_periods=window_size
        ).mean()
        
        # Plot window average
        plt.plot(
            instance_data['position'],
            window_means,
            label=f'{window_size}-token Window Average',
            alpha=0.8,
            linewidth=2
        )
        
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1-bit Threshold')
        plt.title(f'{window_size}-Token Window Average - Conversation {conv_id}')
        plt.xlabel('Token Position')
        plt.ylabel('Entropy (bits)')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_path = instance_dir / f'window_{window_size}.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create cumulative distribution plot for this window size
        plot_cumulative_distribution(
            window_means.dropna().values,
            f'{window_size}-Token Window Average - Conversation {conv_id}',
            instance_dir / f'window_{window_size}_cumulative.png'
        )

def select_random_instances(df, n=5, seed=42):
    """Select n random conversations from the dataset."""
    # Get unique conversation IDs
    conversation_ids = df['conversation_id'].unique()
    
    # Randomly select n conversations
    random.seed(seed)
    return random.sample(list(conversation_ids), min(n, len(conversation_ids)))

def create_prompt_wordcloud(df, output_dir):
    """Create a wordcloud from the prompts."""
    try:
        from wordcloud import WordCloud
        import nltk
        from nltk.corpus import stopwords
        
        # Ensure NLTK resources are available
        nltk.download('stopwords', quiet=True)
        
        # Get all unique prompts
        prompts = df['prompt'].dropna().unique()
        
        if len(prompts) == 0:
            logging.warning("No prompts found for wordcloud")
            return
            
        # Combine all prompts into one text
        all_text = ' '.join(prompts)
        
        # Create wordcloud
        stop_words = set(stopwords.words('english'))
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            colormap='viridis',
            stopwords=stop_words,
            max_words=200
        ).generate(all_text)
        
        # Plot wordcloud
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / 'prompt_wordcloud.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logging.info(f"Prompt wordcloud saved to {plot_path}")
    except ImportError:
        logging.warning("WordCloud or NLTK not installed. Skipping wordcloud creation.")

def plot_entropy_by_position(df, output_dir, max_positions=100):
    """Plot average entropy by token position."""
    # Group by position and calculate statistics
    position_stats = df.groupby('position').agg({
        'entropy': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten columns
    position_stats.columns = ['position', 'mean_entropy', 'std_entropy', 'count']
    
    # Filter to positions with enough data
    position_stats = position_stats[position_stats['count'] >= 50]
    
    # Limit to max_positions for clarity
    position_stats = position_stats[position_stats['position'] < max_positions]
    
    # Plot
    plt.figure(figsize=(15, 8))
    plt.errorbar(
        position_stats['position'],
        position_stats['mean_entropy'],
        yerr=position_stats['std_entropy'],
        alpha=0.6,
        capsize=3
    )
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1-bit Threshold')
    plt.title('Average Token Entropy by Position')
    plt.xlabel('Token Position')
    plt.ylabel('Entropy (bits)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'entropy_by_position.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info(f"Entropy by position plot saved to {plot_path}")

def plot_entropy_distribution(df, output_dir):
    """Plot overall distribution of entropy values."""
    plt.figure(figsize=(15, 8))
    
    # Plot histogram and kernel density estimate
    sns.histplot(df['entropy'], kde=True, bins=50)
    
    plt.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='1-bit Threshold')
    plt.title('Distribution of Token Entropy Values')
    plt.xlabel('Entropy (bits)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'entropy_distribution.png'
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info(f"Entropy distribution plot saved to {plot_path}")

def generate_summary_report(df, stats, output_dir):
    """Generate a summary report with main findings."""
    report_path = output_dir / 'entropy_analysis_summary.md'
    
    with open(report_path, 'w') as f:
        f.write("# Entropy Analysis Summary Report\n\n")
        
        f.write("## Dataset Overview\n")
        f.write(f"- Total conversations analyzed: {df['conversation_id'].nunique()}\n")
        f.write(f"- Total tokens analyzed: {len(df)}\n")
        f.write(f"- Average tokens per conversation: {len(df) / df['conversation_id'].nunique():.2f}\n\n")
        
        f.write("## Key Findings\n")
        f.write(f"- Overall average entropy: {df['entropy'].mean():.2f} bits\n")
        f.write(f"- Percentage of tokens with < 1 bit entropy: {stats['low_entropy_percent']:.2f}%\n")
        f.write(f"- Percentage of conversations with avg entropy < seq length: {stats['instance_entropy_percent']:.2f}%\n\n")
        
        f.write("## Sliding Window Analysis\n")
        for window_size, window_stats in stats['window_stats'].items():
            f.write(f"### Window Size: {window_size} tokens\n")
            f.write(f"- Percentage of windows with avg < 1 bit: {window_stats['percent_low_entropy']:.2f}%\n")
            f.write(f"- Average entropy across all windows: {window_stats['average_entropy']:.2f} bits\n\n")
        
        f.write("## Top 10 Lowest Entropy Tokens\n")
        low_entropy_tokens = df.nsmallest(10, 'entropy')[['token', 'entropy', 'conversation_id', 'position']]
        f.write("| Token | Entropy | Conversation ID | Position |\n")
        f.write("|-------|---------|-----------------|----------|\n")
        for _, row in low_entropy_tokens.iterrows():
            f.write(f"| {row['token']} | {row['entropy']:.4f} | {row['conversation_id']} | {row['position']} |\n")
        
        f.write("\n## Top 10 Highest Entropy Tokens\n")
        high_entropy_tokens = df.nlargest(10, 'entropy')[['token', 'entropy', 'conversation_id', 'position']]
        f.write("| Token | Entropy | Conversation ID | Position |\n")
        f.write("|-------|---------|-----------------|----------|\n")
        for _, row in high_entropy_tokens.iterrows():
            f.write(f"| {row['token']} | {row['entropy']:.4f} | {row['conversation_id']} | {row['position']} |\n")
        
    logging.info(f"Summary report generated: {report_path}")
    return report_path

def main(seed=42):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Setup directories
    input_file = Path('outputs/wildchat_generation_entropy.csv')
    output_dir = Path('plots/entropy_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting entropy analysis. Output directory: {output_dir}")
    
    # Setup plotting
    setup_plotting()
    
    # Load data
    df = load_entropy_data(input_file)
    
    # Calculate statistics
    low_entropy_percent = calculate_low_entropy_token_percentage(df)
    instance_entropy_percent = calculate_instance_entropy_stats(df)
    window_stats = calculate_sliding_window_stats(df)
    
    # Create overall plots
    plot_entropy_distribution(df, output_dir)
    plot_entropy_by_position(df, output_dir)
    
    # Try to create prompt wordcloud if data available
    if 'prompt' in df.columns:
        create_prompt_wordcloud(df, output_dir)
    
    # Select and plot random instances
    random_instances = select_random_instances(df, n=5, seed=seed)
    for conv_id in random_instances:
        plot_instance_entropy(df, conv_id, output_dir)
    
    # Generate summary stats for report
    stats = {
        'low_entropy_percent': low_entropy_percent,
        'instance_entropy_percent': instance_entropy_percent,
        'window_stats': window_stats
    }
    
    # Generate summary report
    report_path = generate_summary_report(df, stats, output_dir)
    
    # Print results to console
    print("\nEntropy Analysis Results")
    print("=======================")
    print(f"\nPercentage of tokens with < 1 bit entropy: {low_entropy_percent:.2f}%")
    print(f"\nPercentage of instances with average entropy < sequence length: {instance_entropy_percent:.2f}%")
    
    print("\nSliding Window Analysis")
    print("----------------------")
    for window_size, stats in window_stats.items():
        print(f"\nWindow size {window_size}:")
        print(f"  Percentage of windows with average < 1 bit: {stats['percent_low_entropy']:.2f}%")
        print(f"  Average entropy across all windows: {stats['average_entropy']:.2f}")
    
    print(f"\nPlots and report have been saved to {output_dir}")
    print(f"Summary report: {report_path}")

if __name__ == "__main__":
    main(seed=42)