import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def visualize_grouping_idx_percentage(df):
    """
    Visualize the percentage of positive vs. negative scores for each grouping_idx
    as horizontal bar charts using an orange and blue color scheme.

    :param df: A Pandas DataFrame with columns:
               'struct_name', 'grouping_idx', 'score'
    """
    # Calculate positive and negative percentages by grouping_idx
    grouping_data = df.groupby('grouping_idx').apply(
        lambda x: {
            'positive': (x['score'] > 0).mean() * 100,  # Percentage positive
            'negative': (x['score'] <= 0).mean() * 100  # Percentage negative
        }
    ).reset_index()

    # Convert to a proper DataFrame for plotting
    grouping_data = pd.DataFrame(grouping_data.to_dict('records'))
    grouping_data.columns = ['grouping_idx', 'percentages']
    grouping_data['positive'] = grouping_data['percentages'].apply(lambda x: x['positive'])
    grouping_data['negative'] = grouping_data['percentages'].apply(lambda x: x['negative'])

    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    y_positions = np.arange(len(grouping_data))  # One bar per grouping_idx
    bar_height = 0.4  # Height of bars

    # Positive and negative bars
    ax.barh(y_positions, grouping_data['positive'], bar_height, label='Positive %', color='blue')
    ax.barh(y_positions, -grouping_data['negative'], bar_height, label='Negative %', color='orange')

    # Label and style the chart
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Grouping Idx {idx}" for idx in grouping_data['grouping_idx']])
    ax.set_xlabel('Percentage (%)')
    ax.set_ylabel('Grouping Index')
    ax.set_title('Percentage of Positive vs. Negative Scores by Grouping Index')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')  # Divider for positive/negative
    ax.legend(loc='best')
    plt.tight_layout()

    # Save the figure
    plt.savefig('grouping_idx_percentages.png')
    print("Figure saved to grouping_idx_percentages.png")

# Example usage with your dataframe
data = {
    'struct_name': ['struct.Hello', 'struct.Hello', 'struct.World', 'struct.World', 'struct.World'],
    'grouping_idx': [0, 1, 0, 1, 2],
    'score': [0.049, -0.030, 0.070, -0.020, 0.0]
}
df = pd.DataFrame(data)
visualize_grouping_idx_percentage(df)

