import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

PATH = r"data\19.csv"

def animate_xy_coordinates(csv_file):
    """
    Visualize the x and y coordinates from the CSV file as an animation through all rows.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from {csv_file}")
    
    # Create a color mapping for different activities
    colors = {'walking': 'blue', 'looking': 'green', 'cleaning': 'red', 'Collecting': 'purple'}
    
    # Define connections between points to form a skeleton
    connections = [
        (1, 2), (2, 3), (3, 18),  # Spine and head
        (2, 4), (4, 5), (5, 6), (6, 7),  # Right arm
        (2, 8), (8, 9), (9, 10), (10, 11),  # Left arm
        (0, 12), (12, 13), (13, 14),  # Right leg
        (0, 15), (15, 16), (16, 17),  # Left leg
        (0, 1)  # Lower spine
    ]
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(left=0.1, bottom=0.15)
    
    # Initialize empty line objects for the skeleton
    lines = []
    for _ in connections:
        line, = ax.plot([], [], 'gray', linewidth=2)
        lines.append(line)
    
    # Initialize empty scatter for the points
    scatter = ax.scatter([], [], s=50, c='blue', zorder=10)
    
    # Reference point (will be highlighted in red)
    ref_point = ax.scatter([], [], s=80, c='red', zorder=15)
    
    # Text labels for point indices
    point_labels = []
    for i in range(19):  # 19 points (0-18)
        label = ax.text(0, 0, str(i), fontsize=20, ha='center', va='center', visible=False)
        point_labels.append(label)
    
    # Text for activity label and frame counter
    activity_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    counter_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=10)
    
    # Set fixed axis limits based on data range
    x_min, x_max = df[[col for col in df.columns if col.startswith('x')]].min().min(), df[[col for col in df.columns if col.startswith('x')]].max().max()
    y_min, y_max = df[[col for col in df.columns if col.startswith('y')]].min().min(), df[[col for col in df.columns if col.startswith('y')]].max().max()
    
    # Add some padding to the limits
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    
    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title('Animation of XY Coordinates')
    ax.grid(True)
    
    # Animation speed slider
    from matplotlib.widgets import Slider
    
    # Add a slider for controlling animation speed
    ax_speed = plt.axes([0.25, 0.02, 0.65, 0.03])
    speed_slider = Slider(ax_speed, 'Speed', 10, 500, valinit=100, valstep=10)
    speed_slider.label.set_size(10)
    
    # Animation interval (ms) - Will be controlled by the slider
    interval = 100
    
    # Function to update the figure for each frame
    def update(frame):
        # Get the row data for this frame
        row = df.iloc[frame]
        activity = row['label']
        
        # Extract points for this frame
        points = []
        for j in range(19):  # 19 points (0-18)
            x = row[f'x{j}']
            y = row[f'y{j}']
            points.append((x, y))
        
        points = np.array(points)
        #printing norm
        print("norma " + str(frame) + ":" + str(np.linalg.norm(points)))
        
        # Update the lines (skeleton connections)
        for i, (start, end) in enumerate(connections):
            lines[i].set_data([points[start, 0], points[end, 0]], [points[start, 1], points[end, 1]])
            lines[i].set_color(colors.get(activity, 'gray'))
        
        # Update the points
        scatter.set_offsets(points)
        scatter.set_color(colors.get(activity, 'blue'))
        
        # Update the reference point (point 1)
        ref_point.set_offsets([points[1, 0], points[1, 1]])
        
        # Update the point labels
        for i, (x, y) in enumerate(points):
            point_labels[i].set_position((x, y))
            point_labels[i].set_visible(True)
        
        # Update activity label and frame counter
        activity_text.set_text(f'Activity: {activity}')
        counter_text.set_text(f'Frame: {frame+1}/{len(df)}')
        
        # Return all updated artists
        return lines + [scatter, ref_point] + point_labels + [activity_text, counter_text]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(df), interval=interval, blit=True)
    
    # Update interval when slider is changed
    def update_speed(val):
        nonlocal ani
        ani.event_source.stop()  # Stop the current animation
        ani = FuncAnimation(fig, update, frames=len(df), interval=int(1000/val), blit=True)
        fig.canvas.draw_idle()
    
    speed_slider.on_changed(update_speed)
    
    # Add play/pause functionality
    def on_key(event):
        if event.key == ' ':  # Spacebar
            if ani.event_source.is_running():
                ani.event_source.stop()
            else:
                ani.event_source.start()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Add a legend
    legend_elements = []
    for activity, color in colors.items():
        if activity in df['label'].unique():
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=4, label=activity))
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add instruction text
    plt.figtext(0.5, 0.01, "Press spacebar to play/pause", ha="center", fontsize=10)
    
    plt.show()
    
    print("Animation complete!")


if __name__ == "__main__":
    # Path to your CSV file
    csv_file = PATH
    
    # Create and display the animation
    animate_xy_coordinates(csv_file)