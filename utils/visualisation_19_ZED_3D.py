import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

PATH = "data_19\19.csv"

def animate_xyz_coordinates(csv_file):
    """
    Visualize the x, y, and z coordinates from the CSV file as a 3D animation through all rows.
    
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
    
    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.1, bottom=0.2)  # More bottom space for widgets
    
    # Initialize empty line objects for the skeleton
    lines = []
    for _ in connections:
        line, = ax.plot([], [], [], 'gray', linewidth=2)
        lines.append(line)
    
    # Initialize empty scatter for the points
    scatter = ax.scatter([], [], [], s=50, c='blue', zorder=10)
    
    # Reference point (will be highlighted in red)
    ref_point = ax.scatter([], [], [], s=80, c='red', zorder=15)
    
    # Text labels for point indices
    point_labels = []
    for i in range(19):  # 19 points (0-18)
        label = ax.text(0, 0, 0, str(i), fontsize=10, ha='center', va='center', visible=False)
        point_labels.append(label)
    
    # Text for activity label and frame counter
    activity_text = fig.text(0.02, 0.95, '', fontsize=12)
    counter_text = fig.text(0.02, 0.9, '', fontsize=10)
    
    # Set fixed axis limits based on data range
    x_min, x_max = df[[col for col in df.columns if col.startswith('x')]].min().min(), df[[col for col in df.columns if col.startswith('x')]].max().max()
    y_min, y_max = df[[col for col in df.columns if col.startswith('y')]].min().min(), df[[col for col in df.columns if col.startswith('y')]].max().max()
    
    # Check if z coordinates exist in the DataFrame
    z_columns = [col for col in df.columns if col.startswith('z')]
    has_z_data = len(z_columns) > 0
    
    if has_z_data:
        z_min, z_max = df[z_columns].min().min(), df[z_columns].max().max()
    else:
        # If z coordinates don't exist, create synthetic z values
        print("Z coordinates not found in data. Using synthetic z values.")
        # We'll generate z values during animation
        z_min, z_max = -1, 1  # Default range for synthetic z values
    
    # Add some padding to the limits
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min if has_z_data else 2  # Default range for synthetic z
    
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
    
    # Set labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('3D Animation of XYZ Coordinates')
    ax.grid(True)
    
    # Add view angle sliders for 3D rotation control
    ax_elev = plt.axes([0.25, 0.05, 0.65, 0.03])
    ax_azim = plt.axes([0.25, 0.02, 0.65, 0.03])
    
    elev_slider = Slider(ax_elev, 'Elevation', -90, 90, valinit=30, valstep=5)
    azim_slider = Slider(ax_azim, 'Azimuth', -180, 180, valinit=-60, valstep=5)
    
    # Animation speed slider
    ax_speed = plt.axes([0.25, 0.08, 0.65, 0.03])
    speed_slider = Slider(ax_speed, 'Speed', 10, 500, valinit=100, valstep=10)
    
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
            
            # Get z coordinate if available, otherwise use synthetic z
            if has_z_data:
                z = row[f'z{j}']
            else:
                # Create synthetic z values - could be derived from x,y or just static
                # Here we use a simple function of x and y for demonstration
                z = np.sin(0.1 * x) * np.cos(0.1 * y) * 0.5
            
            points.append((x, y, z))
        
        points = np.array(points)
        
        # Printing norm
        print(f"norm {frame}: {np.linalg.norm(points)}")
        
        # Update the lines (skeleton connections)
        for i, (start, end) in enumerate(connections):
            lines[i].set_data([points[start, 0], points[end, 0]], [points[start, 1], points[end, 1]])
            lines[i].set_3d_properties([points[start, 2], points[end, 2]])
            lines[i].set_color(colors.get(activity, 'gray'))
        
        # Update the points
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        scatter.set_color(colors.get(activity, 'blue'))
        
        # Update the reference point (point 1)
        ref_point._offsets3d = ([points[1, 0]], [points[1, 1]], [points[1, 2]])
        
        # Update the point labels
        for i, (x, y, z) in enumerate(points):
            point_labels[i].set_position((x, y))
            point_labels[i].set_3d_properties(z, 'z')
            point_labels[i].set_visible(True)
        
        # Update activity label and frame counter
        activity_text.set_text(f'Activity: {activity}')
        counter_text.set_text(f'Frame: {frame+1}/{len(df)}')
        
        # Return all updated artists
        return lines + [scatter, ref_point] + point_labels
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(df), interval=interval, blit=False)
    
    # Functions to update the 3D view
    def update_elev(val):
        ax.view_init(elev=val, azim=azim_slider.val)
        fig.canvas.draw_idle()
    
    def update_azim(val):
        ax.view_init(elev=elev_slider.val, azim=val)
        fig.canvas.draw_idle()
    
    elev_slider.on_changed(update_elev)
    azim_slider.on_changed(update_azim)
    
    # Update interval when slider is changed
    def update_speed(val):
        nonlocal ani
        ani.event_source.stop()  # Stop the current animation
        ani = FuncAnimation(fig, update, frames=len(df), interval=int(1000/val), blit=False)
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
    
    # Set initial view angle
    ax.view_init(elev=30, azim=-60)
    
    plt.show()
    
    print("3D Animation complete!")


if __name__ == "__main__":
    # Path to your CSV file
    csv_file = PATH
    
    # Create and display the animation
    animate_xyz_coordinates(csv_file)