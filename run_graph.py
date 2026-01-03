import matplotlib.pyplot as plt
import numpy as np

def create_data(x_offset=0, y_offset=0, amplitude=2.0, frequency=1):
    """Generate x values and compute sin(x) with optional modifications."""
    x = np.linspace(0, 2 * np.pi, 100) + x_offset
    y = amplitude * np.sin(frequency * x) + y_offset
    return x, y

def create_plot(x, y, title="Sine Wave"):
    """Create the plot with labels."""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'r-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title(title)
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)

if __name__ == "__main__":
    x, y = create_data(amplitude=2.0)
    create_plot(x, y)
    plt.savefig('/home/thorin/truthspace-lcm/output/sine_wave.png', dpi=150)
    print("Saved to /home/thorin/truthspace-lcm/output/sine_wave.png")
