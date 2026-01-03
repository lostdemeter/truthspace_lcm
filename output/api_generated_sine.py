import matplotlib.pyplot as plt
import numpy as np

def create_data():
    """Generate x values and compute sin(x)."""
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    return x, y

def create_plot(x, y):
    """Create the plot with labels."""
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title('Sine Wave')
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)

if __name__ == "__main__":
    x, y = create_data()
    create_plot(x, y)
    plt.savefig('/home/thorin/truthspace-lcm/output/sine_wave.png', dpi=150)
    print("Saved to /home/thorin/truthspace-lcm/output/sine_wave.png")
