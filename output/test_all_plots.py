"""Generate all plot types and save them."""
import matplotlib.pyplot as plt
import numpy as np

# 1. Bar Chart
def bar_chart():
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color='steelblue')
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.title('Bar Chart')
    plt.savefig('/home/thorin/truthspace-lcm/output/bar_chart.png', dpi=150)
    plt.close()
    print("Saved bar_chart.png")

# 2. Scatter Plot
def scatter_plot():
    np.random.seed(42)
    x = np.random.randn(50)
    y = x + np.random.randn(50) * 0.5
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c='steelblue', alpha=0.7)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot')
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/thorin/truthspace-lcm/output/scatter_plot.png', dpi=150)
    plt.close()
    print("Saved scatter_plot.png")

# 3. Histogram
def histogram():
    np.random.seed(42)
    data = np.random.randn(1000)
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, color='steelblue', edgecolor='white')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.savefig('/home/thorin/truthspace-lcm/output/histogram.png', dpi=150)
    plt.close()
    print("Saved histogram.png")

# 4. Cosine Wave
def cosine_wave():
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.cos(x)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('cos(x)')
    plt.title('Cosine Wave')
    plt.grid(True)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.savefig('/home/thorin/truthspace-lcm/output/cosine_wave.png', dpi=150)
    plt.close()
    print("Saved cosine_wave.png")

if __name__ == "__main__":
    bar_chart()
    scatter_plot()
    histogram()
    cosine_wave()
    print("\nAll plots saved to /home/thorin/truthspace-lcm/output/")
