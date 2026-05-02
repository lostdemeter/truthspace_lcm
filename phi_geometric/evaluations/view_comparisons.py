#!/usr/bin/env python3
"""
Simple HTTP server to view colorizer comparisons.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8765
DIRECTORY = "/home/thorin/truthspace-lcm/phi_geometric/evaluations/colorizer_comparisons"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def main():
    os.chdir(DIRECTORY)
    
    # Generate index.html
    images = sorted(Path(DIRECTORY).glob("*.jpg"))
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Colorizer Comparisons</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: white; }
        h1 { text-align: center; }
        .comparison { margin: 20px 0; text-align: center; }
        img { max-width: 100%; border: 2px solid #333; }
        .labels { display: flex; justify-content: space-around; margin-top: 10px; }
        .label { flex: 1; text-align: center; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Geometric Colorizer vs DDColor</h1>
    <p style="text-align: center;">Layout: Original | Grayscale | DDColor | Geometric (ours)</p>
"""
    
    for img in images:
        html += f"""
    <div class="comparison">
        <img src="{img.name}" alt="{img.stem}">
        <div class="labels">
            <div class="label">Original</div>
            <div class="label">Grayscale</div>
            <div class="label">DDColor</div>
            <div class="label">Geometric</div>
        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    with open(Path(DIRECTORY) / "index.html", "w") as f:
        f.write(html)
    
    print(f"Serving comparisons at http://localhost:{PORT}")
    print(f"Directory: {DIRECTORY}")
    print("Press Ctrl+C to stop")
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    main()
