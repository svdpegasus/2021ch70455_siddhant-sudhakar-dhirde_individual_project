"""
Self-Organized Criticality in Forest Fire Systems
===================================================
Simulation of the Drossel-Schwabl Forest Fire Model (1992)
demonstrating Self-Organized Criticality (SOC).

Author: [Student Name]
Date: April 2026

Rules:
  1. A burning tree becomes empty (ash) in the next step.
  2. A green tree with at least one burning neighbour catches fire.
  3. A green tree with no burning neighbour ignites with probability f (lightning).
  4. An empty cell grows a new tree with probability p.

At steady state with p/f >> 1, the system self-organizes to a critical state
where avalanche (fire) sizes follow a power-law distribution.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from collections import deque
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── States ────────────────────────────────────────────────────────────────────
EMPTY   = 0   # ash / bare soil
TREE    = 1   # green tree
BURNING = 2   # burning tree

# ── Parameters ────────────────────────────────────────────────────────────────
L        = 128          # grid size (L x L)
p        = 0.05         # tree-growth probability per step
f        = 0.0005       # lightning probability  (p/f = 100)
N_STEPS  = 8_000        # total Monte-Carlo steps
BURN_IN  = 2_000        # steps to discard (transient)

CONNECTIVITY_VALUES = [4, 8]   # 4-neighbour (von Neumann) vs 8-neighbour (Moore)

# ─────────────────────────────────────────────────────────────────────────────
def bfs_fire_size(grid, start_r, start_c, conn):
    """
    BFS to find all cells ignited by a single lightning strike.
    Returns the number of trees burned (avalanche size).
    Modifies grid in-place: sets burned cells to BURNING.
    """
    rows, cols = grid.shape
    queue = deque()
    queue.append((start_r, start_c))
    grid[start_r, start_c] = BURNING
    size = 0

    if conn == 4:
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
    else:
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    while queue:
        r, c = queue.popleft()
        size += 1
        for dr, dc in directions:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == TREE:
                grid[nr, nc] = BURNING
                queue.append((nr, nc))
    return size


def run_simulation(L, p, f, n_steps, burn_in, conn):
    """
    Run the Drossel-Schwabl forest fire model.
    Returns list of avalanche sizes recorded after burn-in.
    """
    # Initialise grid: 50% trees
    grid = RNG.choice([EMPTY, TREE], size=(L, L))
    avalanche_sizes = []

    for step in range(n_steps):
        new_grid = grid.copy()

        # Step 1: Burning → Empty
        new_grid[grid == BURNING] = EMPTY

        # Step 2: Spread fire to neighbours (already handled via BFS below)
        # We re-implement the spreading as a synchronous BFS on each step's burning cells
        # Here we use the simpler simultaneous update:
        #   Any TREE adjacent to a BURNING cell catches fire
        if conn == 4:
            kernel = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=bool)
        else:
            kernel = np.ones((3,3), dtype=bool); kernel[1,1] = False

        from scipy.ndimage import convolve
        burning_mask = (grid == BURNING).astype(np.int8)
        neighbour_fire = convolve(burning_mask, kernel.astype(np.int8),
                                  mode='constant', cval=0)
        catch_fire = (grid == TREE) & (neighbour_fire > 0)
        new_grid[catch_fire] = BURNING

        # Step 3: Lightning ignition of isolated trees
        lightning = (grid == TREE) & (neighbour_fire == 0) & (RNG.random((L,L)) < f)

        # For each lightning strike, do a BFS and record avalanche size
        for (r, c) in zip(*np.where(lightning)):
            if new_grid[r, c] == TREE:   # still a tree (not already burning)
                size = bfs_fire_size(new_grid, r, c, conn)
                if step >= burn_in:
                    avalanche_sizes.append(size)

        # Step 4: Empty → Tree with probability p
        grow = (grid == EMPTY) & (RNG.random((L,L)) < p)
        new_grid[grow] = TREE

        grid = new_grid

    return avalanche_sizes, grid


def plot_power_law(sizes, conn, ax, color):
    """Log-log histogram + power-law fit."""
    sizes = np.array(sizes)
    sizes = sizes[sizes > 0]
    if len(sizes) < 10:
        return

    # Logarithmic binning
    bins = np.logspace(0, np.log10(sizes.max()), 40)
    counts, edges = np.histogram(sizes, bins=bins)
    centres = 0.5*(edges[:-1] + edges[1:])
    mask = counts > 0
    x, y = centres[mask], counts[mask]

    ax.scatter(x, y, s=18, color=color, alpha=0.7,
               label=f'Conn={conn} (n={len(sizes)})')

    # Linear regression in log-log space for power-law exponent
    slope, intercept, r, *_ = stats.linregress(np.log10(x), np.log10(y))
    xfit = np.linspace(x.min(), x.max(), 200)
    ax.plot(xfit, 10**intercept * xfit**slope, '--', color=color, linewidth=1.5,
            label=f'Slope τ = {-slope:.2f}  (R²={r**2:.3f})')

    return -slope


def snapshot(grid, title, fname):
    """Save a colour snapshot of the grid."""
    cmap = mcolors.ListedColormap(['#d2b48c', '#228B22', '#FF4500'])
    norm = mcolors.BoundaryNorm([0,1,2,3], cmap.N)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)

    exponents = {}
    all_sizes  = {}

    for conn in CONNECTIVITY_VALUES:
        print(f"Running simulation with connectivity = {conn} ...")
        sizes, final_grid = run_simulation(L, p, f, N_STEPS, BURN_IN, conn)
        all_sizes[conn] = sizes
        print(f"  Total avalanches recorded: {len(sizes)}")
        print(f"  Max avalanche size       : {max(sizes) if sizes else 0}")

        snapshot(final_grid,
                 f'Forest Fire Grid (conn={conn})',
                 f'figures/grid_conn{conn}.png')

    # ── Power-law plot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7,5))
    colors = ['#1f77b4', '#d62728']
    for conn, color in zip(CONNECTIVITY_VALUES, colors):
        tau = plot_power_law(all_sizes[conn], conn, ax, color)
        if tau: exponents[conn] = tau

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Avalanche size $s$ (number of trees burned)', fontsize=12)
    ax.set_ylabel('Frequency $N(s)$', fontsize=12)
    ax.set_title('Power-Law Distribution of Fire Avalanches\n'
                 '(Drossel–Schwabl Forest Fire Model, SOC)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig('figures/power_law.png', dpi=150)
    plt.close()
    print("Saved figures/power_law.png")

    # ── Connectivity vs exponent ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5,4))
    conns = list(exponents.keys())
    taus  = [exponents[c] for c in conns]
    bars = ax.bar([str(c) for c in conns], taus, color=['#1f77b4','#d62728'],
                  width=0.4, edgecolor='k')
    for bar, tau in zip(bars, taus):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{tau:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xlabel('Connectivity (number of neighbours)', fontsize=11)
    ax.set_ylabel('Power-law exponent τ', fontsize=11)
    ax.set_title('Effect of Connectivity on Avalanche Exponent', fontsize=11)
    ax.set_ylim(0, max(taus)*1.3 if taus else 3)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('figures/exponent_vs_connectivity.png', dpi=150)
    plt.close()
    print("Saved figures/exponent_vs_connectivity.png")

    # ── Time-series of avalanche sizes ────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(9,5), sharex=False)
    for ax, conn, color in zip(axes, CONNECTIVITY_VALUES, colors):
        s = all_sizes[conn]
        ax.plot(range(len(s)), s, '.', markersize=1.5, color=color, alpha=0.5)
        ax.set_ylabel('Avalanche size', fontsize=9)
        ax.set_title(f'Connectivity = {conn}', fontsize=9)
        ax.set_yscale('log')
        ax.grid(True, linestyle=':', alpha=0.3)
    axes[-1].set_xlabel('Event index', fontsize=10)
    fig.suptitle('Time Series of Avalanche Sizes (log scale)', fontsize=11)
    plt.tight_layout()
    plt.savefig('figures/timeseries.png', dpi=150)
    plt.close()
    print("Saved figures/timeseries.png")

    print("\nAll simulations and figures complete.")
    print("Exponents:", exponents)
