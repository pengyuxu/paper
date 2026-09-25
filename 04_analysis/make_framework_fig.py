"""Generate Fig. 1: system framework diagram.

Three-branch late-fusion architecture, all consuming the 8192-pt cloud
(geometry + two electrostatic channels):
  - Branch A (deep):  RIConv++ (incl. φ/ψ channels) -> MC-FPS K=25 -> softmax -> p_A
  - Branch C1 (3DZD): Zernike on whole/positive/negative surfaces, 363-dim
                      -> 1-NN + vol gate -> softmax -> p_C1
  - Branch C2 (FPFH): 612-dim pooled FPFH incl. 18-dim potential statistics
                      -> RBF-SVM -> softmax -> p_C2
  - Fusion: p_A + p_C1 + p_C2  (equal-weight late fusion) -> argmax -> 97-class

Writes figures/framework.{pdf,png}; no data or GPU required.
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = pathlib.Path(__file__).resolve().parent.parent
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 12); ax.axis("off")

# ---- palette (consistent with Fig. 2) ----
C_DEEP = "#3B6FB6"
C_ZDZ  = "#2E9E6B"
C_FPFH = "#E8A33D"
C_FUSE = "#D9534F"
C_OUT  = "#7B5EA7"
C_IN   = "#666666"

def soft(c):
    h = c.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r, g, b = [int(v + (255 - v) * 0.78) for v in (r, g, b)]
    return f"#{r:02x}{g:02x}{b:02x}"

def box(x, y, w, h, text, edge, fs=8, bold=False, fc=None):
    if fc is None:
        fc = soft(edge)
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.13",
                       fc=fc, ec=edge, lw=1.4)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight="bold" if bold else "normal", color="#222")

def arrow(x1, y1, x2, y2, color="#444", lw=1.8, ms=12):
    """Thick, visible arrow with solid head (light shrink so short
    inter-box gaps still render a full arrowhead)."""
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                        color=color, lw=lw, shrinkA=1, shrinkB=2,
                        connectionstyle="arc3,rad=0")
    ax.add_patch(a)

def branch_bg(x, y, w, h, edge, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.12,rounding_size=0.18",
                 fc=soft(edge), ec=edge, lw=1.0, alpha=0.28, linestyle="--"))
    ax.text(x + 0.22, y + h - 0.30, label, fontsize=9, color=edge, weight="bold")

# ============================================================
# INPUT  (far left)
# ============================================================
box(0.2, 9.2, 1.9, 1.0, "VTK mesh\n(protein surface)", C_IN, fs=8.5)
box(0.2, 7.0, 1.9, 1.5, "8192-pt cloud\n$[xyz, n, \\phi, \\psi]$", C_IN, fs=8.5, bold=True)
arrow(1.15, 9.2, 1.15, 8.5, color=C_IN)

# junction label (below the cloud, clear of the fan of arrows)
ax.text(1.15, 6.55, "point cloud\nto all 3 branches", fontsize=7.5,
        color=C_IN, ha="center", va="center", style="italic")

# ============================================================
# BRANCH A (deep)  -- top
# ============================================================
branch_bg(4.0, 7.7, 3.1, 4.1, C_DEEP, "Layer A  (deep)")
box(4.3, 10.50, 2.5, 0.85, "RIConv++\n(5 SA stages)", C_DEEP, fs=8.5)
box(4.3, 9.15, 2.5, 0.85, "MC-FPS  K=25\n(mean-logit)", C_DEEP, fs=8.5)
box(4.3, 7.90, 2.5, 0.75, "softmax  $\\tau=1.0$", C_DEEP, fs=8.5)
arrow(5.55, 10.50, 5.55, 10.00, color=C_DEEP)
arrow(5.55, 9.15, 5.55, 8.65, color=C_DEEP)
# input -> first box of A (RIConv++), straight slanted line
arrow(2.1, 7.75, 4.3, 10.925, color=C_IN)

# ============================================================
# BRANCH C1 (3DZD) -- middle
# ============================================================
branch_bg(4.0, 4.55, 3.1, 2.85, C_ZDZ, "Layer C1  (3DZD + $\\phi$)")
box(4.3, 6.10, 2.5, 0.85, "3DZD $\\times$3 surfaces\nshape / pos / neg, 363-d",
    C_ZDZ, fs=7.8)
box(4.3, 4.75, 2.5, 0.85, "1-NN + vol gate\n$V_q/V_g \\in [.8,1.2]$", C_ZDZ, fs=8)
arrow(5.55, 6.10, 5.55, 5.60, color=C_ZDZ)
# input -> first box of C1 (3DZD), straight slanted line
arrow(2.1, 7.75, 4.3, 6.525, color=C_IN)

# ============================================================
# BRANCH C2 (FPFH) -- bottom
# ============================================================
branch_bg(4.0, 0.15, 3.1, 4.1, C_FPFH, "Layer C2  (FPFH + $\\phi$)")
box(4.3, 2.95, 2.5, 0.85, "FPFH + $\\phi$ hist/stats\n612-d, r=0.2",
    C_FPFH, fs=7.8)
box(4.3, 1.60, 2.5, 0.85, "RBF-SVM\n$C=64, \\gamma=12$", C_FPFH, fs=8.5)
box(4.3, 0.35, 2.5, 0.75, "softmax  $\\tau=1.0$", C_FPFH, fs=8.5)
arrow(5.55, 2.95, 5.55, 2.45, color=C_FPFH)
arrow(5.55, 1.60, 5.55, 1.10, color=C_FPFH)
# input -> first box of C2 (FPFH), straight slanted line
arrow(2.1, 7.75, 4.3, 3.375, color=C_IN)

# ============================================================
# FUSION  (right) -- equal-weight sum (no misleading x2)
# ============================================================
box(7.9, 4.6, 3.1, 1.6, "Late fusion\n$p_A + p_{C1} + p_{C2}$\n(equal weight)",
    C_FUSE, fs=9, bold=True)

# arrows from each branch softmax output into fusion
arrow(6.8, 8.275, 7.9, 5.9, color=C_DEEP)
ax.text(7.62, 7.21, "equal", fontsize=7.5, color=C_DEEP, style="italic", weight="bold")

arrow(6.8, 5.175, 7.9, 5.4, color=C_ZDZ)
ax.text(7.22, 5.55, "equal", fontsize=7.5, color=C_ZDZ, style="italic", weight="bold")

arrow(6.8, 0.725, 7.9, 4.9, color=C_FPFH)
ax.text(7.64, 2.73, "equal", fontsize=7.5, color=C_FPFH, style="italic", weight="bold")

# ============================================================
# OUTPUT
# ============================================================
box(8.1, 7.6, 2.8, 1.0, "argmax  $\\rightarrow$  97-class", C_OUT, fs=9.5, bold=True)
arrow(9.45, 6.2, 9.45, 7.6, color=C_OUT, lw=2.0, ms=14)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"framework.{ext}", dpi=220, bbox_inches="tight")
print("saved", OUT / "framework.pdf")
