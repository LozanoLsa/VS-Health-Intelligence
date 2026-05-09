from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.pyplot as plt


def draw_rect(ax, x, y, w, h, facecolor, edgecolor="#0D1B2A",
              linewidth=1.2, alpha=0.85, zorder=2):
    rect = Rectangle((x, y), w, h, linewidth=linewidth,
                      edgecolor=edgecolor, facecolor=facecolor,
                      alpha=alpha, zorder=zorder)
    ax.add_patch(rect)
    return rect


def draw_area_band(ax, x_min, x_max, y_min, y_max,
                   facecolor="#F0F0F0", edgecolor="#CCCCCC",
                   linewidth=0.8, alpha=0.30, zorder=0):
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                      linewidth=linewidth, edgecolor=edgecolor,
                      facecolor=facecolor, alpha=alpha, zorder=zorder)
    ax.add_patch(rect)
