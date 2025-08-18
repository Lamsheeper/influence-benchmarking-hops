#!/usr/bin/env python3
"""
Test script to visualize the Tufte-style color palette
"""

import sys
sys.path.append('/share/u/lofty/influence-benchmarking-hops/experiments')

from utils.influence_visualization import TUFTE_COLORS, get_function_color
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure showing the color palette
fig, ax = plt.subplots(figsize=(12, 8))

# Base functions (hard colors)
base_functions = ['<GN>', '<JN>', '<KN>', '<LN>', '<MN>', '<NN>', '<ON>', '<PN>', '<QN>', '<RN>']
wrapper_functions = ['<FN>', '<IN>', '<HN>', '<SN>', '<TN>', '<UN>', '<VN>', '<WN>', '<XN>', '<YN>']

y_pos = 0
for i, (base_func, wrapper_func) in enumerate(zip(base_functions, wrapper_functions)):
    # Base function (hard color)
    base_color = get_function_color(base_func)
    base_rect = patches.Rectangle((0, y_pos), 2, 0.8, facecolor=base_color, edgecolor='white', linewidth=1)
    ax.add_patch(base_rect)
    ax.text(1, y_pos + 0.4, base_func, ha='center', va='center', fontweight='bold', fontsize=12)
    
    # Wrapper function (soft color)
    wrapper_color = get_function_color(wrapper_func)
    wrapper_rect = patches.Rectangle((2.5, y_pos), 2, 0.8, facecolor=wrapper_color, edgecolor='white', linewidth=1)
    ax.add_patch(wrapper_rect)
    ax.text(3.5, y_pos + 0.4, wrapper_func, ha='center', va='center', fontweight='bold', fontsize=12)
    
    y_pos += 1

ax.set_xlim(-0.5, 5)
ax.set_ylim(-0.5, len(base_functions))
ax.set_title('Tufte-Style Color Palette\nBase Functions (Hard Colors) vs Wrapper Functions (Soft Colors)', 
             fontsize=14, fontweight='bold', pad=20)

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Add legend
ax.text(1, -0.3, 'Base Functions\n(Hard Colors)', ha='center', va='top', fontsize=10, style='italic')
ax.text(3.5, -0.3, 'Wrapper Functions\n(Soft Colors)', ha='center', va='top', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('/share/u/lofty/influence-benchmarking-hops/tests/scratchpad/tufte_color_palette.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Color palette visualization saved to tests/scratchpad/tufte_color_palette.png")
print("\nColor mappings:")
for func in sorted(TUFTE_COLORS.keys()):
    print(f"  {func}: {TUFTE_COLORS[func]}")