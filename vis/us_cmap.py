import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot
from matplotlib import cycler


cdict = {
    'USblue': (0, 81, 158),  # 004191
    'USlightblue': (0, 190, 255),  # 00BEFF
    'USblack': (62, 68, 76),  # 323232
    'USred': (222, 39, 22),  # DE2716
    'USgreen': (59, 69, 3),  # 3B4503
    'USviolet': (168, 53, 155),  # A8359B
    'USlightgreen': (135, 184, 0),  # 87B800
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}
steps = list(i * 0.1 for i in range(1, 11))


def normalize(values):
    return tuple(v / 255 for v in values)


def get_rgba(color_name, alpha=1):
    return normalize(cdict[color_name]) + (alpha,)


def scale_white_amount(rgb, percent):
    scaled_color = []
    for color in rgb:
        color = color / 255.0
        scaled_color.append(color + (1 - color) * (1 - percent))

    return tuple(scaled_color)


def register_name(new_name, color):
    try:
        # just a relabeling of the name
        matplotlib.colors.ColorConverter.colors[new_name] = matplotlib.colors.ColorConverter.colors[color]
    except KeyError:
        if type(color) is str:
            color = get_rgba(color)
        elif max(color) > 1:
            color = tuple(c / 255 for c in color) + (1,)
        matplotlib.colors.ColorConverter.colors[new_name] = color


def color_cycler():
    return cycler(color=[color for color in cdict if 'US' in color])


def activate():
    matplotlib.pyplot.set_cmap('US')
    matplotlib.rc('image', cmap='US')
    matplotlib.rc('axes', prop_cycle=color_cycler())


list_cmap = []
for name, values in cdict.items():
    rgba = get_rgba(name)
    list_cmap.append(rgba)
    matplotlib.colors.ColorConverter.colors[name] = rgba
    for step in steps:
        matplotlib.colors.ColorConverter.colors[f'{name}!{step * 100:.0f}'] = scale_white_amount(values, step)

matplotlib.cm.register_cmap(name='US', cmap=matplotlib.colors.ListedColormap(list_cmap))

register_name('others', (200, 200, 200))
register_name('graytext', (120, 120, 120))
# register_name('ours', 'KITgreenCMYK')
# register_name('ours2', 'KITlilaCMYK')
# register_name('cutoffline', 'KITred40')
# register_name('cutofftext', 'KITred60')
#
# register_name('half', 'KITlilaCMYK')
# register_name('third', 'KITgreenCMYK')
# register_name('third55', 'KITblueCMYK')
#
# register_name('mw_disturbs', 'KITlilaCMYK')
# register_name('mw_writes', 'KITgreenCMYK')

if __name__ == '__main__':
    from pathlib import Path

    activate()
    colors = {}
    for base_name in cdict:
        step_labels = [f'{step * 100:.0f}' for step in steps]
        for name in [base_name] + [f'{base_name}!{sl}' for sl in step_labels]:
            colors[name] = ', '.join(f'{v:.3f}' for v in matplotlib.colors.to_rgb(name))

    out = Path('uni-stuttgart-colors.sty')
    out.write_text(
        r'\NeedsTeXFormat{LaTeX2e}'
        r'\ProvidesPackage{colorsUS}[2020 Color map for Uni Stuttgart]'
        r'\RequirePackage{xcolor}'
    )
    out.write_text(
        '\n'.join(f'\\definecolor{{{name}}}{{rgb}}{{{rgb}}}' for name, rgb in colors.items())
    )
    print(f'wrote color map to {out.as_posix()}')