import numpy as np
import pylab as pl

from scipy import stats as sstats
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from skimage.measure import find_contours

from utils import compute_mean_traces, compute_auc_tone, compute_auc_pretone, compute_all_dffs

def plot_mean(time_ax_single, traces_means, traces_std, cell, ax=None, shift=0, color='k'):
    if ax is None:
        fig, ax = pl.subplots(1, 1)
    ax.plot(time_ax_single, traces_means[:, cell]+shift, color='k')
    ax.fill_between(time_ax_single,
                 traces_means[:, cell]+shift-traces_std[:, cell],
                 traces_means[:, cell]+shift+traces_std[:, cell],
                 color=color, zorder=0, alpha=0.5, lw=0)
    return ax

def plot_summary(time_ax_single, traces_means, traces_std, ncells_x, ncells_y, cells,
                 rescalex=2, rescaley=1, cmap=pl.cm.rainbow,
                 cs_start_end=(0, 4), us_start_end=(8, 12), cs_color='r', us_color='g'):
    colors = cmap(np.linspace(0, 1, traces_means.shape[1]))
    fig, axs = pl.subplots(ncells_y, ncells_x, sharex=True, sharey=True,
                           figsize=(ncells_x*rescalex, ncells_y*rescaley))
    for cell, ax, col in zip(cells, axs.flatten(), colors[cells[0]-1:]):
        ax = plot_mean(time_ax_single, traces_means, traces_std, cell-1, ax=ax, color=col)
        plot_period_bar(ax, -0.1, 0.02, color=cs_color, start_end=cs_start_end)
        plot_period_bar(ax, -0.1, 0.02, color=us_color, start_end=us_start_end)
        ax.text(0, 0.3, cell)
    ax.set_xticks((time_ax_single[0], time_ax_single[-1]))
    ax.set_yticks((-.15, 0.5))
    ax.set_xlim((time_ax_single[0], time_ax_single[-1]))
    ax.set_ylim((-.15, 0.5))
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Cell #')    
    return fig, ax
    
def plot_period_bar(ax, y, delta_y=0.5, color='b', start_end=(0, 30)):
    ax.fill_between([start_end[0], start_end[1]],
                    (y, y), (y+delta_y, y+delta_y), color=color, lw=0)

def plot_auc(time_ax, dff, cycles, time_ax_single, auc_baseline, ax=None,
    tone_start=0, tone_duration=30, pretone_duration=30):
    auc_tone = compute_auc_tone(time_ax, dff, cycles, time_ax_single, auc_baseline,
    	tone_start=tone_start, tone_duration=tone_duration)
    auc_pretone = compute_auc_pretone(time_ax, dff, cycles, time_ax_single, auc_baseline,
    	tone_start=tone_start, tone_duration=tone_duration,
	pretone_duration=pretone_duration)

    lines = []
    lines.append(pl.bar(np.arange(len(auc_tone))+0.5, auc_tone, color='', edgecolor='b'))
    lines.append(pl.bar(np.arange(len(auc_tone))+0.5, auc_pretone, color='',
                        edgecolor='r'))
    lines.append(pl.bar(np.arange(len(auc_tone))+0.5, auc_tone-auc_pretone, color=(0.8, 0.8, 0.8),
                        edgecolor='', zorder=0))
    pl.hlines(0, 1, len(auc_pretone)+1, color='k')
    pl.xlabel('Cell #')
    pl.xticks(range(1, len(auc_pretone)+1, 5));
    pl.ylabel(r'AUC ($s \cdot \Delta f/f)$')
    pl.legend(lines, ['Tone', 'Pretone', 'Difference'])
    pl.ylim(-1, 1)

def plot_lick_ratios(lick_ratios, is_rwt, is_apt, axs=None, colors=['b', 'r']):
    
    if axs is None:
        fig, axs = pl.subplots(1, 2, gridspec_kw={'width_ratios':(1, 5)}, sharey=True)
    ax = axs[1]
    y, bins, patches = ax.hist([lick_ratios[is_rwt], lick_ratios[is_apt]], color=colors,
               bins=np.arange(0, 1.1, .25))
    ax.text(0.15, 12, 'BLUE = reward, RED = airpuff')
    ax.set_xlabel("Anticipatory licking ratio")
    ax.set_xticks(bins)
    # ax.set_xticklabels(['no licks', 0, 0.5, 1])
    ax = axs[0]
    ax.hist([lick_ratios[is_rwt], lick_ratios[is_apt]], color=['b', 'r'],
               bins=np.arange(-1, 0, 0.25))
    ax.set_xlim(-1, -0.75)
    ax.set_xticks((-0.875,))
    ax.set_xticklabels(('no licks',))
    ax.set_ylabel('Frequency')
    
    return axs


def plot_licks(onsets, licks, ax=None, positions=None, filter_ons=None, t_start=-8, t_end=8, **vlines_args):
    if ax is None:
        fig, ax = pl.subplots(1, 1)
    if filter_ons is None:
        filter_ons = [True] * len(onsets)
    if positions is None:
        positions = range(len(onsets))
    for i, ons in zip(positions, onsets):
        lick_filt = ((licks-ons)>t_start) * ((licks-ons)<t_end)
        ax.vlines((licks-ons)[lick_filt], i, i+1, **vlines_args)
    return ax


def add_significance(ax, array1, array2, x1, x2, y, ticksize=0.02, sig_func=None, thresholds=(0.05, 0.01, 0.001)):
    if sig_func is None:
        sig_func = lambda x, y: sstats.mannwhitneyu(x, y, alternative='two-sided')
    p = sig_func(array1, array2)[-1]
    deltay = np.diff(ax.axis()[-2:])*ticksize
    line = Line2D([x1, x1, x2, x2], [y-deltay, y, y, y-deltay], lw=1, color='k', clip_on=False)
    ax.add_line(line)
    ax.text(np.mean([x1, x2]), y+deltay,
            'n.s.' if p>thresholds[0] else
            '*' if p>thresholds[1] else
            '**' if p>thresholds[2] else
            '***',
            ha='center', fontsize=5)


def nicer_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def barplot(ax, pos, y, yerr=None, color=None):
    ax.bar(pos, y, yerr=yerr, color=(1, 1, 1, 1), ecolor=color, lw=1, edgecolor=color)


def plot_bars(list_of_values, xpos=None, colors='k', stderr=False, ax=None):
    if ax is None:
        fig, ax = pl.subplots(1, 1)
    if xpos is None:
        xpos = range(len(list_of_values))
    ax.bar(xpos,
           [np.mean(values) for values in list_of_values],
           yerr=[np.std(values)/(1. if not stderr else 1.*np.sqrt(len(values)-1)) for values in list_of_values],
           color=(0, 0, 0, 0), ecolor=colors, lw=1, edgecolor=colors,
           error_kw={'lw':1})
    return ax

def plot_violins(list_of_values, xpos=None, colors='k', ax=None):
    if ax is None:
        fig, ax = pl.subplots(1, 1)
    if xpos is None:
        xpos = range(len(list_of_values))
    res = ax.violinplot(list_of_values, positions=xpos, showmedians=True, showextrema=False)
    kolors = colors if isinstance(colors, list) else [colors]*len(xpos)
    for c, b in zip(kolors, res['bodies']):
        b.set_color(c)
    #res['cbars'].set_linewidths(1)
    res['cmedians'].set_color(kolors)
    # res['cmaxes'].set_color((0, 0, 0, 0))
    # res['cmins'].set_color((0, 0, 0, 0))
    #res['cbars'].set_color((0, 0, 0, 0))
    return ax


def plot_rois(mean_image, contours, list_of_cells, colors=None, text_color=None, alpha_mean_image=0, ax=None):
    """
    Code from `ROI plot.ipynb` by someone.
    """
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=(3, 3))
    
    # filter out weird jumps
    for neuron in xrange(contours.shape[0]):
        for dim in xrange(2):
            mask=np.where(abs(np.diff(contours[neuron][dim]))>10)
            for x in mask:
                x+=1
            contours[neuron][dim][mask]=np.nan
    #find number of coordinates in smallest ROI (store this value in min_coor)
    min_coor=1000
    for neuron in contours:
        temp=neuron.shape[1]
        if temp<min_coor:
            min_coor=temp
    #plot mean_image, which is the correlation image and overlay the ROIs 
    patches=[]
    ax.imshow(mean_image, cmap=pl.cm.gray, alpha=alpha_mean_image)
    if colors is None:
        colors = ['r'*len(l) for l in list_of_cells]
    for l, color in zip(list_of_cells, colors):
        for neuron in l:
            polygon = Polygon(contours[neuron].T, color=color, alpha=0.3, lw=0)
            patches.append(polygon)
            if text_color is not None:
                ax.text(contours[neuron][0,min_coor-1], contours[neuron][1, min_coor-1], neuron, color=text_color, size=8)
            ax.add_patch(polygon)
    # polygon = Polygon(np.transpose(contours[18][:,10:]))
    # patches.append(polygon)
    # ax.text(contours[18][0,min_coor-1], contours[18][1,min_coor-1], 21+1, color='y',size=8)
    # colors = 100*np.random.rand(len(patches))
    #p = PatchCollection(patches, alpha=0.65)
    #p.set_array(colors)
    #ax.add_collection(p)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_areas(areas, cells, color='r', plot_borders=True, ax=None, alpha=0.3):
    """
    Used in diggsie.
    """
    
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=(2, 2))

    ax.imshow(np.zeros_like(areas[0]), alpha=0)
    patches = []
    for cell in cells:
        contours = np.r_[find_contours(areas[cell]>0, 0.5)]
        polygon = Polygon(contours[0], color=color, alpha=alpha, lw=0)
        ax.add_patch(polygon)

    if plot_borders:
        all_the_rest = np.arange(len(areas))
        patches = []
        for cell in all_the_rest:
            contours = np.r_[find_contours(areas[cell]>0, 0.5)]
            ax.plot(contours[0][:, 0], contours[0][:, 1], lw=0.5, color='0.7', zorder=0)
        
    ax.set_xticks(())
    ax.set_yticks(())
    
#     for s in ax.spines.itervalues():
#         s.set_visible(False)
    
    return ax

def plot_heat_map(time_ax, traces, cell, cycles, time_ax_single, vlines=(0, 4), ax=None, **args):
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=(3, 2))
    alls = compute_all_dffs(time_ax, traces, cell, cycles, time_ax_single)
    ax.imshow(alls, origin='lower', aspect='auto',
              extent=(time_ax_single[0], time_ax_single[-1], 0, len(cycles)), **args)
    ymax = ax.axis()[-1]
    ax.vlines(vlines, 0, ymax, color='red', lw=1)
    return ax

def remove_axes(ax):
    for s in ax.spines.itervalues():
        s.set_visible(False)
    ax.set_xticks(())
    ax.set_yticks(())


def plot_cumulative(values, ax=None, bins=50, **args):
    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=(3, 2))
    y1, x = np.histogram(values, bins=bins)
    ax.step(x[1:], 1.*np.cumsum(y1)/np.sum(y1), where='pre', **args)
    
def draw_scale_line_xy(ax, length=(10, 10), offset=(0, 0), **line_args):
    xmin, xmax, ymin, ymax = ax.axis()
    xdata = (xmin+offset[0], xmin+offset[0], xmin+offset[0]+length[0])
    ydata = (ymin+offset[1]+length[1], ymin+offset[1], ymin+offset[1])
    if 'color' not in line_args.keys():
            line_args.update({'color':'k'})
    l = Line2D(xdata, ydata, **line_args)
    l.set_clip_on(False)
    ax.add_line(l)
    
def plot_spatial_footprints(areas, cells, background=False, color='r', lw=0.5, cont_threshold=0.2,
                            blur_size=5, ax=None, cmap=pl.cm.gray, every=3, facecolor='w', **args):

    if ax is None:
        fig, ax = pl.subplots(1, 1, figsize=(2, 2))

    ax.imshow(np.zeros_like(areas[0]), alpha=0)
    patches = []
    
    if background:
        for cell in range(len(areas)):
            a = areas[cell][:].T
            a = exposure.rescale_intensity(a)
            psf = np.ones((blur_size, blur_size)) / blur_size**2
            img = convolve2d(a, psf, 'same')
            img += 0.1 * img.std() * np.random.standard_normal(img.shape)
            img[img<0.2] = np.nan
            ax.imshow(img+0.2, cmap, vmin=0, vmax=1, alpha=1)
    for cell in cells:
        a = exposure.rescale_intensity(areas[cell])
        contours = np.r_[find_contours(a, cont_threshold)]
        con = contours[np.argmax([len(c) for c in contours])]
        x, y = con.T
        ax.plot(x[::every], y[::every], color=color, lw=lw)

    ax.set_facecolor(facecolor)
        
    ax.set_xticks(())
    ax.set_yticks(());

    return ax
