import sys
sys.path.append('..')

import numpy as np
import gudhi as gd
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


from src.utils.converters import extract_simplices




def distance_plot(labels, logits, 
                    fig_in=None, ax_in=None,
                    figsize=(6,6),
                    title=''):
    #------ distance plot --------
    
    sort_indx=np.argsort(labels)
    sorted_labels=labels[sort_indx]
    sorted_logits=logits[sort_indx]
    
    if (fig_in is None) or (ax_in is None):
        fig=plt.figure()
        ax=fig.add_subplot(111)
    else:
        fig=fig_in
        ax=ax_in
        
    ax.plot(range(len(sorted_logits)),sorted_logits, 
            range(len(sorted_labels)), sorted_labels)
    ax.legend(['learned', 'true'], loc='lower right')
    ax.set_title(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('simplices')
    ax.set_ylabel('distance')
    
    return fig, ax


def distance_plot_w_inset(pts, simplices, labels, logits, 
                          default_vals=[0.1,0.0,0.5], 
                          fig_in=None, ax_in=None,
                          figsize=(6,6), alphas=(0.0, 1.0,0.3), cmap='jet', show_cbar=True, normalize_cbar=True,
                          title='',
                          inset_loc=[0.12, 0.5, 0.45, 0.45]):

    #------ distance plot --------
    fig, ax=distance_plot(labels, logits, fig_in=fig_in, ax_in=ax_in, figsize=figsize, title=title)
    
    #------- complex inset ---------
    rect=inset_loc
    
    plain_distances=[np.array([default_vals[0]]*len(simplices[0])),np.array([default_vals[1]]*len(simplices[1])),np.array([default_vals[2]]*len(simplices[2]))]

    distances_L=plain_distances
    distances_L[1]=logits
    
    if pts.shape[1]==3:
        axins = fig.add_axes(rect, anchor='NW', projection='3d')
    else:
        axins = fig.add_axes(rect, anchor='NW')
        
    complex_pyplot(pts, simplices, distances_L,figsize=figsize, alphas=alphas, fig=fig, ax=axins, cmap=cmap, show_cbar=False, normalize_cbar=normalize_cbar)
    axins.set_axis_off()
    axins.patch.set_alpha(0.0)

    return fig,ax


#----------- complex vizualization ------------

def complex_pyplot(pts, simplices, distances, alphas=(1.0, 1.0, 1.0), fig=None, ax=None, 
                figsize=(20,15), view=(90,0), cmap='jet', show_cbar=True, normalize_cbar=True, **kwargs):

    if len(simplices)>1:
        lines=get_positions(pts,simplices,1)    
    else:
        lines=[]
        
    if len(simplices)>2:
        triangles=get_positions(pts, simplices,2)
    else:
        triangles=[]
    
    if (fig is None) or (ax is None):
        if pts.shape[1]==2:
            fig, ax = plt.subplots(figsize=figsize)
        elif pts.shape[1]==3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
    
    s_0_indx=[list(s[0])[0] for s in simplices[0].items()]
    
    if len(triangles)>0:
        plot_triangles(triangles, distances[2], ax, zorder=2, alpha=alphas[2], cmap=cmap, normalize_cbar=normalize_cbar, **kwargs)    
    if len(lines)>0:
        plot_edges(lines, distances[1], ax, zorder=1, alpha=alphas[1], cmap=cmap, normalize_cbar=normalize_cbar, **kwargs)

    plot_nodes(pts[s_0_indx,:], distances[0], ax, zorder=0, alpha=alphas[0], cmap=cmap, normalize_cbar=normalize_cbar, **kwargs)

    if show_cbar:
        cb=mpl.cm.ScalarMappable(cmap=cmap)
        fig.colorbar(cb, ax=ax, fraction=0.026, pad=0.02)
        if not normalize_cbar:
            cb.set_clim(np.min(np.hstack(distances)), np.max(np.hstack(distances)))
    
    
    ax.set_xlim(np.min(pts[:,0]), np.max(pts[:,0]))
    ax.set_ylim(np.min(pts[:,1]), np.max(pts[:,1]))
    if pts.shape[1]==3:
        ax.set_zlim(np.min(pts[:,2]), np.max(pts[:,2]))
    
    if pts.shape[1]==3:
        ax.view_init(*view)

    return fig, ax



def get_positions(points, simplices, dim):
    polygons = list()
    for i, simplex in enumerate(simplices[dim].keys()):
        assert simplices[dim][simplex] == i  # Dictionary is ordered.
        polygon = list()
        for vertex in simplex:
            polygon.append(points[vertex])
        polygons.append(polygon)
    return polygons





def value2color(values, cmap='jet', normalize=True):
    values=np.array(values)
    if normalize:
        if not values.min()==values.max():
            values -= values.min()
            values /= values.max()
    if type(cmap) is str:
        return mpl.cm.get_cmap(cmap)(values)
    else:
        return cmap(values)




def plot_nodes(points, colors, ax=None, cmap='jet', normalize_cbar=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    colors=value2color(colors, cmap=cmap, normalize=normalize_cbar)
    
    scargs={}
    if 's' in kwargs:
        scargs['s']=kwargs['s']
    if 'alpha' in kwargs:
        scargs['alpha']=kwargs['alpha']
    
    if points.shape[1]==2:
        ax.scatter(points[:, 0], points[:, 1], c=colors, **scargs)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, **scargs)
    return ax.figure, ax

def plot_edges(lines, colors, ax=None, cmap='jet', normalize_cbar=True, **kwargs):
    if ax is None:
        fig, ax = plt.subfigs()
    colors = value2color(colors, cmap=cmap, normalize=normalize_cbar)
    
    lcargs={}
    if 'linewidths' in kwargs:
        lcargs['linewidths']=kwargs['linewidths']
    if 'alpha' in kwargs:
        lcargs['alpha']=kwargs['alpha']
    
    if len(lines[0][0])==2:
        collection = mpl.collections.LineCollection(lines, colors=colors, **lcargs)
        ax.add_collection(collection)
    elif len(lines[0][0])==3:
        collection = Line3DCollection(lines, colors=colors, **lcargs )
        ax.add_collection(collection)
    return ax.figure, ax


def plot_triangles(triangles, colors, ax=None, cmap='jet', normalize_cbar=True, **kwargs):
    if ax is None:
        fig, ax = plt.subfigs()
    scargs={}
    if 'alpha' in kwargs:
        scargs['alpha']=kwargs['alpha']
    if len(triangles[0][0])==2:
        colors = value2color(colors, cmap=cmap, normalize=normalize_cbar)
        for triangle, color in zip(triangles, colors):
            triangle = plt.Polygon(triangle, color=color, **scargs)
            ax.add_patch(triangle)
    elif len(triangles[0][0]==3):
            colors = value2color(colors, cmap=cmap, normalize=normalize_cbar)
            collection = Poly3DCollection(triangles, facecolors=colors, **scargs)
            ax.add_collection(collection)
    return ax.figure, ax



