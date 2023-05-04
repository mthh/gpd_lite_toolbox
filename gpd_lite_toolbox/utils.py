# -*- coding: utf-8 -*-
"""
gpd_lite_toolboox utils
@author: mthh
"""

import time
import math
import rtree
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances
from shapely.errors import TopologicalError

def nrepeat(iterable, n):
    return iter([i for i in iterable for j in range(n)])


def dbl_range(df_item):
    for i in df_item.iterrows():
        for j in df_item.iterrows():
            if i[0] != j[0]:
                yield i[0], i[1], j[0], j[1]


def db_creation(path):
    import sqlite3 as db
    conn = db.connect(path)
    conn.enable_load_extension(True)
    conn.load_extension('/usr/local/lib/mod_spatialite.so.7.1.0')
    conn.executescript('PRAGMA synchronous=off; PRAGMA cache_size=1536000;')
    print('Initializing Spatial Metadata...')
    conn.execute('SELECT InitSpatialMetadata();')
    conn.commit()
    return conn


def db_connect(path):
    import sqlite3 as db
    conn = db.connect(path)
    conn.enable_load_extension(True)
    conn.load_extension('/usr/local/lib/mod_spatialite.so.7.1.0')
    conn.executescript('PRAGMA synchronous=off; PRAGMA cache_size=1536000;')
    conn.commit()
    return conn


def idx_generator_func(bounds):
    for i, bound in enumerate(bounds):
        yield (i, bound, i)


def make_index(bounds):
    return rtree.index.Index([z for z in idx_generator_func(bounds)],
                             Interleaved=True)


def mparams(gdf1):
    params = []
    for i in range(len(gdf1)):
        ftg = gdf1.geometry[i]
        len_v = len(ftg.coords.xy[0]) - 1
        first_pt_x, first_pt_y = \
            ftg.coords.xy[0][0], ftg.coords.xy[1][0]
        last_pt_x, last_pt_y = \
            ftg.coords.xy[0][len_v], ftg.coords.xy[1][len_v]
        orientation = 180 + math.atan2(
            (first_pt_x - last_pt_x), (first_pt_y - last_pt_y)
            ) * (180 / math.pi)
        params.append(
            (ftg.centroid.x, ftg.centroid.y, ftg.length, orientation))
    return params


def fh2_dist_lines2(li1, li2):
    c1 = np.array([i for i in zip(li1.coords.xy[0], li1.coords.xy[1])])
    c2 = np.array([i for i in zip(li2.coords.xy[0], li2.coords.xy[1])])
    return (pairwise_distances(c1, c2, metric='euclidean', n_jobs=1)).max()


def hav_dist(locs1, locs2):
    # (lat, lon) (lat, lon)
    locs1 = locs1 * 0.0174532925
    locs2 = locs2 * 0.0174532925
    cos_lat1 = np.cos(locs1[..., 0])
    cos_lat2 = np.cos(locs2[..., 0])
    cos_lat_d = np.cos(locs1[..., 0] - locs2[..., 0])
    cos_lon_d = np.cos(locs1[..., 1] - locs2[..., 1])
    return 6367 * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))


def bearing_180(li1):
    len_li = len(li1.coords.xy[0]) - 1
    first_pt_x, first_pt_y = li1.coords.xy[0][0], li1.coords.xy[1][0]
    last_pt_x, last_pt_y = li1.coords.xy[0][len_li], li1.coords.xy[1][len_li]
    b = 180 + math.atan2(
        (first_pt_x - last_pt_x), (first_pt_y - last_pt_y)
        ) * (180 / math.pi)
    if b > 180:
        return 180 - b
    else:
        return b

def dorling_radius(poly, value_field, ratio,
                   pi=np.pi, sqrt=np.sqrt):
    cum_dist, cum_rad = 0, 0
    centroids = poly.geometry.centroid
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            if i != j:
                l = centroids.geometry[i].distance(centroids.geometry[j])
                d = sqrt(poly.iloc[i][value_field]/pi) \
                    + sqrt(poly.iloc[j][value_field]/pi)
                cum_dist = cum_dist + l
                cum_rad = cum_rad + d

    scale = cum_dist / cum_rad
    radiuses = sqrt(poly[value_field]/pi) * scale * ratio
    norm_areas = normalize(
        [poly.geometry[i].area for i in range(len(poly))]
        )[0]
    return radiuses * norm_areas


def dorling_radius2(poly, value_field, ratio, mat_shared_border,
                   pi=np.pi, sqrt=np.sqrt):
    cum_dist, cum_rad = 0, 0
    centroids = poly.geometry.centroid
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            if i != j:
                fp = abs(
                    round(mat_shared_border[i][j] / mat_shared_border[i].sum(), 2) - 1)
                l = centroids.geometry[i].distance(centroids.geometry[j])
                d = sqrt(poly.iloc[i][value_field]/pi) \
                    + sqrt(poly.iloc[j][value_field]/pi)
                cum_dist = cum_dist + l*(fp/2)
                cum_rad = cum_rad + d
#                print(fp, cum_dist, cum_rad)
    scale = cum_dist / cum_rad
    radiuses = sqrt(poly[value_field]/pi) * scale * ratio
    norm_areas = normalize(
        [poly.geometry[i].area for i in range(len(poly))]
        )[0]
    return radiuses * norm_areas

def l_shared_border(gdf, touche_table):
    dim = len(touche_table)
    mat_sh_bord = np.empty((dim,dim))
    for id1 in range(dim):
        for id2 in touche_table.iloc[id1]:
            mat_sh_bord[id1][id2] = \
                (gdf.geometry[id1].buffer(0.01).intersection(gdf.geometry[id2])).length
            mat_sh_bord[id2][id1] = mat_sh_bord[id1][id2]
    return mat_sh_bord

def ftouches_byid(geoms1, geoms2, tolerance=0):
    """
    Return a table with a row for each features of *geoms1*, containing the id
    of each *geoms2* touching features.
    The test is not based on the *touches* predicat but on a intersects
    between the two features (which are buffered with *tolerance*)

    Parameters
    ----------
    geoms1: GeoSeries or GeoDataFrame
        Collection on which the touching table will be based.
    geoms2: GeoSeries or GeoDataFrame
        Collection to test against the first one.
    tolerance: Float
        The tolerance within two features as considered as touching.
        (in unit of both input collections)

    Returns
    -------
    touching_table: pandas.Series
        A Series with the same index id as geoms1, each row containg the ids of
        the features of geoms2 touching it.
    """
    return geoms1.geometry.apply(
        lambda x: [i for i in range(len(geoms2.geometry))
                   if x.intersects(geoms2.geometry[i].buffer(tolerance))]
        )


def intersection_part(g1, g2):
    """
    Return the part of *g1* which is covered by *g2*.
    Return 0 if no intersection or invalid geom(s).

    Parameters
    ----------
    g1: Shapely.geometry
    g2: Shapely.geometry
    """
    try:
        if g1.intersects(g2):
            return g1.intersection(g2).area / g1.area
        else:
            return 0
    except TopologicalError as err:
        print('Warning : {}'.format(err))
        return 0


def intersection_part_table(geoms1, geoms2):
    return geoms1.geometry.apply(
        lambda x: [intersection_part(x, geoms2.geometry[i]) for i in range(len(geoms2.geometry))]
        )


def make_prop_lines(gdf, field_name, color='red', normalize_values=False,
                    norm_min=None, norm_max=None, axes=None):
    """
    Display a GeoDataFrame collection of (Multi)LineStrings,
    proportionaly to numerical field.

    Parameters
    ----------
    gdf: GeoDataFrame
        The collection of linestring/multilinestring to be displayed
    field_name: String
        The name of the field containing values to scale the line width of
        each border.
    color: String
        The color to render the lines with.
    normalize_values: Boolean, default False
        Normalize the value of the 'field_name' column between norm_min and
        norm_max.
    norm_min: float, default None
        The linewidth for the minimum value to plot.
    norm_max: float, default None
        The linewidth for the maximum value to plot.
    axes:
        Axes on which to draw the plot.

    Return
    ------
    axes: matplotlib.axes._subplots.AxesSubplot
    """
    from geopandas.plotting import plot_linestring, plot_multilinestring
    from shapely.geometry import MultiLineString
    import matplotlib.pyplot as plt

    if normalize_values and norm_max and norm_min:
        vals = (norm_max - norm_min) * (normalize(gdf[field_name].astype(float))).T + norm_min
    elif normalize_values:
        print('Warning : values where not normalized '
              '(norm_max or norm_min is missing)')
        vals = gdf[field_name].values
    else:
        vals = gdf[field_name].values

    if not axes:
        axes = plt.gca()
    for nbi, line in enumerate(gdf.geometry.values):
        if isinstance(line, MultiLineString):
            plot_multilinestring(axes, gdf.geometry.iloc[nbi],
                                 linewidth=vals[nbi], color=color)
        else:
            plot_linestring(axes, gdf.geometry.iloc[nbi],
                            linewidth=vals[nbi], color=color)
    return axes


class Borderiz(object):
    def __init__(self, gdf_polygon):
            self.gdf = gdf_polygon.copy()

    def run(self, tol, col_name):
        self.multi_to_singles()
        self._buffer(tol)
        self._polygon_to_lines()
        self._buff_line_intersection(col_name)
        self._grep_border()
        return self.border

    def _polygon_to_lines(self):
        s_t = time.time()
        self.gdf.geometry = self.gdf.geometry.boundary
#        print('{:.2f}s'.format(time.time()-s_t))

    def multi_to_singles(self):
        """Return a new geodataframe where each feature is a single-geometry"""
        values = self.gdf[[i for i in self.gdf.columns if i != 'geometry']]
        geom = self.gdf.geometry
        geoms, attrs = [], []
        for i in range(len(self.gdf)-1):
            try:
                for single_geom in geom.iloc[i]:
                    geoms.append(single_geom)
                    attrs.append(values.iloc[i])
            except:
                geoms.append(geom.iloc[i])
                attrs.append(values.iloc[i])
        self.gdf = gpd.GeoDataFrame(
            attrs, geometry=geoms,
            columns=[i for i in self.gdf.columns if i != 'geometry'],
            index=pd.Int64Index([i for i in range(len(geoms))]))

    def _buffer(self, tol):
        s_t = time.time()
        self.buffered = self.gdf.copy()
        self.buffered.geometry = self.buffered.geometry.buffer(tol)
#        print('{:.2f}s'.format(time.time()-s_t))

    def _buff_line_intersection(self, col_name):
        s_t = time.time()
        resgeom, resattrs = [], []
        resgappd = resgeom.append
        resaappd = resattrs.append
        res = self.intersects_table()
        for i, _ in enumerate(res):
            fti = self.gdf.iloc[i]
            i_geom = self.gdf.geometry.iloc[i]
            for j in res[i]:
                ftj = self.buffered.iloc[j]
                j_geom = self.buffered.geometry.iloc[j]
                tmp = i_geom.intersection(j_geom)
                if 'Collection' not in tmp.geom_type:
                    resgappd(i_geom.intersection(j_geom))
                    resaappd(
                        (fti[col_name]+'-'+ftj[col_name],
                         ftj[col_name]+'-'+fti[col_name]))
                else:
                    pass
        self.result = gpd.GeoDataFrame(
            resattrs, geometry=resgeom,
            columns=['FRONT', 'FRONT_r'],
            index=pd.Int64Index([i for i in range(len(resattrs))])
            )
#        self.result = self.result.ix[[not self.result.geometry[i].is_ring
#                                      for i in range(len(self.result)-1)]]
#        print(
#            '{:.2f}s ({} features)'.format(time.time()-s_t, len(self.result))
#            )

    def _filt(self, ref):
        for ii in range(len(self.result)):
            if self.result.iloc[ii]['FRONT_r'] in ref['FRONT'] \
                    and self.result.iloc[ii]['FRONT'] != self.result.iloc[ii]['FRONT_r']:
                yield self.result.iloc[ii]

    def _grep_border(self):
        s_t = time.time()
        self.border = []
        seen = {}
        for i in range(len(self.result)):
            fti = self.result.iloc[i]
            for j in self._filt(fti):
                try:
                    puidj = str(round(j.geometry.length, -2))
                    if j['FRONT'] + puidj not in seen \
                            and j['FRONT_r'] + puidj not in seen:
                        key1 = j['FRONT'] + puidj
                        key2 = j['FRONT_r'] + puidj
                        seen[key1] = 1
                        seen[key2] = 1
                        self.border.append(j)
                except TypeError as err:
                    print(err)
                    pass
        self.border = gpd.GeoDataFrame(self.border)
#        print('{:.2f}s'.format(time.time()-s_t))
#        print('len(seen) : ', len(seen), ' | len(border) : ', len(self.border))

    def intersects_table(self):
        """
        Return a table with a row for each features of g1, each one containing
        the id of each g2 intersecting features
        """
        return self.gdf.geometry.apply(
            lambda x: [i for i in range(len(self.buffered.geometry))
                       if x.intersects(self.buffered.iloc[i].geometry)]
            )
