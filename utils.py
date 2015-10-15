# -*- coding: utf-8 -*-
"""
@author: mthh
"""

import time
import math
import rtree
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from sklearn.metrics.pairwise import pairwise_distances


def nrepeat(iterable, n):
    return iter([i for i in iterable for j in range(n)])


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
    import math
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

class Borderiz(object):
    def __init__(self, gdf_polygon):
            self.gdf = gdf_polygon.copy()
#    def __init__(self, polygon_layer_path):
#            self.gdf = gpd.GeoDataFrame.from_file(polygon_layer_path)

    def run(self, tol, col_name):
        self.multi_to_singles()
        self._buffer(tol)
        self._polygon_to_lines()
        self._buff_line_intersection(col_name)
        self._grep_border()
        return self.border

    def run_v(self, tol, col_name):
        print('MultiPolygons to Polygons..')
        self.multitosingle()
        print('Bufferize the polygons..')
        self._buffer(tol)
        print('(Non bufferized) Polygons to MultiLineStrings..')
        self._polygon_to_lines()
        print('Intersection of MultiLineStrings and buffers...')
        self._buff_line_intersection(col_name)
        print('Finding borders..')
        self._grep_border()

    def _polygon_to_lines(self):
        s_t = time.time()
        self.gdf.geometry = self.gdf.geometry.boundary
        print('{:.2f}s'.format(time.time()-s_t))

    def multi_to_singles(self):
        """Return a new geodataframe where each feature is a single-geometry"""
        values = self.gdf[[i for i in self.gdf.columns if i != 'geometry']]
        geom = self.gdf.geometry
        geoms, attrs = [], []
        for i in range(len(self.gdf)-1):
            try:
                for single_geom in geom[i]:
                    geoms.append(single_geom)
                    attrs.append(values.iloc[i])
            except:
                geoms.append(geom[i])
                attrs.append(values.iloc[i])
        self.gdf = gpd.GeoDataFrame(
            attrs, geometry=geoms,
            columns=[i for i in self.gdf.columns if i != 'geometry'],
            index=pd.Int64Index([i for i in range(len(geoms))]))

    def _buffer(self, tol):
        s_t = time.time()
        self.buffered = self.gdf.copy()
        self.buffered.geometry = self.buffered.geometry.buffer(tol)
        print('{:.2f}s'.format(time.time()-s_t))

    def _buff_line_intersection(self, col_name):
        s_t = time.time()
        resgeom, resattrs = [], []
        resgappd = resgeom.append
        resaappd = resattrs.append
        res = self.intersects_table()
#        print(res)
        for i, _ in enumerate(res):
            fti = self.gdf.iloc[i]
            i_geom = self.gdf.geometry.iloc[i]
#            print(res[i])
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
        print(
            '{:.2f}s ({} features)'.format(time.time()-s_t, len(self.result))
            )

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
        print('{:.2f}s'.format(time.time()-s_t))
        print('len(seen) : ', len(seen), ' | len(border) : ', len(self.border))

    def intersects_table(self):
        """
        Return a table with a row for each features of g1, each one containing
        the id of each g2 intersecting features
        """
        return self.gdf.geometry.apply(
            lambda x: [i for i in range(len(self.buffered.geometry))
                       if x.intersects(self.buffered.iloc[i].geometry)]
            )
