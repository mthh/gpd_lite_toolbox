# -*- coding: utf-8 -*-
"""
Geopandas lite toolbox
"""
import shapely.ops
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from geopandas import GeoDataFrame
from sklearn.metrics.pairwise import pairwise_distances

from cycartogram import Cartogram
from utils import (
    db_creation, db_connect, hav_dist, Borderiz,
    make_index, nrepeat, mparams
    )

__all__ = ['get_borders', 'find_borders', 'transform_cartogram', 'dissolve',
           'intersects_byid', 'multi_to_single', 'dumb_multi_to_single',
           'snap_to_nearest', 'read_spatialite', 'match_lines']


def match_lines(gdf1, gdf2, method='cheap_hausdorff', limit=None):
    """
    Return a Serie (with the length of gdf1) with each row containing the
    id of the matching feature in gdf2 (i.e the closest based on the
    computation of a "hausdorff-distance-like" between the two lines)
    or nothing 
    If a limit is passed in arguments, features situed far from this distance
    will not be taken into account (in order to avoid retrieving the id of 
    too far located segments, even if the closest when no one seems to
    be matching).
    """
    if 'cheap_hausdorff' in method:
        if limit :
            return (gdf1.geometry.apply(
                lambda x: [fh_dist_lines(x, gdf2.geometry[i]) for i in range(len(gdf2))]
                )).apply(lambda x: [nb for nb, i in enumerate(x) if i == min(x) and i < limit])
        else :
            return (gdf1.geometry.apply(
                lambda x: [fh_dist_lines(x, gdf2.geometry[i]) for i in range(len(gdf2))]
                )).apply(lambda x: [nb for nb, i in enumerate(x) if i == min(x)])

    elif 'cluster' in method:
        return match_line_cluster(gdf1, gdf2)

    else:
        raise ValueError('Incorrect matching method\nMethod should '
                         'be \'cheap_hausdorff\' or \'cluster\'.')


def match_line_cluster(gdf1, gdf2):
    """
    Try to match two layers of linestrings with KMeans cluster analysis based
    on a triplet of descriptive attributes : 
    (centroid coords., rounded length, approximate bearing)
    """
    p1, p2 = list(map(mparams, [gdf1, gdf2]))
    k_means = KMeans(init='k-means++', n_clusters=len(gdf1), n_init=10, max_iter=1000)
    k_means.fit(np.array((p1+p2)))
    df1 = pd.Series(k_means.labels_[len(gdf1):])
    df2 = pd.Series(k_means.labels_[len(gdf1):])
#    gdf1['fid_layer2'] = \
#        df1.apply(lambda x: df2.where(gdf2['key'] == x).notnull().nonzero()[0][0])
    return pd.DataFrame(
        index=list(range(len(gdf1))),
        data=df1.apply(
            lambda x: df2.where(df2 == x).notnull().nonzero())
        )


def fh_dist_lines(li1, li2):
    """
    Compute a cheap distance (based on hausdorff-distance) between two lines.
    """
    c1 = np.array([i for i in zip(li1.coords.xy[0], li1.coords.xy[1])])
    c2 = np.array([i for i in zip(li2.coords.xy[0], li2.coords.xy[1])])
    if len(c2) > len(c1):
        c1, c2 = c2, c1
    dist_mat = pairwise_distances(c1, c2, metric='euclidean', n_jobs=1)
    chkl = round(len(c1)/len(c2))
    return max([dist_mat[i, j]
                for i, j in zip(list(range(len(c1))),
                                list(nrepeat(range(len(c2)), chkl))[:len(c1)])])

def get_borders(gdf, tol, col_name='id'):
    """
    Return lines corresponding to the border between each polygon of the
    dataset, each line containing the id of the two polygon around
    (quicker computation than find_borders).

    Likely a minimalist python port of cartography::getBorders R function from
    https://github.com/Groupe-ElementR/cartography/blob/master/R/getBorders.R
    """
    buff = gdf.geometry.buffer(tol)
    intersect_table = intersects_byid(buff, buff)
    attr, new_geoms = [], []
    for i in range(len(gdf)-1):
        tmp1 = gdf.iloc[i]
        buff_geom1 = buff[i]
        for j in intersect_table[i]:
            if not i == j:
                tmp2 = gdf.iloc[j]
                buff_geom2 = buff[j]
                new_geoms.append(
                    (buff_geom1.intersection(buff_geom2)).boundary
                    )
                attr.append(tmp1[col_name] + '-' + tmp2[col_name])
    return GeoDataFrame(attr, geometry=new_geoms, columns=[col_name])


def find_borders(gdf, tol, col_name):
    """
    Return lines corresponding to the border between each polygon of the
    dataset, each line containing the id of the two polygon around it.
    This function is like the slower and complicated version of get_borders()
    """
    if col_name not in gdf.columns:
        raise ValueError("Column name error : can't find {}".format(col_name))
    bor = Borderiz(gdf)
    return bor.run(tol, col_name)


def transform_cartogram(geodf, field_name, iterations=5, inplace=False):
    """
    Make a continuous cartogram on a geopandas.GeoDataFrame collection
    of Polygon/MultiPolygon (wrapper to call the core functions
    written in cython).

    :param geopandas.GeoDataFrame geodf: The GeoDataFrame containing the
        geometry and a field to use for the transformation.

    :param string field_name: The name of the field (Series) containing the
        value to use.

    :param integer iterations: The number of iterations to make.
        [default=5]

    :param bool inplace: Append in place if True is set. Otherwhise return a
        new GeoDataFrame with transformed geometry.
        [default=False]
    """
    assert isinstance(iterations, int) and iterations > 0, \
        "Iteration number have to be a positive integer"
    assert field_name in geodf.columns
#    assert all(geodf.geometry.is_valid)
    if inplace:
        crtgm = Cartogram(geodf, field_name, iterations)
        crtgm.make()
    else:
        crtgm = Cartogram(geodf.copy(), field_name, iterations)
        return crtgm.make()


def intersects_byid(geoms1, geoms2):
    """
    Return a table with a row for each features of g1, each one containing
    the id of each g2 intersecting features
    """
    return geoms1.geometry.apply(
        lambda x: [i for i in range(len(geoms2.geometry))
                   if x.intersects(geoms2.geometry[i])]
        )


def dissolve(gdf, colname):
    """
    Return a new geodataframe with
    dissolved features around the selected columns
    """
    df2 = gdf.groupby(colname)
    gdf.set_index(colname, inplace=True)
    gdf['geometry'] = df2.geometry.apply(shapely.ops.unary_union)
    gdf.reset_index(inplace=True)
    return gdf.drop_duplicates(colname)


def multi_to_single(gdf):
    """Return a new geodataframe where each feature is a single-geometry"""
    values = gdf[[i for i in gdf.columns if i != 'geometry']]
    geom = gdf.geometry
    geoms, attrs = [], []
    for i in range(len(gdf)-1):
        try:
            for single_geom in geom.iloc[i]:
                geoms.append(single_geom)
                attrs.append(values.iloc[i])
        except:
            geoms.append(geom.iloc[i])
            attrs.append(values.iloc[i])
    return GeoDataFrame(attrs, index=[i for i in range(len(geoms))],
                        geometry=geoms,
                        columns=[i for i in gdf.columns if i != 'geometry'])


def snap_to_nearest(pts_ref, target_layer, inplace=False,
                    searchframe=0.3, max_searchframe=0.8):
    new_geoms = pts_ref.geometry.values.copy()
    target_geoms = target_layer.geometry.values
    start_buff = searchframe
    index = make_index([i.bounds for i in target_geoms])

    for id_pts_ref in range(len(new_geoms)):
        while True:
            try:
                tmp = {
                    (new_geoms[id_pts_ref].distance(target_geoms[fid])): fid
                    for fid in list(index.intersection(
                        new_geoms[id_pts_ref].buffer(searchframe).bounds,
                        objects='raw'))
                }
                road_ref = tmp[min(tmp.keys())]
                break
            except ValueError as err:
                searchframe += (max_searchframe-start_buff)/3
                if searchframe > max_searchframe:
                    break
        try:
            res = {new_geoms[id_pts_ref].distance(Point(x, y)): Point(x, y)
                   for x, y in zip(*target_geoms[road_ref].coords.xy)}
            new_geoms[id_pts_ref] = res[min(res.keys())]
        except NameError as err:
            print(err, 'No value for {}'.format(id_pts_ref))

    if inplace:
        pts_ref.set_geometry(new_geoms, drop=True, inplace=True)
    else:
        result = pts_ref.copy()
        result.set_geometry(new_geoms, drop=True, inplace=True)
        return result


def dumb_multi_to_single(gdf):
    """
    A "dumb" (but sometimes usefull) multi-to-single function, only returning
    the first single geometry of each multi-part geometry.
    """
    values = gdf[[i for i in gdf.columns if i != 'geometry']]
    geom = gdf.geometry
    geoms, attrs = [], []
    for i in range(len(gdf)-1):
        try:
            for single_geom in geom.iloc[i]:
                geoms.append(single_geom)
                attrs.append(values.iloc[i])
                break
        except:
            geoms.append(geom.iloc[i])
            attrs.append(values.iloc[i])
    return GeoDataFrame(attrs, index=[i for i in range(len(geoms))],
                        geometry=geoms,
                        columns=[i for i in gdf.columns if i != 'geometry'])


def read_spatialite(sql, conn, geom_col='geometry', crs=None,
                    index_col=None, coerce_float=True, params=None,
                    db_path=None):
    """
    Wrapper of read_postgis() function, allowing to read from spatialite
    without overhead.
    """
    from geopandas import read_postgis
    if '*' in sql:
        raise ValueError('Column names have to be specified')

    if not conn and db_path:
        conn = db_connect(db_path)
    elif not conn:
        raise ValueError(
            'A connection object or a path to the DB have to be provided')

    if sql.lower().find('select') == 0 and sql.find(' ') == 6:
        sql = sql[:7] \
            + "HEX(ST_AsBinary({0})) as {0}, ".format(geom_col) + sql[7:]
    else:
        raise ValueError(
            'Unable to understand the query')

    return read_postgis(
        sql, conn, geom_col=geom_col, crs=crs, index_col=index_col,
        coerce_float=coerce_float, params=params
        )


def mean_coordinates(gdf, id_field=None, weight_field=None):
    """
    Compute the mean coordinate(s) of a set of points. If a *weight-field*
    (numerical field) is provided, the point(s) will be located according it.
    If an *id_field* is given, a mean coordinate pt will be calculated for each
    subset of points differencied by this *id_field*.

    :param gdf:

    :param str id_field:

    :param str weight_field:

    Return a new GeoDataFrame with the location of computed point(s).
    """
    assert 'Multi' not in gdf.geometry.geom_type, \
        "Multipart geometries aren't allowed"
    fields = ['geometry']
    if id_field:
        assert id_field in gdf.columns
        fields.append(id_field)
    if weight_field:
        assert weight_field in gdf.columns
        fields.append(weight_field)
    else:
        weight_field = 'count'
    tmp = gdf[fields].copy()
    tmp['x'] = tmp.geometry.apply(lambda x: x.coords.xy[0][0])
    tmp['y'] = tmp.geometry.apply(lambda x: x.coords.xy[1][0])
    tmp.x = tmp.x * tmp[weight_field]
    tmp.y = tmp.y * tmp[weight_field]
    tmp['count'] = 1
    if id_field:
        tmp = tmp.groupby(id_field).sum()
    else:
        tmp = tmp.sum()
        tmp = tmp.T
    tmp.x = tmp.x / tmp[weight_field]
    tmp.y = tmp.y / tmp[weight_field]
    tmp['geometry'] = [Point(i[0], i[1]) for i in tmp[['x', 'y']].values]
    return GeoDataFrame(tmp[weight_field], geometry=tmp['geometry'],
                        index=tmp.index).reset_index()
