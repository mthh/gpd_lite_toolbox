# -*- coding: utf-8 -*-
"""
Geopandas lite toolbox
"""
import shapely.ops
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from shapely.geometry import Point, Polygon
from geopandas import GeoDataFrame
from sklearn.metrics.pairwise import pairwise_distances
#from sklearn.preprocessing import normalize

from cycartogram import Cartogram
from utils import (
    db_connect, Borderiz, dbl_range,
    make_index, nrepeat, mparams, dorling_radius
    )

__all__ = ['get_borders', 'find_borders', 'transform_cartogram', 'dissolve',
           'intersects_byid', 'multi_to_single', 'dumb_multi_to_single',
           'snap_to_nearest', 'read_spatialite', 'match_lines',
           'non_contiguous_cartogram', 'make_grid', 'random_pts_on_surface']


def match_lines(gdf1, gdf2, method='cheap_hausdorff', limit=None):
    """
    Return a pandas.Series (with the length of *gdf1*) with each row containing
    the id of the matching feature in *gdf2* (i.e the closest based on the
    computation of a "hausdorff-distance-like" between the two lines or
    the most similar based on some geometry properties) or nothing if nothing
    is found according to the *limit* argument.

    If a *limit* is given, features situed far from this distance
    will not be taken into account (in order to avoid retrieving the id of
    too far located segments, even if the closest when no one seems to
    be matching).

    Parameters
    ----------
    gdf1: GeoDataFrame of LineStrings (the reference dataset).
    gdf2: GeoDataFrame of LineStrings (the dataset to match).
    limit: Integer
        The maximum distance, where it is sure that segments
        couldn't match.

    Returns
    -------
    match_table: pandas.Series containing the matching table (with index
        based on *gdf1*)
    """
    if 'cheap_hausdorff' in method:
        if limit:
            return (gdf1.geometry.apply(
                lambda x: [fh_dist_lines(x, gdf2.geometry[i]) for i in range(len(gdf2))]
                )).apply(lambda x: [nb for nb, i in enumerate(x) if i == min(x) and i < limit])
        else:
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

    Parameters
    ----------
    gdf1: GeoDataFrame
        The reference dataset.
    gdf2: GeoDataFrame
        The collection of LineStrings to match.

    Returns
    -------
    matching_table: pandas.Series
        A table (index-based on *gdf1*) containing the id of the matching
        feature found in *gdf2*.
    """
    param1, param2 = list(map(mparams, [gdf1, gdf2]))
    k_means = KMeans(init='k-means++', n_clusters=len(gdf1),
                     n_init=10, max_iter=1000)
    k_means.fit(np.array((param1+param2)))
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
    Compute a cheap distance (based on hausdorff-distance) between
    *li1* and *li2*, two LineString.

    Parameters
    ----------
    li1: shapely.geometry.LineString
    li2: shapely.geometry.LineString

    Returns
    -------
    max_dist: Float of the distance between li1 and li2.

    """
    coord_li1 = np.array([i for i in zip(li1.coords.xy[0], li1.coords.xy[1])])
    coord_li2 = np.array([i for i in zip(li2.coords.xy[0], li2.coords.xy[1])])
    if len(coord_li2) > len(coord_li2):
        coord_li1, coord_li2 = coord_li2, coord_li1
    dist_mat = pairwise_distances(
        coord_li1, coord_li2, metric='euclidean', n_jobs=2
        )
    chkl = round(len(coord_li1)/len(coord_li2))
    return max(
        [dist_mat[i, j] for i, j in zip(
            list(range(len(coord_li1))),
            list(nrepeat(range(len(coord_li2)), chkl))[:len(coord_li1)])]
        )


def get_borders(gdf, tol=1, col_name='id'):
    """
    Get the lines corresponding to the border between each
    polygon from the dataset, each line containing the *col_name* of the
    two polygons around (quicker computation than :py:func:`find_borders`).
    Likely a minimalist python port of cartography::getBorders R function from
    https://github.com/Groupe-ElementR/cartography/blob/master/R/getBorders.R

    Parameters
    ----------
    gdf: :py:class: `geopandas.GeoDataFrame`
        Input collection of polygons.
    tol: int, default=1
        The tolerance (in units of :py:obj:`gdf`).
    col_name: str, default='id'
        The field name of the polygon to yield.

    Returns
    -------
    borders: GeoDataFrame
        A GeoDataFrame of linestrings corresponding to the border between each
        polygon from the dataset, each line containing the *col_name* of the
        two polygon around.
    """
    buff = gdf.geometry.buffer(tol)
    intersect_table = intersects_byid(buff, buff)
    attr, new_geoms = [], []
    for i in range(len(gdf)):
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


def find_borders(gdf, tol=1, col_name='id'):
    """
    Parameters
    ----------
    gdf: :py:class::`geopandas.GeoDataFrame`
        Input collection of polygons.
    tol: int, default=1
        The tolerance (in units of :py:obj:`gdf`).
    col_name: str, default='id'
        The field name of the polygon to yield.

    Returns
    -------
    borders: GeoDataFrame
        Return lines corresponding to the border between each polygon of the
        dataset, each line containing the id of the two polygon around it.
        This function is slower/more costly than :py:func:`get_borders`.
    """
    if col_name not in gdf.columns:
        raise ValueError("Column name error : can't find {}".format(col_name))
    bor = Borderiz(gdf)
    return bor.run(tol, col_name)


def transform_cartogram(gdf, field_name, iterations=5, inplace=False):
    """
    Make a continuous cartogram on a geopandas.GeoDataFrame collection
    of Polygon/MultiPolygon (wrapper to call the core functions
    written in cython).
    Based on the transformation of Dougenik and al.(1985).

    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
        The GeoDataFrame containing the geometry and a field to use for the
        transformation.
    field_name: String
        The label of the field containing the value to use.
    iterations: Integer, default 5
        The number of iteration to make.
    inplace, Boolean, default False
        Append in place if True. Otherwhise return a new :py:obj:GeoDataFrame
        with transformed geometry.

    Returns
    -------
    GeoDataFrame: A new GeoDataFram (or None if inplace=True)

    References
    ----------
    ``Dougenik, J. A, N. R. Chrisman, and D. R. Niemeyer. 1985.
    "An algorithm to construct continuous cartograms."
    Professional Geographer 37:75-81``
    """
    assert isinstance(iterations, int) and iterations > 0, \
        "Iteration number have to be a positive integer"
    assert field_name in gdf.columns
#    assert all(gdf.geometry.is_valid)
    if inplace:
        crtgm = Cartogram(gdf, field_name, iterations)
        crtgm.make()
    else:
        crtgm = Cartogram(gdf.copy(), field_name, iterations)
        return crtgm.make()


def intersects_byid(geoms1, geoms2):
    """
    Return a table with a row for each features of *geoms1*, containing the id
    of each *geoms2* intersecting features (almost like an intersecting matrix).

    Parameters
    ----------
    geoms1: GeoSeries or GeoDataFrame
        Collection on which the intersecting table will be based.
    geoms2: GeoSeries or GeoDataFrame
        Collection to test on intersect.

    Returns
    -------
    intersect_table: pandas.Series
        A Series with the same index id as geoms1, each row containg the ids of
        the features of geoms2 intersecting it.
    """
    return geoms1.geometry.apply(
        lambda x: [i for i in range(len(geoms2.geometry))
                   if x.intersects(geoms2.geometry[i])]
        )


def dissolve(gdf, colname):
    """
    Parameters
    ----------
    gdf: GeoDataFrame
        The geodataframe to dissolve
    colname: String
        The label of the column containg the common values to use to dissolve
        the collection.

    Returns
    -------
    Return a new :py:obj:`geodataframe` with
    dissolved features around the selected columns.
    """
#    gdf = gdf.copy()
    df2 = gdf.groupby(colname)
    gdf.set_index(colname, inplace=True)
    gdf['geometry'] = df2.geometry.apply(shapely.ops.unary_union)
    gdf.reset_index(inplace=True)
    return gdf.drop_duplicates(colname)


def multi_to_single(gdf):
    """
    Return a new geodataframe with exploded geometries (where each feature
    has a single-part geometry).

    Parameters
    ----------
    gdf: GeoDataFrame
        The input GeoDataFrame to explode to single part geometries.

    Returns
    -------
    gdf: GeoDataFrame
        The exploded result.

    See-also
    --------
    The method GeoDataFrame.explode() in recent versions of **geopandas**.
    """
    values = gdf[[i for i in gdf.columns if i != 'geometry']]
    geom = gdf.geometry
    geoms, attrs = [], []
    for i in range(len(gdf)):
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
                    searchframe=50, max_searchframe=500):
    """
    Snap each point from :py:obj:`pts_ref` on the nearest
    line-segment/polygon-vertex of :py:obj:`target_layer` according to a
    *searchframe* defined in units of both two input layers.
    Append inplace or return a new object.
    (A larger search frame can be set in *max_searchframe* : the search frame
    will be progressivly increased from *searchframe* to *max_searchframe* in
    order to snap the maximum of points without using a large orginal search
    frame)

    Parameters
    ----------
    pts_ref: GeoDataFrame
        The collection of points to snap on *target_layer*.
    target_layer: GeoDataFrame
        The collection of LineString or Polygon on which *pts_ref* will be
        snapped, according to the *max_searchframe*.
    inplace: Boolean, default=False
        Append inplace or return a new GeoDataFrame containing moved points.
    searchframe: Integer or float, default=50
        The original searchframe (in unit of the two inputs GeoDataFrame),
        which will be raised to *max_searchframe* if there is no objects to
        snap on.
    max_searchframe: Integer or float, default=500
        The maximum searchframe around each features of *pts_ref* to search in.

    Returns
    -------
    snapped_pts: GeoDataFrame
        The snapped collection of points (or None if inplace=True, where
        points are moved in the original geodataframe).
    """
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
    A "dumb" (but sometimes usefull) multi-to-single function, returning a
    GeoDataFrame with the first single geometry of each multi-part geometry
    (and also return single geometry features untouched), so the returned
    GeoDataFrame will have the same number of features.

    Parameters
    ----------
    gdf: GeoDataFrame
        The input collection of features.

    Returns
    -------
    gdf: GeoDataFrame
        The exploded result.
    """
    values = gdf[[i for i in gdf.columns if i != 'geometry']]
    geom = gdf.geometry
    geoms, attrs = [], []
    for i in range(len(gdf)):
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
    Wrap :py:func:`geopandas.read_postgis()` and allow to read from spatialite.

    Returns
    -------
    gdf: GeoDataframe

    Exemple
    -------
    >>> # With a connection object (conn) already instancied :
    >>> gdf = read_spatialite("SELECT PK_UID, pop_t, gdp FROM countries", conn,
                              geom_col="GEOM")
    >>> # Without being already connected to the database :
    >>> gdf = read_spatialite("SELECT PK_UID, pop_t, gdp FROM countries", None,
                              geom_col="GEOM", db_path='/home/mthh/tmp/db.sqlite')
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
    Compute the (weighted) mean coordinate(s) of a set of points. If provided
    the point(s) will be located according to *weight_field* (numérical field).
    If an *id_field* is given, a mean coordinate pt will be calculated for each
    subset of points differencied by this *id_field*.

    Parameters
    ----------
    gdf: GeoDataFrame
        The input collection of Points.
    id_field: String, optional
        The label of the field containing a value to weight each point.
    weight_field: String, optional
        The label of the field which differenciate features of *gdf* in subsets
        in order to get multiples mean points returned.

    Returns
    -------
    mean_points: GeoDataFrame
        A new GeoDataFrame with the location of the computed point(s).
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


def random_pts_on_surface(gdf, coef=1, nb_field=None):
    """
    For each polygon, return a point (or a set of points, according to
    *nb_field* and *coef*), lying on the polygon surface.

    Parameters
    ----------
    gdf: GeoDataFrame
        A collection of polygons on which generate points.
    coef: Integer, default 1
        The multiplicant, which applies to each feature of *gdf*,
        If used, the values contained on *nb_field* will also be multiplicated.
    nb_field: String, optional
        The name of the field to read, containing an integer
        which will be used as the number of points to create.

    Returns
    -------
    rand_points: GeoDataFrame
        A collection of points, located on *gdf*, accordingly to *coef* and
        values contained in *nb_field* if used.

    """
    nb_ft = len(gdf)
    if nb_field:
        nb_pts = gdf[nb_field].values * coef
    else:
        nb_pts = np.array([coef for i in range(nb_ft)])
    res = []
    for i in range(nb_ft):
        pts_to_create = nb_pts[i]
        while True:
            (minx, miny, maxx, maxy) = gdf.geometry[i].bounds
            xpt = (maxx-minx) * np.random.random_sample((pts_to_create,)) + minx
            ypt = (maxy-miny) * np.random.random_sample((pts_to_create,)) + miny
            points = np.array([xpt, ypt]).T
            for pt_ in points:
                pt_geom = Point((pt_[0], pt_[1]))
                if gdf.geometry[i].contains(pt_geom):
                    res.append(pt_geom)
                    pts_to_create -= 1
            if pts_to_create == 0:
                break
    return res


def make_grid(gdf, height, cut=True):
    """
    Return a grid, based on the shape of *gdf* and on a *height* value (in
    units of *gdf*). If cut=False, the grid will not be intersected with gdf
    (i.e it makes a grid on the bounding-box of *gdf*).

    Parameters
    ----------
    gdf: GeoDataFrame
        The collection of polygons the be covered by the grid.
    height: Integer
        The dimension (will be used as height an width) of the ceils to create,
        in units of *gdf*.
    cut: Boolean, default True
        Cut the grid to fit the shape of *gdf* (ceil partially covering it will
        be truncated, see *preserve_ceil* parameter). If False, the returned
        grid fit the counding box of gdf.
    preserve_ceil: Boolean, default False
        Only used if cut=True, so only ceils who cover *gdf* will be returned
        but ceils who only partially cover it won't be cut.

    Returns
    -------
    grid: GeoDataFrame
        A collection of polygon.
    """
    from math import ceil
    from shapely.ops import unary_union
    xmin, ymin = [i.min() for i in gdf.bounds.T.values[:2]]
    xmax, ymax = [i.max() for i in gdf.bounds.T.values[2:]]
    rows = ceil((ymax-ymin) / height)
    cols = ceil((xmax-xmin) / height)

    x_left_origin = xmin
    x_right_origin = xmin + height
    y_top_origin = ymax
    y_bottom_origin = ymax - height

    res_geoms = []
    for countcols in range(cols):
        y_top = y_top_origin
        y_bottom = y_bottom_origin
        for countrows in range(rows):
            res_geoms.append((
                (x_left_origin, y_top), (x_right_origin, y_top),
                (x_right_origin, y_bottom), (x_left_origin, y_bottom)
                ))
            y_top = y_top - height
            y_bottom = y_bottom - height
        x_left_origin = x_left_origin + height
        x_right_origin = x_right_origin + height
    if cut:
        return GeoDataFrame(
            geometry=pd.Series(res_geoms).apply(lambda x: Polygon(x)),
            crs=gdf.crs
            ).intersection(unary_union(gdf.geometry))
    else:
        return GeoDataFrame(
            index=[i for i in range(len(res_geoms))],
            geometry=pd.Series(res_geoms).apply(lambda x: Polygon(x)),
            crs=gdf.crs
            )


def non_contiguous_cartogram(gdf, value, nrescales,
                             n_iter=2, tol=100, buff_kws={}):
    """
    Make a non-contiguous cartogram on a geopandas.GeoDataFrame collection
    of Polygon/MultiPolygon.

    Parameters
    ----------
    gdf: :py:obj:`geopandas.GeoDataFrame`
        The GeoDataFrame containing the geometry and a field to use
        for the transformation.
    field_name: String
        The name of the column of *gdf* containing the value to use.
    n_rescales: Integer
        The number of iterations to make, each one scaling down the radius of
        the circle in order to avoid overlapping.
    n_iter: Integer
        The number of iterations to make, within each scale ratio.
    tol: Integer
        The tolerance to consider for overlapping (an overlapping within the
        tolerance will not be considered as an overlapping).
    buf_kws: Dict
        A dict of parameter for the shapely function :py:func: buffer to choose
        the shape of obtained geometry (like square, octogone, circle).

    Returns
    -------
    cartogram: GeoDataFrame
        A new geodataframe with transformed geometry, ready to map.
    """
    ratios = [1 - i/nrescales for i in range(nrescales)]
    gdf2 = gdf.copy()
#    gdf2[value] = normalize(gdf2[value])[0]
    gdf2.geometry = gdf2.geometry.centroid
    for ratio in ratios:
        radius = dorling_radius(gdf, value, ratio)
        for n_time in range(n_iter):
            overlap_count = 0
            for idxa, fta, idxb, ftb in dbl_range(gdf2):
                dx = fta.geometry.coords.xy[0][0] \
                    - ftb.geometry.coords.xy[0][0]
                dy = fta.geometry.coords.xy[1][0] \
                    - ftb.geometry.coords.xy[1][0]
                l = np.sqrt(dx**2+dy**2)
                d = radius[idxa] + radius[idxb]
                prop = (l-d)/l
                dx = dx * prop
                dy = dy * prop
                if l < d and abs(l - d) > tol:
                    overlap_count += 1
                    gdf2.loc[idxa, 'geometry'] = Point(
                        (fta.geometry.coords.xy[0][0] - dx,
                         fta.geometry.coords.xy[1][0] - dy)
                        )
        if overlap_count == 0:
            break
    geoms = [gdf2.geometry[i].buffer(radius[i], **buff_kws)
             for i in range(len(gdf2))]
    gdf2['geometry'] = geoms
    return gdf2
