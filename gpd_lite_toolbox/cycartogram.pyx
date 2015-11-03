# -*- coding: utf-8 -*-
#cython: boundscheck = False
#cython: wraparound = False
#cython: cdivision = True
"""
 cycartogram extension:

    Easy construction of continuous cartogram on a Polygon/MultiPolygon
    GeoDataFrame (modify the geometry in place or create a new GeoDataFrame).

    Code adapted (to fit the geopandas.GeoDataFrame datastructure) from
    Carson Farmer's code (https://github.com/carsonfarmer/cartogram which was
    part of 'Cartogram' QGis python plugin), itself partially related to
    'pyCartogram.py' from Eric Wolfs.

    Algorithm itself based on :
    ```
    Dougenik, J. A, N. R. Chrisman, and D. R. Niemeyer. 1985.
    "An algorithm to construct continuous cartograms."
    Professional Geographer 37:75-81
    ```

    No warranty concerning the result.
    Copyright (C) 2013 Carson Farmer, 2015  mthh
"""
from shapely.geometry import LineString, Polygon, MultiPolygon
from libc.math cimport sqrt
from cpython cimport array
from libc.stdlib cimport malloc, free


def make_cartogram(geodf, field_name, iterations=5, inplace=False):
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
    GeoDataFrame: A new GeoDataFrame (or None if inplace=True)

    References
    ----------
    ``Dougenik, J. A, N. R. Chrisman, and D. R. Niemeyer. 1985.
    "An algorithm to construct continuous cartograms."
    Professional Geographer 37:75-81``
    """
    assert isinstance(iterations, int) and iterations > 0, \
        "Iteration number have to be a positive integer"
    try:
        f_idx = geodf.columns.get_loc(field_name)
    except KeyError:
        raise KeyError('Column name \'{}\' not found'.format(field_name))

    if inplace:
        Cartogram(geodf, f_idx, iterations).make()
    else:
        return Cartogram(geodf.copy(), f_idx, iterations).make()


ctypedef public struct Holder:
    unsigned int lFID
    double ptCenter_x
    double ptCenter_y
    double dValue
    double dArea
    double dMass
    double dRadius

cdef class Cartogram(object):
    cdef object geodf, temp_geo_serie
    cdef unsigned int iterations, total_features
    cdef float dForceReductionFactor
    cdef Holder *aLocal
    cdef double[:] values

    def __init__(self, object geodf not None, int field_idx, unsigned int iterations):
        cdef set geom_type, allowed = {'MultiPolygon', 'Polygon'}

        geom_type = set(list(geodf.geom_type))
        if not geom_type.issubset(allowed):
            raise ValueError(
                "Geometry type doesn't match 'Polygon'/'MultiPolygon"
                )
        self.geodf = geodf
        self.temp_geo_serie = geodf.geometry[:]
        self.iterations = iterations
        self.total_features = <unsigned int>len(self.geodf)
        self.dForceReductionFactor = 0
        self.values = geodf[[field_idx]].values.T[0]
        self.aLocal = <Holder *>malloc(self.total_features * sizeof(Holder))
        if not self.aLocal:
            raise MemoryError()

    cpdef object make(self):
        """Fetch the result and make it available"""

        self.cartogram()
        self.geodf.set_geometry(self.temp_geo_serie, inplace=True)
        free(self.aLocal)
        return self.geodf

    cdef object cartogram(self):
        """
        Compute for transformation
        (recursively, according to the specified iteration number)
        """
        cdef unsigned int ite=0, nbi=0
        
        for ite in range(self.iterations):
            self.getinfo()
            for nbi in range(self.total_features):
                self.temp_geo_serie[nbi] = self.transform_geom(
                    self.temp_geo_serie[nbi]
                    )

    cdef void getinfo(self):
        """
        Gets the information required for calcualting size reduction factor
        """
        cdef unsigned int fid=0, i, featCount = self.total_features
        cdef float dPolygonValue, dPolygonArea, dFraction, dDesired, dRadius
        cdef float dSizeError=0.0, dMean, pi=3.14159265
        cdef float area_total, value_total, tmp, dSizeErrorTotal = 0.0

        area_total = sum(self.temp_geo_serie.area)
        value_total = sum(self.values)
        for fid in range(featCount):
            geom = self.temp_geo_serie.iloc[fid]
            self.aLocal[fid].dArea = geom.area  # save area of this feature
            self.aLocal[fid].lFID = fid  # save id for this feature
            # save weighted 'area' value for this feature :
            self.aLocal[fid].dValue = self.values[fid]
            # save centroid coord for the feature :
            (self.aLocal[fid].ptCenter_x, self.aLocal[fid].ptCenter_y) = \
                (geom.centroid.coords.ctypes[0], geom.centroid.coords.ctypes[1])

        dFraction = area_total / value_total
        with nogil:
            for i in range(featCount):
                dPolygonValue = self.aLocal[i].dValue
                dPolygonArea = self.aLocal[i].dArea
                if dPolygonArea < 0:  # area should never be less than zero
                    dPolygonArea = 0
                # this is our 'desired' area...
                dDesired = dPolygonValue * dFraction
                # calculate radius, a zero area is zero radius
                dRadius = sqrt(dPolygonArea / pi)
                self.aLocal[i].dRadius = dRadius
                tmp = dDesired / pi
                if tmp > 0:
                    # calculate area mass, don't think this should be negative
                    self.aLocal[i].dMass = sqrt(dDesired / pi) - dRadius
                else:
                    self.aLocal[i].dMass = 0
                # both radius and mass are being added to the feature list for
                # later on...
                # calculate size error...
                dSizeError = \
                    max(dPolygonArea, dDesired) / min(dPolygonArea, dDesired)
                # this is the total size error for all polygons
                dSizeErrorTotal += dSizeError
        # average error
        dMean = dSizeErrorTotal / featCount
        # need to read up more on why this is done
        self.dForceReductionFactor = 1 / (dMean + 1)

    cdef object transform_geom(self, object geom,
                               Polygon=Polygon, MultiPolygon=MultiPolygon,
                               LineString=LineString):
        """
        Core function computing the transformation on the Polygon (or on each
            polygon, if multipolygon layer), using previously retieved informations
            about its geometry and about other feature geometries.
        """
        cdef unsigned int i, k, it_geom=0, it_bound=0, l_coord_bound=0
        cdef double x, y, x0, y0, cx, cy, distance, Fij, xF
        cdef Py_ssize_t nb_geom, nb_bound
        cdef Holder *lf
        cdef object boundarys
        cdef double[:] xs, ys
        cdef list tmp_bound, new_geom = []

        if isinstance(geom, Polygon):
            geom = [geom]
            nb_geom = 1
        else:
            nb_geom = len(geom)
        for it_geom in range(nb_geom):
            boundarys = geom[it_geom].boundary
            tmp_bound = []
            try:
                nb_bound = <unsigned int>len(boundarys)
            except:
                boundarys = [boundarys]
                nb_bound = 1
            for it_bound in range(nb_bound):
                line_coord = []
                xs, ys = boundarys[it_bound].coords.xy
                l_coord_bound = <unsigned int>len(xs)
                with nogil:
                    for k in range(l_coord_bound):
                        x = xs[k]
                        y = ys[k]
                        x0, y0 = x, y
                        # Compute the influence of all shapes on this point
                        for i in range(self.total_features):
                            lf = &self.aLocal[i]
                            cx = lf.ptCenter_x
                            cy = lf.ptCenter_y
                            # Pythagorean distance
                            distance = sqrt((x0 - cx) ** 2 + (y0 - cy) ** 2)
                            if distance > lf.dRadius:
                                # Calculate the force on verteces far away
                                # from the centroid of this feature
                                Fij = lf.dMass * lf.dRadius / distance
                            else:
                                # Calculate the force on verteces far away
                                # from the centroid of this feature
                                xF = distance / lf.dRadius
                                Fij = lf.dMass * (xF ** 2) * (4 - (3 * xF))
                            Fij = Fij * self.dForceReductionFactor / distance
                            x = (x0 - cx) * Fij + x
                            y = (y0 - cy) * Fij + y
                        with gil:
                            line_coord.append((x, y))
                tmp_bound.append(line_coord)

            if nb_bound == 1:
                new_geom.append(Polygon(tmp_bound[0]))
            else:
                for it_bound in range(nb_bound):
                    new_geom.append(Polygon(tmp_bound[it_bound]))

        if nb_geom > 1:
            return MultiPolygon(new_geom)
        elif nb_geom == 1:
            return new_geom[0]
 
