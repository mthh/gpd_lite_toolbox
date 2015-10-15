# -*- coding: utf-8 -*-
"""
 cartogram_geopandas v0.0.0c:

    Easy construction of continuous cartogram on a Polygon/MultiPolygon
    GeoDataFrame (modify the geometry in place or create a new GeoDataFrame).

    Code adapted to fit the geopandas.GeoDataFrame datastructure from
    Carson Farmer's code (https://github.com/carsonfarmer/cartogram : former
    code in use in 'Cartogram' QGis python plugin). Carson Farmer's code is
    partially related to 'pyCartogram.py' from Eric Wolfs.

    Algorithm itself based on
        { Dougenik, J. A, N. R. Chrisman, and D. R. Niemeyer. 1985.
          "An algorithm to construct continuous cartograms."
          Professional Geographer 37:75-81 }

    No warranty concerning the result.
    Copyright (C) 2013 Carson Farmer, 2015  mthh
"""

import math
from geopandas import GeoSeries
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon
from libc.math cimport sqrt
from cpython cimport array

cdef object transform_geom(list aLocal, float dForceReductionFactor,
                           object geom, int featCount):
    """
    Core function computing the transformation on the Polygon (or on each
        polygon, if multipolygon layer), using previously retieved informations
        about its geometry and about other feature geometries.
    """
    cdef size_t i, k
    cdef Holder lf
    cdef list new_geom, tmp_bound, line_coord
    cdef float x, y, x0, y0, cx, cy, distance, Fij, xF
    cdef array.array xs, ys
    cdef Py_ssize_t l_coord_bound
    
    new_geom = []
    if isinstance(geom, Polygon):
        geom = [geom]
    for single_geom in geom:
        boundarys = single_geom.boundary
        tmp_bound = []
        if not isinstance(boundarys, MultiLineString):
            boundarys = [boundarys]
        for single_boundary in boundarys:
            line_coord = []
#            line_add_pt = line_coord.append
            xs, ys = single_boundary.coords.xy
            l_coord_bound = len(xs)
            for k in range(l_coord_bound):
                x = xs[k]
                y = ys[k]
                x0, y0 = x, y
                # Compute the influence of all shapes on this point
                for i in range(featCount):
                    lf = aLocal[i]
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
                    Fij = Fij * dForceReductionFactor / distance
                    x = (x0 - cx) * Fij + x
                    y = (y0 - cy) * Fij + y
                line_coord.append((x, y))
#            line = LineString(line_coord)
#            tmp_bound.append(line) 
            tmp_bound.append(line_coord)

        if len(tmp_bound) == 1:
            poly = Polygon(tmp_bound[0])
        else:
            poly = MultiPolygon(
                [Polygon(sl) for sl in MultiLineString(tmp_bound)]
                )
        new_geom.append(poly)

    if len(new_geom) > 1:
        return MultiPolygon(new_geom)
    elif len(new_geom) == 1:
        return new_geom[0]


cdef class Holder(object):
    cdef int lFID
    cdef float ptCenter_x, ptCenter_y, dValue, dArea, dMass, dRadius

    def __init__(self):
        self.lFID = 0
        self.ptCenter_x = -1
        self.ptCenter_y = -1
        self.dValue = -1
        self.dArea = -1
        self.dMass = -1
        self.dRadius = -1


cdef class Cartogram(object):
    cdef object geodf
    cdef int iterations, index_field
    cdef Py_ssize_t total_features

    def __init__(self, object geodf, field_name, int iterations):
        cdef Py_ssize_t total_features
        cdef set allowed, geom_type

        allowed = {'MultiPolygon', 'Polygon'}
        geom_type = {i for i in geodf.geom_type}
        if not geom_type.issubset(allowed):
            raise ValueError(
                "Geometry type doesn't match 'Polygon'/'MultiPolygon"
                )
        self.geodf = geodf
        self.iterations = iterations
        self.index_field = [i for i, j in enumerate(list(self.geodf.columns))
                            if field_name in j][0]
        self.total_features = len(self.geodf)

    def make(self):
        """Fetch the result and make it available"""
        cdef int iterations_done
        cdef object res_geom

        res_geom, iterations_done = self.cartogram()
        assert iterations_done == self.iterations
        self.geodf.set_geometry(res_geom, inplace=True)
        return self.geodf

    def cartogram(self):
        """
        Compute for transformation
        (recursively, according to the specified iteration number)
        """
        cdef int iterations, total_features, ite=0
        cdef size_t nbi
        cdef list aLocal
        cdef float dForceReductionFactor
        cdef object temp_geo_serie

        total_features = self.total_features
        iterations = self.iterations
        temp_geo_serie = self.geodf.geometry.copy()

        for ite in range(iterations):
            (aLocal, dForceReductionFactor) = self.getinfo(self.index_field)

            for nbi in range(total_features):
                temp_geo_serie[nbi] = transform_geom(
                    aLocal, dForceReductionFactor,temp_geo_serie[nbi], total_features
                    )
        ite += 1
        return temp_geo_serie, ite

    def getinfo(self, int index, float pi=math.pi):
        """
        Gets the information required for calcualting size reduction factor
        """
        cdef int fid=0, i
        cdef float dPolygonValue, dPolygonArea, dFraction, dDesired, dRadius
        cdef float dSizeError=0, dSizeErrorTotal=0, dForceReductionFactor, dMean
        cdef float area_total, value_total, tmp
        cdef list aLocal
        cdef Holder lfeat, lf
        cdef Py_ssize_t featCount

        featCount = self.total_features
        aLocal = []
        area_total = sum(self.geodf.area)
        value_total = sum(self.geodf.iloc[:, index])
        for fid in range(featCount):
            geom = self.geodf.geometry[fid]
            lfeat = Holder()
            lfeat.dArea = geom.area  # save area of this feature
            lfeat.lFID = fid  # save id for this feature
            # save weighted 'area' value for this feature :
            lfeat.dValue = self.geodf.iloc[fid, index]
            # save centroid coord for the feature :
            (lfeat.ptCenter_x, lfeat.ptCenter_y) = \
                (geom.centroid.coords.ctypes[0], geom.centroid.coords.ctypes[1])
            aLocal.append(lfeat)

        dFraction = area_total / value_total

        for i in range(featCount):
            lf = aLocal[i]  # info for current feature
            dPolygonValue = lf.dValue
            dPolygonArea = lf.dArea
            if dPolygonArea < 0:  # area should never be less than zero
                dPolygonArea = 0
            # this is our 'desired' area...
            dDesired = dPolygonValue * dFraction
            # calculate radius, a zero area is zero radius
            dRadius = sqrt(dPolygonArea / pi)
            lf.dRadius = dRadius
            tmp = dDesired / pi
            if tmp > 0:
                # calculate area mass, don't think this should be negative
                lf.dMass = sqrt(dDesired / pi) - dRadius
            else:
                lf.dMass = 0
            # both radius and mass are being added to the feature list for
            # later on...
            # calculate size error...
            dSizeError = \
                max(dPolygonArea, dDesired) / min(dPolygonArea, dDesired)
            # this is the total size error for all polygons
            dSizeErrorTotal = dSizeErrorTotal + dSizeError
        # average error
        dMean = dSizeErrorTotal / featCount
        # need to read up more on why this is done
        dForceReductionFactor = 1 / (dMean + 1)
        return (aLocal, dForceReductionFactor)
