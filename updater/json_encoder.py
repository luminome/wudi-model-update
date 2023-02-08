#!/usr/bin/env python3

from shapely.geometry import Polygon, MultiPolygon, LinearRing, LineString, MultiLineString
import numpy as np
import json


class JsonSafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.nan):
            return '"'+str(obj)+'"'
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return round(float(obj), 5)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, MultiPolygon):
            return str(obj.__class__)
        if isinstance(obj, Polygon):
            return str(obj.__class__)
        if isinstance(obj, LineString):
            return str(obj.__class__)
        if isinstance(obj, MultiLineString):
            return str(obj.__class__)
        if isinstance(obj, LinearRing):
            return str(obj.__class__)

        return super(JsonSafeEncoder, self).default(obj)
