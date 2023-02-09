#!/usr/bin/env python3
import overpy
import json
import time
import pandas as pd
import utilities as util


def get_overpass_data(source_dataframe: pd.DataFrame):
    n = 0
    api = overpy.Overpass()
    location_collection = {}

    while n < source_dataframe.shape[0]:
        wd = source_dataframe.iloc[n]
        util.show_progress(f"get_overpass_data: wudi Nº{int(wd['pid'])}", n, source_dataframe.shape[0])
        #print('n'+str(n).zfill(2))

        try:
            q = f"[out:json];node(around:25000, {wd['M_lat']}, {wd['M_lon']})[place~\"^(city|town|village)$\"];out;"
            result = api.query(q)

            for node in result.nodes:
                location_collection[node.id] = {
                    'lon': node.lon,
                    'lat': node.lat,
                    'eco': int(wd['eco']),
                    'geo': int(wd['geo']),
                    'wudi': int(wd['pid']),
                    'tags': node.tags
                }

            n += 1
            time.sleep(0.35)

        except (overpy.exception.OverpassTooManyRequests, overpy.exception.OverpassGatewayTimeout) as error:
            print("redo", error)
            time.sleep(1.5)
            pass

    return location_collection


def get_wiki_media_data(source_locations: dict):
    import requests
    query = """
        SELECT DISTINCT ?townLabel ?countryLabel ?area ?population ?elevation 
        (group_concat(distinct ?regionLabel;separator=", " ) as ?regionLabels)
        (group_concat(distinct ?waterLabel;separator=", " ) as ?waterLabels) WHERE {

            wd:%(q)s wdt:P17 ?country.
            wd:%(q)s wdt:P2044 ?elevation.
            wd:%(q)s wdt:P1082 ?population.
            wd:%(q)s wdt:P131 ?region.

          OPTIONAL{
            wd:%(q)s wdt:P206 ?water. 
            wd:%(q)s wdt:P2046 ?area.
          }

          SERVICE wikibase:label {
            bd:serviceParam wikibase:language "en". 
            wd:%(q)s rdfs:label ?townLabel.
            ?country rdfs:label ?countryLabel.
            ?region rdfs:label ?regionLabel.
            ?water rdfs:label ?waterLabel.
          } 

        }group by ?townLabel ?countryLabel ?area ?population ?elevation 
        """
    data_flat_list = []

    for i, loc in enumerate(source_locations):
        #print(i, loc)
        util.show_progress(f"get_wiki_media_data", i, len(source_locations))
        q_data = None

        if 'tags' in source_locations[loc]:
            tmp = source_locations[loc]
            tmp['node'] = loc

            if 'population' in tmp['tags']:
                try:
                    tmp['population'] = int(float(tmp['tags']['population']))
                except ValueError:
                    pass

            if 'name' in tmp['tags']:
                tmp['name'] = tmp['tags']['name']

            if 'name:en' in tmp['tags']:
                tmp['name'] = tmp['tags']['name:en']

            if 'int_name' in tmp['tags']:
                tmp['name'] = tmp['tags']['int_name']

            if 'wikidata' in source_locations[loc]['tags']:
                q_data = source_locations[loc]['tags']['wikidata']
                #print(i, loc, q_data)
                try:
                    wikidata_sparql_url = "https://query.wikidata.org/sparql"
                    query_string = query % {'q': q_data}
                    response = requests.get(wikidata_sparql_url, params={"query": query_string, "format": "json"})
                    jso = json.loads(response.text)
                    time.sleep(1.125)

                    for di, k in enumerate(jso['head']['vars']):
                        bindings = jso['results']['bindings']
                        if len(bindings):
                            try:
                                tmp[k] = bindings[0][k]['value']
                            except KeyError:
                                pass

                    if 'waterLabels' in tmp and len(tmp['waterLabels']) == 0:
                        del tmp['waterLabels']

                    tmp['source'] = q_data
                except json.decoder.JSONDecodeError:
                    pass

            if 'place' in tmp['tags']:
                tmp['type'] = tmp['tags']['place']

            if 'capital' in tmp['tags']:
                tmp['capital'] = tmp['tags']['capital']
                del tmp['tags']['capital']

            if 'townLabel' not in tmp:
                try:
                    tmp['townLabel'] = tmp['tags']['name']
                    del tmp['tags']['name']
                except KeyError:
                    print(tmp)

            tmp['place_id'] = i

            del tmp['tags']
            data_flat_list.append(tmp)

            # if q_data is not None:
            #     print(tmp)

    return data_flat_list
    # when complete, save this all up
    # with open('locations_super.json', 'w') as f:
    #     json.dump(data_flat_list, f, indent=1, cls=DecimalEncoder)
