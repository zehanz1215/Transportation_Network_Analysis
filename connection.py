from pyrosm import OSM, get_data
import osmnx as ox
import pandas as pd
import networkx as nx

def road_class_to_kmph(road_class):
    """
    Returns a speed limit value based on road class, 
    using typical Finnish speed limit values within urban regions.
    """
    if road_class == "motorway":
        return 100
    elif road_class == "motorway_link":
        return 80
    elif road_class in ["trunk", "trunk_link"]:
        return 60
    elif road_class == "service":
        return 30
    elif road_class == "living_street":
        return 20
    else:
        return 50
    
def assign_speed_limits(edges):
    # Separate rows with / without speed limit information 
    mask = edges["maxspeed"].isnull()
    edges_without_maxspeed = edges.loc[mask].copy()
    edges_with_maxspeed = edges.loc[~mask].copy()

    # Apply the function and update the maxspeed
    edges_without_maxspeed["maxspeed"] = edges_without_maxspeed["highway"].apply(road_class_to_kmph)
    edges = edges_with_maxspeed.append(edges_without_maxspeed)
    edges["maxspeed"] = edges["maxspeed"].astype(int)
    edges["travel_time_seconds"] = edges["length"] / (edges["maxspeed"]/3.6)
    return edges
    
# Fetch data for Helsinki
osm = OSM(get_data("philadelphia"))
nodes, edges = osm.get_network(network_type="driving", nodes=True)

# Assign speed limits for missing ones based on road classs information
edges = assign_speed_limits(edges)

# Remove unnecessary columns to reduce memory footprint 
edges = edges[["highway", "oneway", "travel_time_seconds", "length", "u", "v", "geometry"]]