import pandas as pd
import geopandas as gpd

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from matplotlib.colors import Normalize, LinearSegmentedColormap


def setup(name="smsearch"):
    """
    Set up the geolocation
    :param name: The name to use for geopy
    :return:
    """
    geolocator = Nominatim(user_agent=name)
    gc = lambda query: geolocator.geocode(query, addressdetails=True)
    geocode = RateLimiter(gc, min_delay_seconds=1)
    return geocode


def loc_to_latlon(location, geocode):
    """
    Get a location name and return lat/lon and country name.
    :param location: Name of location
    :param geocode: The geocode finder from geopy
    :return: lat, lon and country codes
    """
    location = geocode(location)
    if location is None:
        return None, None, None
    lat, lon = location.latitude, location.longitude
    try:
        country = location.raw["address"]["country"]
    except KeyError:
        country = "&*COUNTRY_KEY_ERROR*&"
    #todo: some countries are not returned?
    return lat, lon, country


def plot_loc(ax, locations, geocode):
    """
    Plot the locations into a map
    :param ax: the object to draw on
    :param locations: list of locations
    :param geocode: as returned from setup
    :return:
    """
    ll = {"lat": [], "lon": []}
    countries = []
    for loc in locations:
        lt, ln, country = loc_to_latlon(loc, geocode)
        ll["lat"].append(lt)
        ll["lon"].append(ln)
        countries.append(country)

    ll_pd = pd.DataFrame(ll)
    gdf = gpd.GeoDataFrame(ll_pd, geometry=gpd.points_from_xy(ll_pd.lon, ll_pd.lat))
    gdf.plot(ax=ax, color='red')
    return ax, countries


def plot_loc_with_sentiment(ax, locations, sentiments, geocode):
    """
        Plot the locations into a map
        :param ax: the object to draw on
        :param locations: list of locations
        :param geocode: as returned from setup
        :return:
        """
    cmap = LinearSegmentedColormap.from_list('rg', ["r", "w", "g"], N=256)
    norm = Normalize(vmin=0.0, vmax=1.0)
    ll = {"lat": [], "lon": []}
    countries = []
    for loc in locations:
        lt, ln, country = loc_to_latlon(loc, geocode)
        ll["lat"].append(lt)
        ll["lon"].append(ln)
        countries.append(country)

    ll_pd = pd.DataFrame(ll)
    gdf = gpd.GeoDataFrame(ll_pd, geometry=gpd.points_from_xy(ll_pd.lon, ll_pd.lat))
    gdf.plot(ax=ax, color=cmap(norm(sentiments)))

    return ax, countries
