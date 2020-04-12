import yaml
import pandas as pd


def load_yaml(path):
    with open(path, 'r') as f:
        config = yaml.load(f)
    return config


def get_dict_for_geocoding(path):
    try:
        address_dict = {}
        df = pd.read_csv(path)
        for _, addr in df.iterrows():
            address_dict[addr['address']] = {}
            address_dict[addr['address']]['lat'] = addr['lat']
            address_dict[addr['address']]['lon'] = addr['lon']
            address_dict[addr['address']]['country'] = addr['country']
        return address_dict
    except FileNotFoundError:
        return {}


def dump_dict_for_geocoding(address_dict, path):
    addresses, lons, lats, countries = [], [], [], []
    for k in address_dict:
        addresses.append(k)
        lons.append(address_dict[k]['lon'])
        lats.append(address_dict[k]['lat'])
        countries.append(address_dict[k]['country'])
    df = pd.DataFrame.from_dict({'address': addresses, 'country': countries,
                                 'lat': lats, 'lon': lons})
    df.to_csv(path, index=False)


