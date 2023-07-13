import os
import re
import json
from _ctypes import PyObj_FromPtr
import pandas as pd
import collections
import xmltodict

this_path = os.path.dirname(__file__)
main_path = os.path.abspath(os.path.join(this_path, os.pardir, os.pardir))
default_input_path = os.path.join(main_path, 'defaults')


# TODO: add exception classes, print functions here


def nested_update(d_old, d_new):
    # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    for key, val in d_new.items():
        if isinstance(val, collections.abc.Mapping):
            d_old[key] = nested_update(d_old.get(key, {}), val)
        else:
            d_old[key] = val
    return d_old


def load_csv(file_name, sub_folder=None, **kwargs):
    if file_name is None:
        return None

    if not os.path.isabs(file_name):
        if sub_folder is not None:
            file_name = os.path.join(default_input_path, sub_folder, file_name)
        else:
            file_name = os.path.join(default_input_path, file_name)
    return pd.read_csv(file_name, **kwargs)


def convert_hpxml_element(obj):
    # simplify HPXML elements, recursively
    if isinstance(obj, dict):
        if len(obj) == 0:
            return {}

        first = list(obj.values())[0]
        if len(obj) == 1 and isinstance(first, list) and len(first) > 0 and isinstance(first[0], dict):
            if 'SystemIdentifier' in first[0]:
                # convert list into dict using 'SystemIdentifier'
                new_obj = {d['SystemIdentifier']['@id']: d for d in first}
            else:
                # keep list, remove name of length-1 dictionary key
                new_obj = first
            return convert_hpxml_element(new_obj)
        elif len(obj) == 1 and isinstance(first, dict) and 'SystemIdentifier' in first:
            # rename dict key using 'SystemIdentifier'
            key = first['SystemIdentifier']['@id']
            return {key: convert_hpxml_element(first)}
        elif len(obj) == 1 and first is None:
            return list(obj.keys())[0]
        elif len(obj) == 1 and '@id' in obj:
            # Remove ID information (it is redundant)
            return {}
        elif '#text' in obj.keys():
            # remove all ID information, keep only the text
            return convert_hpxml_element(obj['#text'])
        else:
            return {key: convert_hpxml_element(val) for key, val in obj.items()}

    elif isinstance(obj, list):
        if len(obj) > 0 and isinstance(obj[0], dict) and 'SystemIdentifier' in obj[0]:
            # Convert list to dict with ids as keys
            return {item['SystemIdentifier']['@id']: convert_hpxml_element(item) for item in obj}
        else:
            return [convert_hpxml_element(item) for item in obj]

    elif isinstance(obj, str):
        if obj in ['true', 'false']:
            # convert string to boolean
            return eval(obj.capitalize())
        try:
            # convert string to float or list, if possible
            new_obj = eval(obj)
            if isinstance(new_obj, tuple):
                new_obj = list(new_obj)
            return new_obj
        except (NameError, SyntaxError):
            return obj

    else:
        raise Exception(f'Unknown HPXML object type ({type(obj)}: {obj}')


def import_hpxml(hpxml_file, **house_args):
    if not os.path.isabs(hpxml_file):
        hpxml_file = os.path.join(default_input_path, 'Input Files', hpxml_file)

    # Load HPXML file as a dictionary
    with open(hpxml_file) as f:
        hpxml_original = xmltodict.parse(f.read())

    # Check version - requires 3.0 or 4.0 for now
    version = hpxml_original['HPXML']['@schemaVersion']
    assert version in ['4.0']

    # Keep only building details
    hpxml = convert_hpxml_element(hpxml_original['HPXML']['Building']['BuildingDetails'])
    hpxml = dict(hpxml)

    return hpxml


class NoIndent(object):
    """ Value wrapper. """

    def __init__(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError('Only lists and tuples can be wrapped')
        self.value = value


class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'  # Unique string pattern of NoIndent object ids.
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding NoIndent wrapped values.
        ignore = {'cls', 'indent'}

        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(MyEncoder, self).default(obj))

    def iterencode(self, obj, **kwargs):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.

        # Replace any marked-up NoIndent wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super(MyEncoder, self).iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_repr = json.dumps(no_indent.value, **self._kwargs)
                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                encoded = encoded.replace(
                    '"{}"'.format(format_spec.format(id)), json_repr)

            yield encoded


# Not used
# def get_all_items(d):
#     for key, value in d.items():
#         yield key, value
#         if isinstance(value, dict):
#             yield from get_all_items(value)


def save_json(data, file_name):
    # saves json file but writes long lists to a single line
    # see: https://stackoverflow.com/questions/42710879/write-two-dimensional-list-to-json-file
    def parse_object(obj):
        if isinstance(obj, dict):
            return {key: parse_object(val) for key, val in obj.items()}
        elif isinstance(obj, (list, tuple)) and len(obj) > 4:
            return NoIndent(obj)
        else:
            return obj

    data_to_save = parse_object(data)

    with open(file_name, 'w') as f:
        json.dump(data_to_save, f, cls=MyEncoder, indent=4)
