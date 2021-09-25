class Options(object):

    def __init__(self, dictionary):
        for key in dictionary:
            final_value = load_single_obj(dictionary[key])
            setattr(self, key, final_value)

    @property
    def name(self):
        try:
            return self._name
        except AttributeError as e:
            print(e)


def load_single_obj(dictionary):
    if isinstance(dictionary, dict):
        value = Options(dictionary)
    else:
        value = dictionary
    return value
