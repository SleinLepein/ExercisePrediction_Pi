import pickle

# Changes the imports of a given pickle file
class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "train_model.domain.timeseries_forest_classifier":
            renamed_module = "shared_code.timeseries_forest_classifier"
        return super(Unpickler, self).find_class(renamed_module, name)

def unpickler_load(file_obj):
    return Unpickler(file_obj).load()