import pickle


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        """
        changes the imports of a given pickle file

        Parameters
        ----------
        module : String
            imports that needs to be changed
        name : String
        Returns
        -------
        fixed pickle file
        """
        renamed_module = module
        if module == "train_model.domain.timeseries_forest_classifier":
            renamed_module = "src.model.timeseries_forest_classifier"
        return super(Unpickler, self).find_class(renamed_module, name)


def unpickler_load(file_obj):
    return Unpickler(file_obj).load()
