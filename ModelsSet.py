from CustomModel import CustomCnn
import os


class CnnSet:
    def __init__(self, cnn_set_name):
        if cnn_set_name is None:
            raise Exception("cnn_set_name must not be None")
        self.cnn_set = {}
        self.cnn_set_name = cnn_set_name
        self.path = ""

    def _add_model(self, model=None):
        if model is None:
            raise Exception("model must not be None")
        if not isinstance(model, CustomCnn):
            raise Exception("model must be CustomCnn")

        self.cnn_set[model.model_name] = model

    def _save_cnn_set(self, directory=None):
        if directory is None:
            raise Exception("directory must not be None")
        if self.cnn_set_name is None:
            raise Exception("cnn_sets_name must not be None")

        self.path = directory + self.cnn_set_name
        os.makedirs(self.path)
        for model_name, model in self.cnn_set.items():
            model._save_model(self.path + "/")

    def _load_cnn_set(self, cnn_set_path=None):
        if cnn_set_path is None:
            raise Exception("cnn_set_path must not be None")

        self.cnn_set_name = cnn_set_path.split("/")[-1]
        self.path = cnn_set_path

        file_list = os.listdir(self.path)
        for i in range(len(file_list)):
            file_list[i] = file_list[i].replace(".h5", "").replace(".json", "")

        file_list = list(set(file_list))

        for model_name in file_list:
            custom_cnn = CustomCnn(model_name="!@#&(*$!^@#")
            custom_cnn._load_model(self.path + "/", model_name)
            self.cnn_set[model_name] = custom_cnn

    def _get(self, model_name=None):
        if model_name is None:
            raise Exception("model_name must not be None")

        return self.cnn_set[model_name]

    def _delete_model(self, model_name=None):
        if model_name is None:
            raise Exception("model_name must not be None")

        try:
            del self.cnn_set[model_name]
        except:
            print(model_name, "can't find in cnn_set")

        try:
            os.remove(self.path + "/" + model_name + ".h5")
            os.remove(self.path + "/" + model_name + ".json")
        except:
            pass

    def _delete_cnn_set(self):
        self.cnn_set = {}
        file_list = os.listdir(self.path + "/")
        for file in file_list:
            os.remove(self.path + "/" + file)
        os.rmdir(self.path)

    def _info(self):
        return {"cnn_set": list(self.cnn_set.keys()), "cnn_set_name": self.cnn_set_name, "path": self.path}

## 1. 모델 추가 메소드
## 2. 저장 메소드
## 3. 로드 메소드
## 4. 삭제 메소드
