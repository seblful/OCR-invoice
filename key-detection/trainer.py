import ultralytics
import torch


class Trainer():
    def __init__(self,
                 dataset_path,
                 num_epochs,
                 image_size,
                 batch_size,
                 seed,
                 model_type='n'):

        self.dataset_path = dataset_path
        self.num_epochs = num_epochs
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed
        self.model_type = model_type

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_yaml = f"yolov8{model_type}-obb.yaml"
        self.model_type = f"yolov8{model_type}-obb.pt"
        self._data = None

        self._model = None
        self.is_trained = False

    @property
    def model(self):
        if self._model == None:
            # Build a new model from scratch
            model = ultralytics.YOLO(self.model_yaml)
            # Load a pretrained model
            model = ultralytics.YOLO(self.model_type)

            # # Load from my pretrained model
            # model = ultralytics.YOLO('best3.pt')

            self._model = model

        return self._model

    @property
    def data(self):
        if self._data is None:
            self._data = self.dataset_path + '/data.yaml'

        return self._data

    def train(self):
        '''
        Training model
        '''

        result = self.model.train(data=self.data, epochs=self.num_epochs, imgsz=self.image_size,
                                  batch=self.batch_size, seed=self.seed)  # , close_mosaic=0, resume="runs/detect/train/weights/yolov8n.pt", workers=1 maybe
        self.is_trained = True

        return result

    def validate(self):
        '''
        Validating model on test dataset
        '''

        if self.is_trained == True:
            metrics = self.model.val(
                data=self.data, imgsz=self.image_size, split='test')

            return metrics

        else:
            return f"Model is not trained yet."
