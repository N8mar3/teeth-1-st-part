import torch
from vedo import Volume, show, write


class WorkFunction:
    def __init__(self,
                 data_loader,
                 device_for_training,
                 model_name,
                 model,
                 save_3d: bool = False,
                 show_prediction: bool = False,
                 stage: bool = False,
                 smooth: int = 0.5,
                 ):
        self.data_loader = data_loader
        self.device = device_for_training
        self.model_name = model_name
        self.save = save_3d
        self.show = show_prediction
        self.model = model
        self.stage = stage
        self.smooth = smooth

    def predictor(self, switch: bool = True):
        self.model.load_state_dict(torch.load(self.model_name))
        self.model.eval()
        with torch.no_grad():
            for x in self.data_loader:
                x = x.to(self.device, dtype=torch.float)
                prediction = self.model(x)
                if switch: m_tensor, switch = prediction, False
                else: m_tensor = torch.cat((m_tensor, prediction), 0)
        prediction = self.data_decoder(m_tensor)

        return prediction

    def data_decoder(self, prediction):
        data = torch.squeeze(torch.argmax(prediction, dim=1), 1) if self.stage else (prediction > self.smooth)
        data = torch.reshape(data, (168, 120, 120))
        torch.save(data, "data_for_3rd_training.pt")
        data = torch.movedim(data, 0, 2) if self.stage else data
        np_data = data.cpu().numpy()
        self.show_save(np_data)

        return data

    def show_save(self, some_data):
        data_multiclass = Volume(some_data, c='Set2', alpha=(0.1, 1), alphaUnit=0.87, mode=1)
        data_multiclass.addScalarBar3D()
        show([(data_multiclass, "Multiclass teeth segmentation prediction")],
             bg='black', N=1, axes=1).close() if self.show else None
        write(data_multiclass.isosurface(), 'multiclass_.stl') if self.save else None
