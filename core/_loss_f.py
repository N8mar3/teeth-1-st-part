import torch


class LossFunction:
    def __init__(self,
                 prediction,
                 target,
                 device_for_training,
                 semantic_binary: bool = True,
                 ):
        self.prediction = prediction
        self.device = device_for_training
        self.target = target
        self.semantic_binary = semantic_binary

    def forward(self):
        if self.semantic_binary:
            return self.dice_loss(self.prediction, self.target)
        return self.categorical_dice_loss()

    @staticmethod
    def dice_loss(predictions, targets, alpha=1e-5):
        intersection = 2. * (predictions * targets).sum()
        denomination = (torch.square(predictions) + torch.square(targets)).sum()
        dice_loss = 1 - torch.mean((intersection + alpha) / (denomination + alpha))

        return dice_loss

    def categorical_dice_loss(self):
        pr, tr = self.prepare_for_multiclass_loss_f()
        target_categories, losses = torch.unique(tr).tolist(), 0
        for num_category in target_categories:
            categorical_target = torch.where(tr == num_category, 1, 0)
            categorical_prediction = pr[num_category][:][:][:]
            losses += self.dice_loss(categorical_prediction, categorical_target).to(self.device)

        return losses

    def prepare_for_multiclass_loss_f(self):
        prediction_prepared = torch.squeeze(self.prediction, 0)
        target_prepared = torch.squeeze(self.target, 0)
        target_prepared = torch.squeeze(target_prepared, 0)

        return prediction_prepared, target_prepared
