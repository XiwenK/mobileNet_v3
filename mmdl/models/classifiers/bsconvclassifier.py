from mmcls.models.builder import CLASSIFIERS
from mmcls.models.classifiers import ImageClassifier

@CLASSIFIERS.register_module()
class BSConvClassifier(ImageClassifier):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None):
        super(BSConvClassifier, self).__init__(backbone, neck, head, pretrained, train_cfg, init_cfg)

    def train_step(self, data, optimizer=None, **kwargs):

        losses = super().forward(**data)
        loss, log_vars = self._parse_losses(losses)
        reg_loss = self.reg_loss()
        loss += reg_loss

        outputs = dict(
            loss=loss, reg_loss=reg_loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs
    
    def reg_loss(self, alpha=1/6):
        loss = 0.0
        for sub_module in self.modules():
            if hasattr(sub_module, "_reg_loss"):
                loss += sub_module._reg_loss()
        return alpha * loss