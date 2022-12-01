from mmcv.runner.hooks import HOOKS, Hook
import pdb

@HOOKS.register_module()
class TrainsetEvalHook(Hook):

    def __init__(self,):
        pass

    def before_train_epoch(self, runner):
        """

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        # pdb.set_trace()
        runner.model.module.head.reset_pred()


    def after_train_epoch(self, runner):
        """

        Args:
            runner (obj:`EpochBasedRunner`): Epoch based Runner.
        """
        correct, total = runner.model.module.head.get_pred_count()
        accuracy = round(correct / total * 100, 4)
        runner.log_buffer.output['train_accuracy'] = accuracy
        # pdb.set_trace()
