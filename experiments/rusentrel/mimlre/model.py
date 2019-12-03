from core.networks.multi.training.batch import MultiInstanceBatch
from experiments.rusentrel.context.model import ContextLevelTensorflowModel
from experiments.rusentrel.mimlre.helper.initialization import MIMLREModelInitHelper


class MIMLRETensorflowModel(ContextLevelTensorflowModel):

    def create_batch_by_bags_group(self, bags_group):
        return MultiInstanceBatch(bags_group)

    def create_model_init_helper(self):
        return MIMLREModelInitHelper(io=self.IO, config=self.Config)
