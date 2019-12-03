import datetime

import numpy as np
import os

from core.common.utils import create_dir_if_not_exists
from core.evaluation.results.two_class import TwoClassEvalResult
from core.networks.callback import Callback
from core.networks.cancellation import OperationCancellation
from core.networks.context.debug import DebugKeys
from core.networks.tf_model import TensorflowModel
from core.networks.data_type import DataType
from core.networks.predict_log import NetworkInputDependentVariables


class CustomCallback(Callback):

    TrainingLogName = 'stat.csv'

    VocabularyFilepath = u'vocab.txt'
    HiddenParamsTemplate = u'hparams_{}_e{}'
    InputDependentParamsTemplate = u'idparams_{}_e{}'
    PredictVerbosePerFileStatistic = True

    def __init__(self, eval_result_filepath, log_dir):

        self.__eval_result_filepath = eval_result_filepath
        self.__model = None
        self.__test_on_epochs = None
        self.__log_dir = log_dir

        self.__results_history = None
        self.__costs_history = None
        self.__reset_experiment_dependent_parameters()

        self.__costs_window = 5

        self.__key_perform_train_evaluation = True
        self.__key_save_hidden_parameters = True
        self.__key_stop_training_by_cost = False
        self.__cancellation_acc_bound = 0.99

    @property
    def Model(self):
        return self.__model

    def reset_experiment_dependent_parameters(self):
        self.__reset_experiment_dependent_parameters()

    # region event handlers

    def on_fit_finished(self):
        self.__eval_result_file.write("===========\n")
        self.__eval_result_file.write("Avg. test F-1: {}\n".format(self.__print_avg_stat(result_metric_name=TwoClassEvalResult.C_F1)))
        self.__eval_result_file.write("-----------\n")
        self.__eval_result_file.write("Avg. test P_R: {}\n".format(self.__print_avg_stat(result_metric_name=TwoClassEvalResult.C_POS_RECALL)))
        self.__eval_result_file.write("Avg. test N_R: {}\n".format(self.__print_avg_stat(result_metric_name=TwoClassEvalResult.C_NEG_RECALL)))
        p_r = self.__get_avg_stat(TwoClassEvalResult.C_POS_RECALL)
        n_r = self.__get_avg_stat(TwoClassEvalResult.C_NEG_RECALL)
        self.__eval_result_file.write("R(P,N):        {}\n".format(round(1.0 * (p_r + n_r) / 2, 2)))
        self.__eval_result_file.write("-----------\n")
        self.__eval_result_file.write("Avg. test P_P: {}\n".format(self.__print_avg_stat(result_metric_name=TwoClassEvalResult.C_POS_PREC)))
        self.__eval_result_file.write("Avg. test N_P: {}\n".format(self.__print_avg_stat(result_metric_name=TwoClassEvalResult.C_NEG_PREC)))
        p_p = self.__get_avg_stat(TwoClassEvalResult.C_POS_PREC)
        n_p = self.__get_avg_stat(TwoClassEvalResult.C_NEG_PREC)
        self.__eval_result_file.write("P(P,N):        {}\n".format(round(1.0 * (p_p + n_p) / 2, 2)))
        self.__eval_result_file.write("===========\n")

    def on_initialized(self, model):
        assert(isinstance(model, TensorflowModel))
        self.__model = model

    def on_epoch_finished(self, avg_cost, avg_acc, epoch_index, operation_cancel):
        assert(isinstance(avg_cost, float))
        assert(isinstance(avg_acc, float))
        assert(isinstance(operation_cancel, OperationCancellation))

        if DebugKeys.FitEpochCompleted:
            print "{}: Epoch: {}: avg. cost: {:.3f}, avg. acc.: {:.3f}".format(
                str(datetime.datetime.now()),
                epoch_index,
                avg_cost,
                avg_acc)

        if avg_acc >= self.__cancellation_acc_bound:
            print "Stop training process: avg_acc > {}".format(self.__cancellation_acc_bound)
            operation_cancel.Cancel()

        if (epoch_index not in self.__test_on_epochs) and (not operation_cancel.IsCancelled):
            return

        if self.__key_perform_train_evaluation:
            self.__process_for_data_type(data_type=DataType.Train,
                                         epoch_index=epoch_index)

        result_test = self.__process_for_data_type(data_type=DataType.Test,
                                                   epoch_index=epoch_index)

        if avg_acc > 0.90:
            self.__results_history.append(result_test)

        if self.__key_stop_training_by_cost:
            if not self.__check_costs_still_improving(avg_cost):
                print "Cancelling: cost becomes greater than min value {} epochs ago.".format(
                    self.__costs_window)
                operation_cancel.Cancel()

        self.__save_model_hidden_values(epoch_index)
        self.__save_model_vocabulary()
        self.__costs_history.append(avg_cost)

    # endregion

    # region 'set' methods

    def set_key_stop_training_by_cost(self, value):
        assert(isinstance(value, bool))
        self.__key_stop_training_by_cost = value

    def set_key_perform_train_evaluation(self, value):
        assert(isinstance(value, bool))
        self.__key_perform_train_evaluation = value

    def set_key_save_hidden_parameters(self, value):
        assert(isinstance(value, bool))
        self.__key_save_hidden_parameters = value

    def set_test_on_epochs(self, value):
        assert(isinstance(value, list))
        self.__test_on_epochs = value

    def set_cancellation_acc_bound(self, value):
        assert(isinstance(value, float))
        self.__cancellation_acc_bound = value

    # endregion

    # region private methods

    def __reset_experiment_dependent_parameters(self, ):
        self.__results_history = [None]
        self.__costs_history = []

    def __check_costs_still_improving(self, avg_cost):

        history_len = len(self.__costs_history)

        if history_len <= self.__costs_window:
            return True

        return avg_cost < min(self.__costs_history[:history_len - self.__costs_window])

    def __process_for_data_type(self, data_type, epoch_index):
        assert(isinstance(data_type, unicode))
        assert(isinstance(epoch_index, int))

        result, idhp = self.__model.predict(dest_data_type=data_type)

        assert(isinstance(idhp, NetworkInputDependentVariables))
        assert(isinstance(result, TwoClassEvalResult))

        if self.PredictVerbosePerFileStatistic:
            self.__print_verbose_eval_results(result, data_type)

        self.__print_overall_results(result, data_type)
        self.__save_minibatch_all_input_dependent_hidden_values(
            predict_log=idhp,
            data_type=data_type,
            epoch_index=epoch_index)

        return result

    def __save_model_vocabulary(self):
        assert(isinstance(self.__model, TensorflowModel))

        if not self.__key_save_hidden_parameters:
            return

        vocab_path = os.path.join(self.__log_dir, self.VocabularyFilepath)
        np.savez(vocab_path, list(self.__model.iter_inner_input_vocabulary()))

    def __save_model_hidden_values(self, epoch_index):

        if not self.__key_save_hidden_parameters:
            return

        names, values = self.__model.get_hidden_parameters()

        assert(isinstance(names, list))
        assert(isinstance(values, list))
        assert(len(names) == len(values))

        for i, name in enumerate(names):
            variable_path = os.path.join(self.__log_dir, self.HiddenParamsTemplate.format(name, epoch_index))
            print "Save hidden values: {}".format(variable_path)
            create_dir_if_not_exists(variable_path)
            np.save(variable_path, values[i])

    def __save_minibatch_all_input_dependent_hidden_values(self, data_type, epoch_index, predict_log):
        assert(isinstance(predict_log, NetworkInputDependentVariables))

        if not self.__key_save_hidden_parameters:
            return

        for var_name in predict_log.iter_var_names():
            self.__save_minibatch_variable_values(data_type=data_type,
                                                  epoch_index=epoch_index,
                                                  predict_log=predict_log,
                                                  var_name=var_name)

    def __save_minibatch_variable_values(self, data_type, epoch_index, predict_log, var_name):
        assert(isinstance(predict_log, NetworkInputDependentVariables))
        assert(isinstance(var_name, unicode))

        if not self.__key_save_hidden_parameters:
            return

        vars_path = os.path.join(self.__log_dir,
                                 self.InputDependentParamsTemplate.format(
                                     '{}-{}'.format(var_name, data_type),
                                     epoch_index))
        create_dir_if_not_exists(vars_path)

        print "Save input dependent hidden values in a list using np.savez: {}".format(vars_path)

        id_and_value_pairs = list(predict_log.iter_by_parameter_values(var_name))
        id_and_value_pairs = sorted(id_and_value_pairs, key=lambda pair: pair[0])
        np.savez(vars_path, [pair[1] for pair in id_and_value_pairs])

    @staticmethod
    def __print_verbose_eval_results(eval_result, data_type):
        assert(isinstance(eval_result, TwoClassEvalResult))

        print "Verbose statistic for {}:".format(data_type)
        for doc_id, result in eval_result.iter_document_results():
            print doc_id, result

    @staticmethod
    def __print_overall_results(eval_result, data_type):
        assert(isinstance(eval_result, TwoClassEvalResult))

        print "Overall statistic for {}:".format(data_type)
        for metric_name, value in eval_result.iter_results():
            print "\t{}: {}".format(metric_name, round(value, 2))

    def __get_avg_stat(self, result_metric_name):
        assert(isinstance(result_metric_name, unicode))

        avg_f1 = 0.0
        total_count = 0
        for eval_result in self.__results_history:

            if eval_result is None:
                continue

            assert(isinstance(eval_result, TwoClassEvalResult))

            total_count += 1
            avg_f1 += eval_result.get_result_by_metric(result_metric_name)

        return avg_f1 / total_count if total_count > 0 else avg_f1

    def __print_avg_stat(self, result_metric_name):
        assert(isinstance(result_metric_name, unicode))
        avg_value = self.__get_avg_stat(result_metric_name)
        return round(avg_value, 2)

    # endregion

    # region overriden methods

    def __enter__(self):
        create_dir_if_not_exists(self.__eval_result_filepath)
        self.__eval_result_file = open(self.__eval_result_filepath, "w", buffering=0)
        print "Eval filepath: {}".format(self.__eval_result_filepath)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__eval_result_file.close()

    # endregion
