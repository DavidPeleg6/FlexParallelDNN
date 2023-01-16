# =============================================================================
# Libs
# =============================================================================
from dudu_tests.bert.bert_with_strategy import train_and_test
import unittest
import random
import torch
import os
import torch.distributed as dist
from fairscale.nn.model_parallel import initialize_model_parallel, destroy_model_parallel
from dudu_tests import strategy_handler
from fairscale.utils.testing import set_random_seed


# Turns a dictionary into a class
class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


class StrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_args = Dict2Class({'seq_length': 20, 'embed_size': 240, 'hidden_layer_size': 4*240, 'atten_heads': 12,
                                     'encoder_layers': 12, 'batch_size': 256, 'print_params': False, 'epochs': 1,
                                     'budget': 50, 'import_strategy': "strategies/strategy.json",
                                     'export_strategy': "strategies/exported_strat.json", 'name': 'no_name.pt',
                                     'n_iteration': 30})
        cls.vanilla_strategy = strategy_handler.import_strategy('strategies/bert_vanilla3.json')
        cls.local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        backends = {"model_parallel_backend": "nccl", "pipeline_backend": "nccl", "ddp_backend": "nccl"}
        initialize_model_parallel(dist.get_world_size(), 1, **backends)
        set_random_seed(12345)

    @classmethod
    def tearDownClass(cls) -> None:
        destroy_model_parallel()
        dist.destroy_process_group()

    def setUp(self) -> None:
        global_rank = dist.get_rank()
        print(f"[{os.getpid()}] rank = {global_rank}, " + f"world_size = {dist.get_world_size()}")
        torch.cuda.set_device(self.local_rank)
        self.train_args.strategy = self.vanilla_strategy.copy()

    def test_layers_are_parallel(self):
        if self.local_rank == 0 and self.train_args.print_params:
            print(self.train_args.strategy)
        # warmup run to make sure comparison between runs doesnt include set up times and memory pinning
        train_and_test(self.local_rank, self.train_args)
        # run both vanilla and custom strategies to make sure the custom strategy is an improvement
        vanilla_ips = train_and_test(self.local_rank, self.train_args)
        self.train_args.strategy = strategy_handler.import_strategy('strategies/strategy.json')
        best_ips = train_and_test(self.local_rank, self.train_args)
        print(f'vanilla ips: {vanilla_ips}, best ips: {best_ips}')
        self.assertGreater(best_ips, vanilla_ips)

    # def test_a_vanilla_strategy(self):
    #     if self.local_rank == 0 and self.train_args.print_params:
    #         print(self.train_args.strategy)
    #     self.train_args.n_iteration = 30
    #     self.assertIs(type(train_and_test(self.local_rank, self.train_args)), float)
    #
    # def test_b_column_without_gather_then_row_with_input_is_parallel(self):
    #     self.train_args.strategy["encoders.0.ff.linear1"] = LayerStrategy(column_linear=True, gather_output=False)
    #     self.train_args.strategy["encoders.0.ff.linear2"] = LayerStrategy(row_linear=True, input_is_parallel=True)
    #     if self.local_rank == 0 and self.train_args.print_params:
    #         print(self.train_args.strategy)
    #     self.train_args.n_iteration = 30
    #     self.assertIs(type(train_and_test(self.local_rank, self.train_args)), float)
    #
    # def test_c_row_parallel_with_input_is_parallel(self):
    #     self.train_args.strategy["encoders.0.ff.linear1"] = LayerStrategy(row_linear=True, input_is_parallel=True)
    #     if self.local_rank == 0 and self.train_args.print_params:
    #         print(self.train_args.strategy)
    #     self.train_args.n_iteration = 30
    #     with self.assertRaises(RuntimeError):
    #         train_and_test(self.local_rank, self.train_args)
    #
    # def test_d_consecutive_column_parallel_without_gather_in_between(self):
    #     self.train_args.strategy["encoders.0.ff.linear1"] = LayerStrategy(column_linear=True, gather_output=False)
    #     self.train_args.strategy["encoders.0.ff.linear2"] = LayerStrategy(column_linear=True, gather_output=False)
    #     if self.local_rank == 0 and self.train_args.print_params:
    #         print(self.train_args.strategy)
    #     self.train_args.n_iteration = 30
    #     with self.assertRaises(RuntimeError):
    #         train_and_test(self.local_rank, self.train_args)
    #
    # # here just to debug the freezing error I get when running with a lot of iterations + epochs
    # def test_a_a_(self):
    #     best_strategy = self.train_args.strategy.copy()
    #     seen_strategies = [best_strategy]
    #     # the amount of iterations for each simulation iteration
    #     self.train_args.n_iteration = 30
    #     base_ips = train_and_test(self.local_rank, self.train_args)
    #     # search for best strategy
    #     best_ips = base_ips
    #     self.train_args.n_iteration = 30
    #     for i in range(self.train_args.budget):
    #         iter_counter = 0
    #         while self.train_args.strategy in seen_strategies and iter_counter < 100:
    #             # randomize the strategy
    #             self.train_args.strategy = strategy_handler.randomize_strategy(best_strategy.copy(),
    #                                                                       random_layer_amount=True)
    #             iter_counter += 1
    #         seen_strategies.append(self.train_args.strategy.copy())
    #         # TODO delete diagnostic print
    #         if self.local_rank == 0:
    #             print(f'iteration = {i}\nstrategy = {self.train_args.strategy}')
    #         # test the strategy by checking ips in training
    #         try:
    #             avg_ips = train_and_test(self.local_rank, self.train_args)
    #             torch.cuda.empty_cache()
    #         except RuntimeError:
    #             raise Exception(f'rank = {self.local_rank}. runtime error occured for the current strategy\n{self.train_args.strategy}')
    #         # if its not good change the layer strategy back
    #         if avg_ips > best_ips:
    #             best_strategy = self.train_args.strategy.copy()
    #             best_ips = avg_ips
    #         if dist.get_rank() == 0 and self.train_args.print_params:
    #             print(f'iteration: {i}, cur_ips: {avg_ips}')
    #     if dist.get_rank() == 0:
    #         print(f'Rank: {dist.get_rank()}, '
    #               f'Best ips: {best_ips}, '
    #               f'Baseline ips: {base_ips}, \n'
    #               f'Best strategy: {strategy_handler.dict_strategy(best_strategy)}')

if __name__ == "__main__":
    unittest.main()



