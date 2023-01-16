# =============================================================================
# Libs
# =============================================================================
from dudu_tests.bert.custom_strategy_bert import train_and_test
import unittest
import random
import torch
import os
import torch.distributed as dist
from fairscale.nn.model_parallel import initialize_model_parallel, destroy_model_parallel
from dudu_tests import strategy_handler
from fairscale.utils.testing import set_random_seed
from dudu_tests.strategy_handler import LayerStrategy


# Turns a dictionary into a class
class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


class StrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        set_random_seed(12345)
        cls.train_args = Dict2Class({'seq_length': 128, 'embed_size': 512, 'hidden_layer_size': 512, 'atten_heads': 8,
                                     'encoder_layers': 2, 'batch_size': 32, 'print_params': False, 'epochs': 1,
                                     'budget': 50, 'import_strategy': "strategies/strategy.json",
                                     'export_strategy': "strategies/exported_strat.json", 'name': 'no_name.pt'})
        cls.vanilla_strategy = strategy_handler.import_strategy('strategies/bert_vanilla.json')
        cls.local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        backends = {"model_parallel_backend": "nccl", "pipeline_backend": "nccl", "ddp_backend": "nccl"}
        initialize_model_parallel(dist.get_world_size(), 1, **backends)

    @classmethod
    def tearDownClass(cls) -> None:
        destroy_model_parallel()
        dist.destroy_process_group()

    def setUp(self) -> None:
        global_rank = dist.get_rank()
        print(f"[{os.getpid()}] rank = {global_rank}, " + f"world_size = {dist.get_world_size()}")
        torch.cuda.set_device(self.local_rank)
        self.train_args.strategy = self.vanilla_strategy.copy()

    # def test_best_vs_vanilla_strategies(self):
    #     if self.local_rank == 0 and self.train_args.print_params:
    #         print(self.train_args.strategy)
    #     # warmup run to make sure comparison between runs doesnt include set up times and memory pinning
    #     self.train_args.n_iteration = 30
    #     self.train_args.epochs = 1
    #     train_and_test(self.local_rank, self.train_args)
    #     # run both vanilla and custom strategies to make sure the custom strategy is an improvement
    #     self.train_args.n_iteration = 200
    #     self.train_args.epochs = 1
    #     vanilla_ips = train_and_test(self.local_rank, self.train_args)
    #     self.train_args.strategy = strategy_handler.import_strategy('strategies/strategy.json')
    #     best_ips = train_and_test(self.local_rank, self.train_args)
    #     print(f'vanilla ips: {vanilla_ips}, best ips: {best_ips}')
    #     self.assertGreater(best_ips, vanilla_ips)

    def test_a_illegal_strategies_test(self):
        n_iteration = 10
        avg_ips, latency = train_and_test(self.local_rank, self.train_args)

    def test_a_vanilla_strategy(self):
        if self.local_rank == 0 and self.train_args.print_params:
            print(self.train_args.strategy)
        self.train_args.n_iteration = 30
        self.assertIs(type(train_and_test(self.local_rank, self.train_args)), float)

    def test_b_column_without_gather_then_row_with_input_is_parallel(self):
        self.train_args.strategy["encoders.0.ff.linear1"] = LayerStrategy(column_linear=True, gather_output=False)
        self.train_args.strategy["encoders.0.ff.linear2"] = LayerStrategy(row_linear=True, input_is_parallel=True)
        if self.local_rank == 0 and self.train_args.print_params:
            print(self.train_args.strategy)
        self.train_args.n_iteration = 30
        self.assertIs(type(train_and_test(self.local_rank, self.train_args)), float)

    def test_c_row_parallel_with_input_is_parallel(self):
        self.train_args.strategy["encoders.0.ff.linear1"] = LayerStrategy(row_linear=True, input_is_parallel=True)
        if self.local_rank == 0 and self.train_args.print_params:
            print(self.train_args.strategy)
        self.train_args.n_iteration = 30
        with self.assertRaises(RuntimeError):
            train_and_test(self.local_rank, self.train_args)

    def test_d_consecutive_column_parallel_without_gather_in_between(self):
        self.train_args.strategy["encoders.0.ff.linear1"] = LayerStrategy(column_linear=True, gather_output=False)
        self.train_args.strategy["encoders.0.ff.linear2"] = LayerStrategy(column_linear=True, gather_output=False)
        if self.local_rank == 0 and self.train_args.print_params:
            print(self.train_args.strategy)
        self.train_args.n_iteration = 30
        with self.assertRaises(RuntimeError):
            train_and_test(self.local_rank, self.train_args)


if __name__ == "__main__":
    unittest.main()



