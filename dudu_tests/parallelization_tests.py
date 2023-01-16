# =============================================================================
# Libs
# =============================================================================
import unittest
import torch
import os
import torch.distributed as dist
from fairscale.nn.model_parallel import initialize_model_parallel, destroy_model_parallel
from dudu_tests.layerwise_data_parallel import DataParallelLayer
from dudu_tests.layerwise_model_parallel import RowParallelLinear, ColumnParallelLinear
from dudu_tests import strategy_handler
from strategy_handler import LayerStrategy
from fairscale.utils.testing import set_random_seed
from torch import nn
import dudu_tests.Optimizer as Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from fairscale.optim.oss import OSS
from torch.optim import SGD
import copy


# Turns a dictionary into a class
class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


class MnistMLP(nn.Module):
    def __init__(self, layers=8):
        super(MnistMLP, self).__init__()
        lin_layers = [nn.Linear(28 * 28, 1024)]
        for i in range(1, layers-1):
            lin_layers += [nn.Linear(1024, 1024)]
        lin_layers += [nn.Linear(1024, 10)]
        self.seq = nn.ModuleList(lin_layers)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        for layer in self.seq:
            x = layer(x)
        return x


class StrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.print_params = False
        dist.init_process_group(backend='nccl')
        backends = {"model_parallel_backend": "nccl", "pipeline_backend": "nccl", "ddp_backend": "nccl"}
        initialize_model_parallel(dist.get_world_size(), 1, **backends)
        set_random_seed(12346)
        cls.local_rank = dist.get_rank()
        cls.vanilla_strategy = strategy_handler.create_vanilla_strategy(model=MnistMLP())
        # cls.vanilla_strategy = strategy_handler.strategy_from_net(Optimizer.wrap_model(model=MnistMLP(), ))
        torch.cuda.set_device(cls.local_rank)

    @classmethod
    def tearDownClass(cls) -> None:
        destroy_model_parallel()
        dist.destroy_process_group()

    def setUp(self) -> None:
        self.strategy = self.vanilla_strategy.copy()
        self.vanilla_network = MnistMLP().to(self.local_rank)
        # change the strategy of some layers including data and model parallel
        self.strategy['seq.3'] = LayerStrategy(gather_input=True, split_output=False)
        self.strategy['seq.4'] = LayerStrategy(column_linear=True, gather_output=True)
        self.strategy['seq.5'] = LayerStrategy(row_linear=True, input_is_parallel=False)
        self.strategy['seq.6'] = LayerStrategy(gather_input=False, split_output=True)
        # todo delete this once youre sure it happens automatically
        # self.strategy['seq.7'] = LayerStrategy(gather_input=True, split_output=False)
        self.custom_model = Optimizer.wrap_model(copy.deepcopy(self.vanilla_network), self.strategy,
                                                 print_params=self.print_params)
        self.custom_model.train()
        self.vanilla_network.train()
        torch.autograd.set_detect_anomaly(True)

    def test_a_test_computation_graph_created(self):
        if self.local_rank == 0:
            print(f'vanilla net: \n{strategy_handler.get_model_graph(self.vanilla_network)}\n'
                  f'wrapped net: \n{strategy_handler.get_model_graph(self.custom_model)}')
        self.assertTrue(True)

    # def test_a_wrappers_wrap(self):
    #     self.assertIs(type(self.custom_model.seq[3]), DataParallelLayer)
    #     self.assertIs(type(self.custom_model.seq[4]), ColumnParallelLinear)
    #     self.assertIs(type(self.custom_model.seq[5]), RowParallelLinear)
    #
    # def test_b_layers_are_stored_correctly(self):
    #     # test row and column parallel
    #     if self.local_rank == 0:
    #         print(f'now testing column parallel')
    #     # test weights
    #     layer1 = self.vanilla_network.seq[4].weight.data
    #     layer2 = self.custom_model.seq[4].get_master_weight()
    #     if self.print_params:
    #         if self.local_rank == 0:
    #             print(f'rank 0 vanilla: shape: {layer1.shape},\n {layer1}\ncustom: shape: {layer2.shape},\n {layer2}')
    #     self.assertTrue(torch.equal(layer1, layer2))
    #     # test biases
    #     if self.vanilla_network.seq[4].bias is not None:
    #         layer1_bias = self.vanilla_network.seq[4].bias.data
    #         layer2_bias = self.custom_model.seq[4].bias.data
    #         if self.print_params:
    #             print(f'rank {self.local_rank}================\n vanilla: shape: {layer1_bias.shape},\n {layer1_bias}'
    #                   f'\ncustom: shape: {layer2_bias.shape},\n {layer2_bias}')
    #         self.assertTrue(False not in torch.eq(layer1_bias[self.local_rank], layer2_bias))
    #     if self.local_rank == 0:
    #         print(f'now testing row parallel')
    #     # test weights
    #     layer1 = self.vanilla_network.seq[5].weight.data
    #     layer2 = self.custom_model.seq[5].get_master_weight()
    #     if self.print_params:
    #         if self.local_rank == 0:
    #             print(f'rank 0 vanilla: shape: {layer1.shape},\n {layer1}\ncustom: shape: {layer2.shape},\n {layer2}')
    #     self.assertTrue(torch.equal(layer1, layer2))
    #     # test biases
    #     if self.vanilla_network.seq[5].bias is not None:
    #         layer1_bias = self.vanilla_network.seq[5].bias.data
    #         layer2_bias = self.custom_model.seq[5].bias.data
    #         if self.print_params:
    #             print(f'rank {self.local_rank}================\n vanilla: shape: {layer1_bias.shape},\n {layer1_bias}'
    #                   f'\ncustom: shape: {layer2_bias.shape},\n {layer2_bias}')
    #         self.assertTrue(False not in torch.eq(layer1_bias, layer2_bias))
    #
    # def test_c_model_parallel(self):
    #     x = torch.rand((4, 4)).to(self.local_rank)
    #     # testing column parallel
    #     x_vanilla = self.vanilla_network.seq[4](x)
    #     x_custom = self.custom_model.seq[4](x)
    #     if self.print_params:
    #         print(f'rank {self.local_rank} vanilla: shape: {x_vanilla.data.shape},\n {x_vanilla.data}\ncustom: shape: '
    #               f'{x_custom.shape},\n {x_custom.data}')
    #     # self.assertTrue(torch.equal(x_vanilla[:, self.local_rank].data.reshape(4, 1), x_custom.data))
    #     self.assertTrue(torch.equal(x_vanilla.data, x_custom.data))
    #     # testing row parallel
    #     x_vanilla = self.vanilla_network.seq[5](x)
    #     x_custom = self.custom_model.seq[5](x)
    #     if self.print_params:
    #         print(f'rank {self.local_rank} vanilla: shape: {x_vanilla.data.shape},\n {x_vanilla.data}\ncustom: shape:'
    #               f' {x_custom.shape},\n {x_custom.data}')
    #     # uncomment this to see that due to all reduce, output between vanilla and split is not identical
    #     # self.assertTrue(False not in torch.eq(x_vanilla.data, x_custom.data))
    #     self.assertTrue(False not in torch.isclose(x_vanilla.data, x_custom.data, equal_nan=True))
    #
    # def test_d_compare_ddp_and_vanilla(self):
    #     vanilla = MnistMLP().to(self.local_rank)
    #     ddp_oss = DDP(copy.deepcopy(vanilla))
    #     ddp_no_oss = DDP(copy.deepcopy(vanilla))
    #     loss = nn.CrossEntropyLoss()
    #     optim_kwargs = {'lr': 0.5}
    #     vanilla_optim = SGD(vanilla.parameters(), **optim_kwargs)
    #     oss_optim = OSS(ddp_oss.parameters(), SGD, **optim_kwargs)
    #     no_oss_optim = SGD(ddp_no_oss.parameters(), **optim_kwargs)
    #     # generate dummy data
    #     x = torch.rand((4, 28 * 28)).to(self.local_rank)
    #     targets = torch.randint(high=10, size=(4,)).to(self.local_rank)
    #     ddp_x, ddp_targets = x[self.local_rank, :], targets[self.local_rank].view(1)
    #     # run it through both networks and check the output and loss are the same
    #     for i in range(20):
    #         oss_optim.zero_grad()
    #         no_oss_optim.zero_grad()
    #         vanilla_optim.zero_grad()
    #         x_vanilla = vanilla(x)
    #         x_oss = ddp_oss(copy.deepcopy(ddp_x))
    #         x_no_oss = ddp_no_oss(copy.deepcopy(ddp_x))
    #         if self.print_params:
    #             print(f'rank = {self.local_rank} iteration: {i} \nx_vanilla = \n{x_vanilla[self.local_rank, :]}\n'
    #                   f'x_oss = \n{x_oss} \n'
    #                   f'x_no_oss = \n{x_no_oss}')
    #         self.assertTrue(False not in torch.isclose(x_vanilla[self.local_rank, :], x_no_oss, rtol=1e-3, atol=1e-4),
    #                         msg=f'vanilla does not match ddp in rank {self.local_rank} (iteration {i})'
    #                             f'vanilla: \n{x_vanilla[self.local_rank, :]}, ddp: \n{x_no_oss}')
    #         self.assertTrue(False not in torch.isclose(x_oss, x_no_oss, rtol=1e-4, atol=1e-7),
    #                         msg=f'ddp with oss does not match ddp without oss in rank {self.local_rank}')
    #         vanilla_loss = loss(x_vanilla, targets)
    #         oss_loss = loss(x_oss, ddp_targets)
    #         no_oss_loss = loss(x_no_oss, ddp_targets)
    #         # self.assertEqual(loss_with_oss, loss_vanil)
    #         self.assertEqual(no_oss_loss, oss_loss)
    #         # calculate gradients and run the optimization step
    #         vanilla_loss.backward()
    #         oss_loss.backward()
    #         no_oss_loss.backward()
    #         vanilla_optim.step()
    #         oss_optim.step()
    #         no_oss_optim.step()
    #
    # def test_e_all_parallelization_options_covered(self):
    #     my_msg = "not all parallelization options covered in {} layer tweaks randomization method.\n" \
    #              "data parallel = {}, row parallel = {}, column parallel = {}"
    #     data_parallel, row_parallel, column_parallel = False, False, False
    #     for i in range(3000):
    #         new_strat = strategy_handler.randomize_strategy(self.vanilla_strategy, random_layer_amount=True)
    #         for layer in new_strat.values():
    #             if layer.gather_input or layer.split_output:
    #                 data_parallel = True
    #             elif layer.row_linear:
    #                 row_parallel = True
    #             elif layer.column_linear:
    #                 column_parallel = True
    #     self.assertTrue(data_parallel and row_parallel and column_parallel,
    #                     msg=my_msg.format('multi', data_parallel, row_parallel, column_parallel))
    #     data_parallel, row_parallel, column_parallel = False, False, False
    #     for i in range(3000):
    #         new_strat = strategy_handler.randomize_strategy(self.vanilla_strategy)
    #         for layer in new_strat.values():
    #             if layer.gather_input or layer.split_output:
    #                 data_parallel = True
    #             elif layer.row_linear:
    #                 row_parallel = True
    #             elif layer.column_linear:
    #                 column_parallel = True
    #     self.assertTrue(data_parallel and row_parallel and column_parallel,
    #                     msg=my_msg.format('single', data_parallel, row_parallel, column_parallel))
    #
    # def test_e_mathematics_is_kept(self):
    #     loss_model = nn.CrossEntropyLoss()
    #     optim_kwargs = {'lr': 1}
    #     # vanilla_optim = OSS(self.vanilla_network.parameters(), SGD, **optim_kwargs)
    #     # custom_optim = OSS(self.custom_model.parameters(), SGD, **optim_kwargs)
    #     vanilla_optim = SGD(self.vanilla_network.parameters(), **optim_kwargs)
    #     custom_optim = SGD(self.custom_model.parameters(), **optim_kwargs)
    #     # generate dummy data
    #     x = torch.rand((4, 28 * 28)).to(self.local_rank)
    #     targets = torch.randint(high=10, size=(4,)).to(self.local_rank)
    #     split_x = x.clone()[self.local_rank, :]
    #     # run it through both networks and check the output and loss are the same
    #     for i in range(10):
    #         vanilla_optim.zero_grad()
    #         custom_optim.zero_grad()
    #         x_vanilla = self.vanilla_network(x)
    #         x_custom = self.custom_model(split_x)
    #         if self.local_rank == 0 and self.print_params:
    #             print(f'x_vanilla = {x_vanilla}\nx_custom = {x_custom}')
    #         message = f'iteration: {i}, rank: {self.local_rank} \nx_vanilla = \n{x_vanilla}\nx_custom = \n{x_custom}'
    #         self.assertTrue(False not in torch.isclose(x_vanilla, x_custom,
    #                                                    equal_nan=True, rtol=(1e-4 * (i + 1)), atol=(1e-6 * (i + 1))),
    #                         msg=message)
    #         # calculate gradients and run the optimization step
    #         loss_model(x_vanilla, targets).backward()
    #         loss_model(x_custom, targets).backward()
    #         # todo delete
    #         # for j in (0, 1, 2):
    #         #     torch.distributed.all_reduce(self.custom_model.seq[j].weight.grad)
    #         #     torch.distributed.all_reduce(self.custom_model.seq[j].bias.grad)
    #         Optimizer.sync_gradients(self.custom_model)
    #         vanilla_optim.step()
    #         custom_optim.step()

    # def mini_print(self, i, vanilla_grads, custom_grads):
    #     if self.local_rank == 0:
    #         return f'\nrank = {self.local_rank}, layer number = {i}\nvanilla grad =\n{vanilla_grads}' \
    #                f'\ncustom grad =\n{custom_grads}'
    #
    # def test_f_backwards_is_kept(self):
    #     loss_model = nn.CrossEntropyLoss()
    #     # generate dummy data
    #     x = torch.rand((4, 28 * 28)).to(self.local_rank)
    #     targets = torch.randint(high=10, size=(4,)).to(self.local_rank)
    #     split_x = x[self.local_rank, :]
    #     # run it through both networks
    #     x_vanilla = self.vanilla_network(x)
    #     x_custom = self.custom_model(split_x)
    #     if self.local_rank == 0 and self.print_params:
    #         print(f'x_vanilla = \n{x_vanilla}\nx_custom = \n{x_custom}')
    #     loss_vanilla = loss_model(x_vanilla, targets)
    #     loss_custom = loss_model(x_custom, targets)
    #     self.assertEqual(loss_custom, loss_vanilla)
    #     loss_vanilla.backward()
    #     loss_custom.backward()
    #     # manually check that layers gradients are computed correctly in reversed order
    #     vanilla_grad, custom_grad = self.vanilla_network.seq[7].bias.grad, self.custom_model.seq[7].bias.grad
    #     self.assertTrue(False not in torch.isclose(vanilla_grad, custom_grad),
    #                     msg=self.mini_print(7, vanilla_grad, custom_grad))
    #
    #     vanilla_grad, custom_grad = self.vanilla_network.seq[6].weight.grad, self.custom_model.seq[6].weight.grad
    #     self.assertTrue(False not in torch.isclose(vanilla_grad, custom_grad),
    #                     msg=self.mini_print(6, vanilla_grad, custom_grad))
    #
    #     vanilla_grad = self.vanilla_network.seq[5].weight.grad[:, self.local_rank]
    #     custom_grad = self.custom_model.seq[5].weight.grad.view_as(vanilla_grad)
    #     self.assertTrue(False not in torch.isclose(vanilla_grad, custom_grad),
    #                     msg=self.mini_print(5, vanilla_grad, custom_grad))
    #
    #     vanilla_grad = self.vanilla_network.seq[4].weight.grad[self.local_rank, :]
    #     custom_grad = self.custom_model.seq[4].weight.grad.view_as(vanilla_grad)
    #     self.assertTrue(False not in torch.isclose(vanilla_grad, custom_grad),
    #                     msg=self.mini_print(4, vanilla_grad, custom_grad))
    #
    #     # torch.distributed.all_reduce(self.custom_model.seq[3].weight.grad)
    #     vanilla_grad, custom_grad = self.vanilla_network.seq[3].weight.grad, self.custom_model.seq[3].weight.grad
    #     self.assertTrue(False not in torch.isclose(vanilla_grad, custom_grad),
    #                     msg=self.mini_print(3, vanilla_grad, custom_grad))
    #
    #     torch.distributed.all_reduce(self.custom_model.seq[2].weight.grad)
    #     vanilla_grad, custom_grad = self.vanilla_network.seq[2].weight.grad, self.custom_model.seq[2].weight.grad
    #     self.assertTrue(False not in torch.isclose(vanilla_grad, custom_grad),
    #                     msg=self.mini_print(7, vanilla_grad, custom_grad))
    #
    #     torch.distributed.all_reduce(self.custom_model.seq[1].weight.grad)
    #     vanilla_grad, custom_grad = self.vanilla_network.seq[1].weight.grad, self.custom_model.seq[1].weight.grad
    #     self.assertTrue(False not in torch.isclose(vanilla_grad, custom_grad),
    #                     msg=self.mini_print(7, vanilla_grad, custom_grad))
    #
    #     torch.distributed.all_reduce(self.custom_model.seq[0].weight.grad)
    #     custom_grad, vanilla_grad = self.custom_model.seq[0].weight.grad, self.vanilla_network.seq[0].weight.grad
    #     self.assertTrue(False not in torch.isclose(vanilla_grad, custom_grad),
    #                     msg=self.mini_print(7, vanilla_grad, custom_grad))

    # def test_f_mcmc_search_works(self):
    #     loss_model = nn.CrossEntropyLoss()
    #     optim_kwargs = {'lr': 1}
    #     custom_optim = SGD(self.custom_model.parameters(), **optim_kwargs)
    #     # generate dummy data
    #     x = torch.rand((8, 28 * 28)).to(self.local_rank)
    #     targets = torch.randint(high=10, size=(8,)).to(self.local_rank)
    #     # split batch
    #     mini_batch_size = x.shape[0] / dist.get_world_size(group=None)
    #     mini_batch_indices = [int(mini_batch_size * dist.get_rank()), int(mini_batch_size * (dist.get_rank() + 1))]
    #     split_x = x[mini_batch_indices[0]: mini_batch_indices[1], :]
    #     new_strategy, latency = Optimizer.find_optimal_strategy(copy.deepcopy(self.vanilla_network), custom_optim,
    #                                                             loss_model, split_x, targets, budget=100, alpha=0.5)
    #     if self.local_rank == 0:
    #         print(f'latency: {latency} strategy:\n{new_strategy}')

if __name__ == "__main__":
    unittest.main()



