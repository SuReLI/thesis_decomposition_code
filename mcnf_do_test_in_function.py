import random
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats
import pickle

from instance_mcnf import generate_instance
from mcnf_continuous import run_DW_Fenchel_model, knapsack_model_solver, compute_possible_paths_per_commodity

def f():

    # Here you choose the setting of the instances and of the solvers

    # Size of the graph
    size_list = [145]*10
    size_list = np.array(size_list)

    # Capacity of the arcs of the graph
    capacity_list = [1000] * len(size_list)


    # Upper bound on the size of the commodities
    max_demand_list = [1000] * len(size_list)

    # creating the parameters for the instances; note that grid graphs and random connected graphs dont use the size parameter in the same way
    test_list = []
    for size, capacity, max_demand in zip(size_list, capacity_list, max_demand_list):
        # test_list += [("grid", (size, size, size, 2*size, capacity, capacity), {"max_demand" : max_demand, "smaller_commodities" : False})]
        # test_list += [("grid", (size, size, size, 2*size, capacity, capacity), {"max_demand" : max_demand, "smaller_commodities" : True})]
        test_list += [("random_connected", (size, 5/size, int(size * 0.1), capacity), {"max_demand" : max_demand, "smaller_commodities" : False})]

    # options for the Fenchel subproblem
    separation_options = (True, True, True)

    # stabilisation method for the Dantzig-Wolfe decomposition
    stabilisation = "momentum"

    # Choice of the tested algorithms
    tested_algorithms = []
    tested_algorithms.append("DW_Fenchel")
    # tested_algorithms.append("knapsack_model")


    results_dict = {algorithm_name : ([],[]) for algorithm_name in tested_algorithms}

    i = -1
    nb_commodity_list = []
    nb_node_list = []
    # graph, commodity_list, initial_solution, origin_list = generate_instance(*test_list[0], max_demand=max_demand_list[0])

    for graph_type, graph_generator_inputs, demand_generator_inputs in test_list:
        i += 1
        print("##############################  ", i,"/",len(test_list))

        # Instance generation
        nb_nodes = size_list[0]
        graph, commodity_list, initial_solution, origin_list = generate_instance(graph_type, graph_generator_inputs, demand_generator_inputs, nb_capacity_modifitcations=100 * nb_nodes)

        total_demand = sum([c[2] for c in commodity_list])
        nb_nodes = len(graph)
        nb_commodities = len(commodity_list)
        print("total_demand is : ", total_demand)
        print("nb_commodities = ", nb_commodities)
        nb_commodity_list.append(len(commodity_list))
        nb_node_list.append(nb_nodes)
        print(graph)


        possible_paths_per_commodity = compute_possible_paths_per_commodity(graph, commodity_list, 4)
        for commodity_index in range(nb_commodities):
            possible_paths_per_commodity[commodity_index].append(initial_solution[commodity_index])

        # Applying the algorithms present in tested_algorithms
        for algorithm_name in tested_algorithms:
            print("Running {}".format(algorithm_name))
            temp = time.time()

            if algorithm_name == "DW_Fenchel" : run_DW_Fenchel_model(graph, commodity_list, nb_iterations=2000, possible_paths_per_commodity=possible_paths_per_commodity, separation_options=separation_options, verbose=1)
            if algorithm_name == "knapsack_model" : knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, stabilisation=stabilisation, verbose=1)

            computing_time = time.time() - temp
            print("Computing_time = ", computing_time)
