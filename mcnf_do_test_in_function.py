import random
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats
import pickle

from instance_mcnf import generate_instance, mutate_instance
from mcnf import *
from mcnf_continuous import run_DW_Fenchel_model, knapsack_model_solver, compute_possible_paths_per_commodity
# from simulated_annealing import simulated_annealing_unsplittable_flows
# from read_telesat_data import read_data
# from mcnf_dynamic import is_correct_path
# from simulated_annealing_LP import simulated_annealing_LP
# from simulated_annealing_LP2 import simulated_annealing_LP2, simulated_annealing_LP3
# from VNS_masri import VNS_masri
# from ant_colony import ant_colony_optimiser
# from lagrangian_decomposition import lagrangian_gradient_descent, lagrangian_decompostion_cutting_plane, lagrangian_differential_evolution, lagrangian_gradient_descent_friction, lagrangian_BFGS


def f():

    # Here you choose the setting of the instances and of the solvers

    # Size of the graph
    size_list = [145]*10
    # size_list = [3, 4, 5, 6, 7, 9, 10, 12, 13, 15]
    # size_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20]
    # size_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20]
    # size_list = [30, 50, 70, 100, 130, 160, 200, 250, 300, 400]
    size_list = np.array(size_list)
    # size_list = size_list**2

    # Capacity of the arcs of the graph
    # capacity_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    capacity_list = [1000] * len(size_list)
    # capacity_list = [10000] * len(size_list)
    # capacity_list = [3] * len(size_list)

    # Threshold of actualisation of the heuristic
    actulisation_threshold_list = None
    # actulisation_threshold_list = 2 ** (np.arange(10) + 4)

    # Upper bound on the size of the commodities
    # max_demand_list = [10, 20 , 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    max_demand_list = [1000] * len(size_list)
    # max_demand_list = [1500] * len(size_list)
    # max_demand_list = [2] * len(size_list)
    # max_demand_list = [capa / 5 for capa in capacity_list]
    # max_demand_list = [int(np.sqrt(capa)) for capa in capacity_list]

    # creating the parameters for the instances; note that grid graphs and random connected graphs dont use the size parameter in the same way
    test_list = []
    for size, capacity, max_demand in zip(size_list, capacity_list, max_demand_list):
        # test_list += [("grid", (size, size, size, 2*size, capacity, capacity), {"max_demand" : max_demand, "smaller_commodities" : False})]
        # test_list += [("grid", (size, size, size, 2*size, capacity, capacity), {"max_demand" : max_demand, "smaller_commodities" : True})]
        test_list += [("random_connected", (size, 5/size, int(size * 0.1), capacity), {"max_demand" : max_demand, "smaller_commodities" : False})]

    # Choice of the tested algorithms
    tested_algorithms = []
    # tested_algorithms.append("heuristic")
    # tested_algorithms.append("heuristic_arc_path")
    # tested_algorithms.append("heuristic_arc_path_with_cut")
    # tested_algorithms.append("heuristic_knapsack")
    # tested_algorithms.append("heuristic_congestion")
    # tested_algorithms.append("approximation")
    # tested_algorithms.append("approximation_arc_path")
    # tested_algorithms.append("approximation_congestion")
    # tested_algorithms.append("heuristic_unsorted")
    # tested_algorithms.append("heuristic_unsorted_arc_path")
    # tested_algorithms.append("new_approx")
    # tested_algorithms.append("simulated_annealing")
    # tested_algorithms.append("simulated_annealing_LP")
    # tested_algorithms.append("MILP_solver")
    # tested_algorithms.append("MILP_solver_restricted")
    # tested_algorithms.append("VNS_masri")
    # tested_algorithms.append("VNS_masri2")
    # tested_algorithms.append("ant_colony")
    # tested_algorithms.append("lagrangian_gradient_descent")
    # tested_algorithms.append("lagrangian_gradient_descent_friction")
    # tested_algorithms.append("lagrangian_decompostion_cutting_plane")
    # tested_algorithms.append("lagrangian_differential_evolution")
    # tested_algorithms.append("lagrangian_BFGS")
    # tested_algorithms.append("DW_Fenchel")
    tested_algorithms.append("knapsack_model")


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

        # instance_list = read_data(1)
        # graph, commodity_list = instance_list[-1]

        # instance_file = open("/home/pc-francois/Bureau/MCNF_solver/instance_files/commodity_scaling_dataset/grid_240_1000_31_0.p", 'rb')
        # graph, commodity_list = pickle.load(instance_file)
        # instance_file.close()

        total_demand = sum([c[2] for c in commodity_list])
        nb_nodes = len(graph)
        nb_commodities = len(commodity_list)
        print("total_demand is : ", total_demand)
        print("nb_commodities = ", nb_commodities)
        nb_commodity_list.append(len(commodity_list))
        nb_node_list.append(nb_nodes)
        print(graph)

        #Setting default Threshold for the heuristic
        if actulisation_threshold_list is None:
            actualisation_threshold = None
        else:
            actualisation_threshold = actulisation_threshold_list[i]

        possible_paths_per_commodity = compute_possible_paths_per_commodity(graph, commodity_list, 4)
        for commodity_index in range(nb_commodities):
            possible_paths_per_commodity[commodity_index].append(initial_solution[commodity_index])

        import cProfile, pstats, io
        from pstats import SortKey
        pr = cProfile.Profile()
        pr.enable()
        # ... do something ...

        # Applying the algorithms present in tested_algorithms
        for algorithm_name in tested_algorithms:
            print("Running {}".format(algorithm_name))
            temp = time.time()

            if algorithm_name == "heuristic" : a = randomized_rounding_heuristic(graph, commodity_list, actualisation_threshold=actualisation_threshold, verbose=1)
            if algorithm_name == "heuristic_arc_path" : a = randomized_rounding_heuristic_arc_path(graph, commodity_list, actualisation_threshold=actualisation_threshold, verbose=1)
            if algorithm_name == "heuristic_unsorted_arc_path" : a = randomized_rounding_heuristic_arc_path(graph, commodity_list, actualisation_threshold=actualisation_threshold, sorted_commodities=False, verbose=1)
            if algorithm_name == "approximation_arc_path" : a = randomized_rounding_heuristic_arc_path(graph, commodity_list, actualisation=False, verbose=1)
            if algorithm_name == "heuristic_arc_path_with_cut" : a = randomized_rounding_heuristic_arc_path(graph, commodity_list, actualisation_threshold=actualisation_threshold, knapsack_cut=True, verbose=1)
            if algorithm_name == "heuristic_knapsack" : a = knapsack_randomized_rounding(graph, commodity_list, actualisation_threshold=actualisation_threshold, verbose=1)
            if algorithm_name == "heuristic_congestion" : a = randomized_rounding_heuristic(graph, commodity_list, actualisation_threshold=actualisation_threshold, linear_objectif="congestion", verbose=1)
            if algorithm_name == "heuristic_unsorted" : a = randomized_rounding_heuristic(graph, commodity_list, actualisation_threshold=actualisation_threshold, sorted_commodities=False, verbose=1)
            if algorithm_name == "new_approx" : a = randomized_rounding_heuristic(graph, commodity_list, actualisation_threshold=actualisation_threshold, proof_constaint=True)
            if algorithm_name == "approximation" : a = randomized_rounding_heuristic(graph, commodity_list, actualisation=False, sorted_commodities=False, verbose=3)
            if algorithm_name == "approximation_congestion" : a = randomized_rounding_heuristic(graph, commodity_list, actualisation=False, sorted_commodities=False, linear_objectif="congestion")
            if algorithm_name == "simulated_annealing" : a = simulated_annealing_unsplittable_flows(graph, commodity_list, nb_iterations=int(len(commodity_list)**1.5)*2, verbose=1)
            if algorithm_name == "simulated_annealing_LP" : a = simulated_annealing_LP3(graph, commodity_list, nb_iterations= 200000)
            if algorithm_name == "MILP_solver" : a = gurobi_unsplittable_flows(graph, commodity_list, verbose=1, time_limit=1000)
            if algorithm_name == "MILP_solver_restricted" : a = gurobi_unsplittable_flows_restricted(graph, commodity_list, nb_paths_per_commodity=4, added_path_per_commodity=initial_solution, verbose=1, time_limit=500, flow_penalisation=0)
            if algorithm_name == "VNS_masri" : a = VNS_masri(graph, commodity_list, nb_iterations=50)
            if algorithm_name == "VNS_masri2" : a = VNS_masri(graph, commodity_list, nb_iterations=int(len(commodity_list)**1.5)//4 * 1000, amelioration=True, verbose=1)
            if algorithm_name == "ant_colony" : a = ant_colony_optimiser(graph, commodity_list, 3000, verbose=1)
            if algorithm_name == "lagrangian_gradient_descent" : a = lagrangian_gradient_descent(graph, commodity_list, 1000, 0.1)
            if algorithm_name == "lagrangian_gradient_descent_friction" : a = lagrangian_gradient_descent_friction(graph, commodity_list, 1000, 0.1)
            if algorithm_name == "lagrangian_decompostion_cutting_plane" : a = lagrangian_decompostion_cutting_plane(graph, commodity_list, 300)
            if algorithm_name == "lagrangian_differential_evolution" : a = lagrangian_differential_evolution(graph, commodity_list, 300)
            if algorithm_name == "lagrangian_BFGS" : a = lagrangian_BFGS(graph, commodity_list)
            if algorithm_name == "DW_Fenchel" : a = run_DW_Fenchel_model(graph, commodity_list, flow_penalisation=0, nb_iterations=2000, possible_paths_per_commodity=possible_paths_per_commodity, separation_options=(True, True, True), verbose=1)
            if algorithm_name == "knapsack_model" : a = knapsack_model_solver(graph, commodity_list, flow_penalisation=0, possible_paths_per_commodity=possible_paths_per_commodity, stabilisation="momentum", verbose=1)

            # commodity_path_list, total_overload = a
            commodity_path_list = initial_solution
            computing_time = time.time() - temp
            # results_dict[algorithm_name][0].append(total_overload / total_demand)
            # results_dict[algorithm_name][1].append(computing_time)
            #
            # # Print the results
            # print("Performance = ", total_overload / total_demand)
            # print("Computing_time = ", computing_time)

            use_graph = [{neighbor : 0 for neighbor in graph[node]} for node in range(len(graph))]
            for commodity_index, path in enumerate(commodity_path_list):
                update_graph_capacity(use_graph, path, -commodity_list[commodity_index][2])

            overload_graph = [{neighbor : max(0, use_graph[node][neighbor] - graph[node][neighbor]) for neighbor in graph[node]} for node in range(len(graph))]
            overload = sum([sum(dct.values()) for dct in overload_graph])
            print("Overload = ", overload)
            print(graph)
            print()


        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())


            # use_graph = [{neighbor : 0 for neighbor in graph[node]} for node in range(nb_nodes)]
            # for commodity_index, path in enumerate(commodity_path_list):
            #     update_graph_capacity(use_graph, path, -commodity_list[commodity_index][2])
            #
            # served_demand = total_demand
            # unserved_users = 0
            # demand_list = [c[2] for c in commodity_list]
            # sorted_commodity_indices = sorted(list(range(nb_commodities)), key=lambda x : demand_list[x])
            # for commodity_index in sorted_commodity_indices:
            #     path = commodity_path_list[commodity_index]
            #
            #     for node_index in range(len(path)-1):
            #         node, neighbor = path[node_index], path[node_index+1]
            #         if use_graph[node][neighbor] > graph[node][neighbor]:
            #             served_demand -= commodity_list[commodity_index][2]
            #             update_graph_capacity(use_graph, path, commodity_list[commodity_index][2])
            #             unserved_users += 1
            #             break
            #
            # print("xxxxxxxxx", unserved_users, total_demand - served_demand)

            # overload_graph = [{neighbor : max(0, use_graph[node][neighbor] - graph[node][neighbor]) for neighbor in graph[node]} for node in range(len(graph))]
            # total_overload = sum([sum(dct.values()) for dct in overload_graph])



    # Store the created instance
    # global_path = "/home/francois/Desktop/"
    # reuslt_file = open(global_path + "MCNF_solver/cut_results.p", "wb" )
    # pickle.dump(results_dict, reuslt_file)
    # reuslt_file.close()


    # Curves drawing

    # abscisse = np.array(nb_commodity_list)
    # abscisse = nb_node_list
    abscisse = actulisation_threshold_list
    # abscisse = size_list

    # res = [0.0015133333333333333, 0.001956521739130435, 0.0017166666666666667, 0.0022316666666666665, 0.0028349397590361446, 0.0027141843971631207, 0.0032712707182320443, 0.0019154676258992807, 0.0018506451612903226, 0.001931353919239905]
    # time = [0.13064312934875488, 0.2518649101257324, 0.46654462814331055, 1.0491209030151367, 1.5732910633087158, 3.8554956912994385, 6.375101327896118, 9.274698734283447, 15.167050838470459, 26.85559320449829]
    # results_dict["heuristic"] = (res, time)

    for algorithm_name in tested_algorithms:
        res = np.array(results_dict[algorithm_name][0])
        mean = sum(res)/len(res)
        print("Mean "+ algorithm_name+ " = ", mean)
        print("Standard deviation "+ algorithm_name+ " = ", np.sqrt(sum((res-mean)**2)/len(res)))

    colors = {"heuristic" : '#1f77b4', "approximation" : '#ff7f0e', "heuristic_congestion" : '#1f77b4',
                "approximation_congestion" : '#ff7f0e', "simulated_annealing" : '#2ca02c', "MILP_solver" : '#d62728',
                "new_approx" : '#9467bd', "iterated_heuristic_unsorted" : "#000000"}

    fig = plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    ax = fig.gca()
    for algorithm_name in tested_algorithms:
        plt.plot(abscisse, results_dict[algorithm_name][0], label=algorithm_name+"_results", color=colors[algorithm_name])
    ax.legend()
    plt.show()

    fig = plt.figure()
    plt.xscale("log")
    plt.yscale("log")
    ax = fig.gca()
    for algorithm_name in tested_algorithms:
        plt.plot(abscisse, results_dict[algorithm_name][1], label=algorithm_name+"_c_time", color=colors[algorithm_name])
    ax.legend()
    plt.plot(abscisse, abscisse ** 1.5 /25000, "r")
    plt.show()

if __name__ == "__main__":
    f()
