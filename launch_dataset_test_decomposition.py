import random
import numpy as np
import time
import pickle
from multiprocessing import Process, Manager

from instance_mcnf import generate_instance, mutate_instance
from mcnf_continuous import *

def launch_dataset(global_path, dataset_name, algorithm_list, nb_repetitions, nb_workers, duration_before_timeout):
    # Launches all the algorithms to test on the instance present in the dataset directory
    # The number of time algorithms are lauched is decided with nb_repetitions

    # Open the file containing the name of the instances
    instance_name_file = open(global_path + "/MCNF_solver/instance_files_decomposition/" + dataset_name + "/instance_name_file.p", "rb" )
    instance_name_list = pickle.load(instance_name_file)
    instance_name_file.close()

    log_file = open(global_path + "/MCNF_solver/log_file.txt", 'w')
    log_file.write("Start\n")
    log_file.close()

    manager = Manager()
    result_dict = {algorithm_name : {instance_name : [None]*nb_repetitions for instance_name in instance_name_list} for algorithm_name in algorithm_list}
    worker_list = []
    # list containing the task to be done
    computation_list = [(repetition_index, instance_index, instance_name, algorithm_name) for repetition_index in range(nb_repetitions)
                                                                                            for instance_index, instance_name in enumerate(instance_name_list)
                                                                                                for algorithm_name in algorithm_list]

    # main loop : checking if processes are finished, creating new ones ...
    while len(computation_list) + len(worker_list) > 0:

        remaining_worker_list = []
        for process, start_time, return_list, computation_info in worker_list:
            repetition_index, instance_index, instance_name, algorithm_name = computation_info

            if not process.is_alive(): # if a process has ended, store the results
                result_dict[algorithm_name][instance_name][repetition_index] = (list(return_list), time.time() - start_time)

            elif time.time() > start_time + duration_before_timeout: # if a process has been running for more than the allowed time, kills it and store the results
                process.terminate()
                result_dict[algorithm_name][instance_name][repetition_index] = (list(return_list), duration_before_timeout)

            else: # else, keep the process in the worker list
                remaining_worker_list.append((process, start_time, return_list, computation_info))

        worker_list = remaining_worker_list

        # create a process if there are still task to be done and if there is space in the worker list
        if len(worker_list) < nb_workers and len(computation_list) > 0:
            computation_info = computation_list.pop(0)
            repetition_index, instance_index, instance_name, algorithm_name = computation_info

            print_string = "repetition : {0}/{1}, instance : {2}/{3}, algorithm : {4}".format(repetition_index, nb_repetitions, instance_index, len(instance_name_list), algorithm_name)
            instance_file_path = global_path + "/MCNF_solver/instance_files_decomposition/" + dataset_name + "/" + instance_name + ".p"
            return_list = manager.list()

            # lauching the process on a task
            process = Process(target=launch_solver_on_instance, args=(instance_file_path, algorithm_name, print_string, global_path, return_list))
            start_time = time.time()
            process.start()
            worker_list.append((process, start_time, return_list, computation_info))


    # Write the results in a file
    result_file = open(global_path + "/MCNF_solver/instance_files_decomposition/" + dataset_name + "/result_file.p", "wb" )
    pickle.dump(result_dict, result_file)
    result_file.close()

    import datetime
    log_file = open(global_path + "/MCNF_solver/log_file.txt", 'a')
    log_file.write(datetime.datetime.now().__str__())
    log_file.close()


def launch_solver_on_instance(instance_file_path, algorithm_name, print_string, global_path, return_list):
    # Lauch the algorithm named algortihm_name on the instance store in the file at instance_file_path

    print(print_string)

    # Read the instance in the instance file
    instance_file = open(instance_file_path, "rb" )
    graph, commodity_list, solution = pickle.load(instance_file)
    instance_file.close()

    total_demand = sum([commodity[2] for commodity in commodity_list])
    nb_commodities = len(commodity_list)
    nb_nodes = len(graph)

    #  creates a list of usable path for each commodity: the 4 shortest paths for each commodity
    possible_paths_per_commodity = compute_possible_paths_per_commodity(graph, commodity_list, 4)
    for commodity_index in range(nb_commodities):
        possible_paths_per_commodity[commodity_index].append(solution[commodity_index])


    temp = time.time()
    l = []

    # Launch the chosen algorithm
    if algorithm_name == "Fenchel" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(False, True, False), verbose=0)
    if algorithm_name == "Fenchel no preprocessing" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(False, False, False), verbose=0)
    if algorithm_name == "DW-Fenchel" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(True, True, False), verbose=0)
    if algorithm_name == "DW-Fenchel iterative" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(True, True, True), verbose=0)
    if algorithm_name == "DW-Fenchel no preprocessing" : run_DW_Fenchel_model(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, separation_options=(True, False, False), verbose=0)
    if algorithm_name == "DW" : knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, stabilisation="", verbose=0)
    if algorithm_name == "DW momentum" : knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, stabilisation="momentum", verbose=0)
    if algorithm_name == "DW interior" : knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=possible_paths_per_commodity, bounds_and_time_list=return_list, stabilisation="interior_point", verbose=0)

    computing_time = time.time() - temp

    log_file = open(global_path + "/MCNF_solver/log_file.txt", 'a')
    log_file.write("Finished : " + instance_file_path + ", " + print_string + "\n")
    log_file.close()

    print("Finished")


if __name__ == "__main__":
    # Set the path to the global directory
    global_path = "/home/sureli/Documents/Francois/"
    # global_path = "/home/disc/f.lamothe"
    # global_path = "/home/francois/Desktop"
    # assert False, "Unassigned global_path : Complete global_path with the path to the main directory"

    # Set the number of repetition
    nb_repetitions = 1
    nb_workers = 10
    duration_before_timeout = 60*60

    # list of the algorithm lauched on each dataset
    settings_list = []
    # settings_list.append(("small_dataset", ["Fenchel", "Fenchel no preprocessing", "DW-Fenchel", "DW-Fenchel iterative", 'DW-Fenchel no preprocessing', 'DW', "DW momentum", "DW interior"]))
    # settings_list.append(("graph_scaling_dataset", ["Fenchel", "DW-Fenchel", "DW-Fenchel iterative", 'DW', "DW momentum", "DW interior"]))
    # settings_list.append(("small_dataset_lower_bound", ["Fenchel", "DW-Fenchel", "DW-Fenchel iterative", 'DW', "DW interior"]))
    settings_list.append(("graph_scaling_dataset_lower_bound", ["DW-Fenchel iterative", "DW interior"]))
    settings_list.append(("capacity_scaling_dataset", ["DW-Fenchel iterative", "DW interior"]))


    for dataset_name, algorithm_list in settings_list:
        launch_dataset(global_path, dataset_name, algorithm_list, nb_repetitions, nb_workers, duration_before_timeout)
