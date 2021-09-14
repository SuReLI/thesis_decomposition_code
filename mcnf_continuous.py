import numpy as np
import random
import time
import heapq as hp
import gurobipy

from k_shortest_path import k_shortest_path_algorithm, k_shortest_path_all_destination
from knapsack_cut import ttt, in_out_separation_decomposition, in_out_separation_decomposition_with_preprocessing, separation_decomposition, separation_decomposition_with_preprocessing, penalized_knapsack_optimizer


def create_arc_path_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=0, verbose=0):
    # creates a linear program for the linear relaxation of the unsplittable flow problem based on an arc-path formulation
    nb_commodities = len(commodity_list)
    nb_nodes = len(graph)
    demand_list = [commodity[2] for commodity in commodity_list]
    arc_list = [(node, neighbor) for node in range(nb_nodes) for neighbor in graph[node]]

    # Create optimization model
    model = gurobipy.Model('netflow')
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = verbose
    model.Params.Method = 3

    # Create variables
    path_and_var_per_commodity = [[(path, model.addVar(obj=(len(path) - 1) * flow_penalisation)) for path in possible_paths] for possible_paths in possible_paths_per_commodity]
    overload_var = model.addVars(arc_list, obj=1, name="overload") # overload variables : we want to minimize their sum
    if verbose:
        print("variables created")

    # Convexity constraints :
    convexity_constraint_dict = model.addConstrs((sum(var for path, var in path_and_var_per_commodity[commodity_index]) == 1 for commodity_index in range(nb_commodities)))
    if verbose:
        print("Convexity constraints created")

    # Capacity constraint
    edge_var_sum_dict = {arc : 0 for arc in arc_list}
    for commodity_index, demand in enumerate(demand_list):
        for path, var in path_and_var_per_commodity[commodity_index]:
            for node_index in range(len(path)-1):
                arc = (path[node_index], path[node_index+1])
                edge_var_sum_dict[arc] += var * demand

    capacity_constraint_dict = model.addConstrs((edge_var_sum_dict[arc] - overload_var[arc] <= graph[arc[0]][arc[1]] for arc in arc_list))
    if verbose:
        print("Capacity constraints created")

    return model, (path_and_var_per_commodity, overload_var), (convexity_constraint_dict, capacity_constraint_dict)



def create_knapsack_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=10**-5, verbose=1):
    # creates the linear program obtained after applying a Dantzig-Wolfe decomposition to the capacity constraints
    # of an arc-path formulation of the unsplittable flow problem
    nb_commodities = len(commodity_list)
    nb_nodes = len(graph)
    demand_list = [commodity[2] for commodity in commodity_list]
    arc_list = [(node, neighbor) for node in range(nb_nodes) for neighbor in graph[node]]

    # creates the model for an arc path formulation that will be modified
    model, variables, constraints = create_arc_path_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=flow_penalisation, verbose=0)
    path_and_var_per_commodity, overload_var = variables
    convexity_constraint_dict, capacity_constraint_dict = constraints

    # obtaining a solution of the arc-path formualtion enables us to create a set of valid variable for the Dantzig-Wolfe model
    model.update()
    model.optimize()
    if verbose : print("continuous ObjVal = ", model.ObjVal)

    flow_per_commodity_per_arc = {(arc): [0]*nb_commodities for arc in arc_list}
    for commodity_index, path_and_var in enumerate(path_and_var_per_commodity):
        for path, var in path_and_var:
            for node_index in range(len(path)-1):
                arc = (path[node_index], path[node_index + 1])
                flow_per_commodity_per_arc[arc][commodity_index] += var.X

    # the flow on each arc is decomposed into a convex combination of commodity patterns
    # which will become the set of pattern variables initially allowed on this arc
    pattern_and_cost_per_arc = {}
    for arc in arc_list:
        arc_capacity = graph[arc[0]][arc[1]]
        _, pattern_cost_and_amount_list = separation_decomposition_with_preprocessing(demand_list, flow_per_commodity_per_arc[arc], arc_capacity)
        pattern_and_cost_per_arc[arc] = [(pattern, pattern_cost) for pattern, pattern_cost, amount in pattern_cost_and_amount_list]
        pattern_and_cost_per_arc[arc].append((list(range(nb_commodities)), sum(demand_list) - arc_capacity))

    # removing the uselles parts of the arc-path formualtion
    for constraint in capacity_constraint_dict.values():
        model.remove(constraint)

    for var in overload_var.values():
        model.remove(var)

    # creating the inital pattern variables for each arc
    pattern_var_and_cost_per_arc = {}
    knapsack_convexity_constraint_dict = {}
    for arc in arc_list:
        pattern_var_and_cost_per_arc[arc] = []

        for pattern, pattern_cost in pattern_and_cost_per_arc[arc]:
            pattern_var_and_cost_per_arc[arc].append((pattern, model.addVar(obj=pattern_cost), pattern_cost))

        knapsack_convexity_constraint_dict[arc] = model.addConstr(sum(var for pattern, var, pattern_cost in pattern_var_and_cost_per_arc[arc]) <= 1)

    # constraints linking the flow variables and the pattern varaibles
    capacity_constraint_dict = {arc : [None] * nb_commodities for arc in arc_list}
    for commodity_index, path_and_var in enumerate(path_and_var_per_commodity):
        # print(commodity_index, end='    \r')
        edge_var_sum_dict = {}
        for path, var in path_and_var:
            for node_index in range(len(path)-1):
                arc = (path[node_index], path[node_index+1])
                if arc not in edge_var_sum_dict:
                    edge_var_sum_dict[arc] = 0
                edge_var_sum_dict[arc] += var

        for arc in edge_var_sum_dict:
            knapsack_var_sum = sum(var for pattern, var, pattern_cost in pattern_var_and_cost_per_arc[arc] if commodity_index in pattern)
            capacity_constraint_dict[arc][commodity_index] = model.addConstr((edge_var_sum_dict[arc] - knapsack_var_sum <= 0 ), "capacity")

    if verbose:
        print("Linking constraints created")

    return model, (path_and_var_per_commodity, pattern_var_and_cost_per_arc), (convexity_constraint_dict, knapsack_convexity_constraint_dict, capacity_constraint_dict)


def knapsack_model_solver(graph, commodity_list, possible_paths_per_commodity=None, nb_initial_path_created=4, var_delete_proba=0.3,
                            flow_penalisation=0, nb_iterations=10**5, bounds_and_time_list=[], stabilisation="interior_point", verbose=1):
    # creates a knapsack model and solves it
    nb_nodes = len(graph)
    nb_commodities = len(commodity_list)
    arc_list = [(node, neighbor) for node in range(len(graph)) for neighbor in graph[node]]
    demand_list = [demand for origin, destination, demand in commodity_list]

    if possible_paths_per_commodity is None:
        possible_paths_per_commodity = compute_possible_paths_per_commodity(graph, commodity_list, nb_initial_path_created)

    model, variables, constraints = create_knapsack_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=flow_penalisation, verbose=verbose>1)

    run_knapsack_model(graph, commodity_list, model, constraints, stabilisation, bounds_and_time_list=bounds_and_time_list, verbose=verbose)



def run_knapsack_model(graph, commodity_list, model, constraints, stabilisation, bounds_and_time_list=[], nb_iterations=10**5, verbose=1):
    # column generation process used to solve the linear relaxation of a knapsack model of the unsplittable flow problem
    nb_commodities = len(commodity_list)
    demand_list = [commodity[2] for commodity in commodity_list]
    arc_list = [(node, neighbor) for node in range(len(graph)) for neighbor in graph[node]]
    convexity_constraint_dict, knapsack_convexity_constraint_dict, capacity_constraint_dict = constraints
    starting_time = time.time()
    added_var_list = []
    nb_var_added = 0
    best_dual_bound = None
    used_dual_var_list_per_arc = None

    # parameter of the solver gurobi
    model.Params.Method = 3
    model.Params.OutputFlag = 0
    if stabilisation == "interior_point": # in this stabilisation, the master model is solved approximatly (10**-3 precision) with an interior point method
        model.Params.Method = 2
        model.Params.BarConvTol = 10**-3
        model.Params.Crossover = 0

    for iter_index in range(nb_iterations):
        if verbose : print("iteration : ", iter_index)

        model.update()
        model.optimize()
        if verbose : print("Objective function value : ", model.ObjVal, nb_var_added, len(added_var_list))
        if verbose : print("Runtime : ", model.Runtime)

        # getting the dual variables of the master model
        dual_var_knapsack_convexity_per_arc = {arc : -knapsack_convexity_constraint_dict[arc].Pi for arc in arc_list}
        dual_var_flow_convexity_per_commoditiy = np.array([convexity_constraint_dict[commodity_index].Pi for commodity_index in range(nb_commodities)])
        dual_var_list_per_arc = {arc : np.array([-constraint.Pi if constraint is not None else 0 for constraint in capacity_constraint_dict[arc]]) for arc in arc_list}


        if stabilisation == "momentum" and used_dual_var_list_per_arc is not None: # another stabilisation, dual variables are aggregated through the iterations
            momentum_coeff = 0.8
            used_dual_var_list_per_arc = {arc : momentum_coeff * used_dual_var_list_per_arc[arc] + (1 - momentum_coeff) * dual_var_list_per_arc[arc] for arc in arc_list}
            used_dual_var_knapsack_convexity_per_arc = {arc : momentum_coeff * used_dual_var_knapsack_convexity_per_arc[arc] + (1 - momentum_coeff) * dual_var_knapsack_convexity_per_arc[arc] for arc in arc_list}
            used_dual_var_flow_convexity_per_commoditiy = momentum_coeff * used_dual_var_flow_convexity_per_commoditiy + (1 - momentum_coeff) * dual_var_flow_convexity_per_commoditiy

        else:
            used_dual_var_list_per_arc = dual_var_list_per_arc
            used_dual_var_knapsack_convexity_per_arc = dual_var_knapsack_convexity_per_arc
            used_dual_var_flow_convexity_per_commoditiy = dual_var_flow_convexity_per_commoditiy


        dual_bound = 0
        for commodity_index in range(nb_commodities):
            dual_bound += used_dual_var_flow_convexity_per_commoditiy[commodity_index] * convexity_constraint_dict[commodity_index].Rhs

        nb_var_added = 0
        # for each arc, a pricing problem is solved for the pattern variables, if a a pattern has a small enough reduced cost it is added to the formulation
        for arc in arc_list:
            if verbose: print(arc, end='   \r')

            arc_capacity = graph[arc[0]][arc[1]]

            # pricing problem reolution
            new_pattern, subproblem_objective_value = penalized_knapsack_optimizer(demand_list, arc_capacity, used_dual_var_list_per_arc[arc])
            dual_bound -= subproblem_objective_value

            pattern_cost = max(0, sum(demand_list[commodity_index] for commodity_index in new_pattern) - arc_capacity)
            reduced_cost = -subproblem_objective_value + used_dual_var_knapsack_convexity_per_arc[arc]

            if reduced_cost < -10**-5: # if the best pattern has a small enough reduced cost it is added to the formulation
                nb_var_added += 1
                column = gurobipy.Column()
                column.addTerms(1, knapsack_convexity_constraint_dict[arc])

                for commodity_index in new_pattern:
                    column.addTerms(-1, capacity_constraint_dict[arc][commodity_index])

                new_var = model.addVar(obj=pattern_cost, column=column)
                added_var_list.append(new_var)

        if verbose : print("Nb added var = ", nb_var_added, ", Nb total var = ", len(added_var_list), ", Dual_bound = ", dual_bound)
        bounds_and_time_list.append((model.ObjVal, dual_bound, time.time() - starting_time))

        if best_dual_bound is None or dual_bound > best_dual_bound:
            best_dual_bound = dual_bound

        if abs(model.ObjVal - best_dual_bound) < 10**-2: # the column generation stops if the bounds are close enough of if no new variable can be added to the master model
            break

        if nb_var_added == 0: # the column generation stops if the bounds are close enough of if no new variable can be added to the master model
            if stabilisation == "":
                break
            else: # stabilisations are disabled in the final iterations of the column generation procedure
                stabilisation = ""
                model.Params.Method = 3
                model.Params.BarConvTol = 10**-8
                model.Params.Crossover = -1

    model.update()
    model.optimize()


def run_DW_Fenchel_model(graph, commodity_list, separation_options=None, possible_paths_per_commodity=None, nb_initial_path_created=4, var_delete_proba=0.3,
                            flow_penalisation=0, bounds_and_time_list=[], nb_iterations=10**5, verbose=1):
    # this algorithm implements the new decomposition method highlighted by this code
    # it uses a Dantzig-Wolfe master problem and a Fenchel master problem
    # its subproblem is a Fenchel subproblem with a special normalisation called "directionnal normalisation"
    # the value of the computed bounds after convergence is the same as the one computed by a Dantzig-Wolfe decomposition algorithm
    # this function also implements other decomposition methods such as the Fenchel decomposition depending on the _separation_option chosen
    nb_nodes = len(graph)
    nb_commodities = len(commodity_list)
    arc_list = [(node, neighbor) for node in range(len(graph)) for neighbor in graph[node]]
    demand_list = [demand for origin, destination, demand in commodity_list]
    starting_time = time.time()

    if separation_options is None:
        separation_options = (True, True, True)

    # creates a set of allowed paths for each commodity, not all paths are consdered allowed in the formulation
    if possible_paths_per_commodity is None:
        possible_paths_per_commodity = compute_possible_paths_per_commodity(graph, commodity_list, nb_initial_path_created)

    # creates the two master problems
    inner_model, inner_variables, inner_constraints = create_knapsack_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=flow_penalisation, verbose=verbose>1)
    outter_model, outter_variables, outter_constraints = create_arc_path_model(graph, commodity_list, possible_paths_per_commodity, flow_penalisation=flow_penalisation, verbose=verbose>1)

    inner_path_and_var_per_commodity, inner_pattern_var_and_cost_per_arc = inner_variables
    outter_path_and_var_per_commodity, outter_overload_vars = outter_variables

    outter_flow_var_dict = {arc : [0]*nb_commodities for arc in arc_list}
    for commodity_index, path_and_var in enumerate(outter_path_and_var_per_commodity):
        for path, var in path_and_var:
            for node_index in range(len(path)-1):
                arc = (path[node_index], path[node_index+1])
                outter_flow_var_dict[arc][commodity_index] += var

    inner_flow_var_dict = {arc : [0]*nb_commodities for arc in arc_list}
    for commodity_index, path_and_var in enumerate(inner_path_and_var_per_commodity):
        for path, var in path_and_var:
            for node_index in range(len(path)-1):
                arc = (path[node_index], path[node_index+1])
                inner_flow_var_dict[arc][commodity_index] += var

    # parameters of the Dantzig-Wolfe master model
    inner_model.Params.OutputFlag = 0
    inner_model.Params.Method = 3
    # inner_model.Params.Crossover = 0
    # inner_model.Params.BarConvTol = 10**-3

    # parameters of the Fenchel master model
    outter_model.Params.OutputFlag = 0
    # outter_model.Params.Method = -1
    # outter_model.Params.Crossover = 0
    # outter_model.Params.BarConvTol = 10**-3

    # main loop of the algorithm
    for iter_index in range(nb_iterations):
        if verbose : print("iteration : ", iter_index)
        # print("aa", bounds_and_time_list)

        # resolution of the two master models
        inner_model.update()
        inner_model.optimize()
        outter_model.update()
        outter_model.optimize()

        if verbose : print("Objective function values : ", inner_model.ObjVal, outter_model.ObjVal)
        if verbose : print("Runtimes : ", inner_model.Runtime, outter_model.Runtime)
        bounds_and_time_list.append((inner_model.ObjVal, outter_model.ObjVal, time.time() - starting_time))

        # the method stops if the bounds are close enough
        if abs(inner_model.ObjVal - outter_model.ObjVal) < 10**-3:
            break

        # variable deletion in the Dantzig-Wolfe model to prevent it from becoming to heavy
        for arc in arc_list:
            l = []
            for pattern, var, pattern_cost in inner_pattern_var_and_cost_per_arc[arc]:
                # if var.X < 10**-2 and random.random() < 0.:
                if var.Vbasis != 0 and random.random() < 0.3:
                    inner_model.remove(var)
                else:
                    l.append((pattern, var, pattern_cost))
            inner_pattern_var_and_cost_per_arc[arc] = l

        # subproblem resolution + adding variables and constraints to the two master problems
        nb_separated_arc = primal_dual_knapsack_separation(graph, demand_list, outter_model, outter_overload_vars, outter_flow_var_dict,
                                            inner_model, inner_flow_var_dict, inner_pattern_var_and_cost_per_arc, inner_constraints, separation_options, verbose=verbose)

        if verbose : print("nb_separated_arc = ", nb_separated_arc)


    inner_model.update()
    inner_model.optimize()

    outter_model.update()
    outter_model.optimize()


def primal_dual_knapsack_separation(graph, demand_list, outter_model, outter_overload_vars, outter_flow_var_dict,
                                    inner_model, inner_flow_var_dict, inner_pattern_var_and_cost_per_arc, inner_constraints, separation_options, verbose=1):
    #  this method calls the algorithms solving a Fenchel like separation subproblem
    # the cuts and variables (here pattern variables) created are added to the two master problems
    arc_list = [(node, neighbor) for node in range(len(graph)) for neighbor in graph[node]]
    nb_commodities = len(demand_list)

    convexity_constraint_dict, knapsack_convexity_constraint_dict, capacity_constraint_dict = inner_constraints

    nb_separated_arc = 0
    total_true_overload = 0

    t = [0]*5

    #  a subproblem is solved for each arc
    for arc in arc_list:
        temp = time.time()
        arc_capacity = graph[arc[0]][arc[1]]
        # print(arc)

        outter_flow_vars = outter_flow_var_dict[arc]
        outter_flow_per_commodity = np.array([0 if vars is 0 else vars.getValue() for vars in outter_flow_vars])
        outter_overload_value = outter_overload_vars[arc].X

        inner_flow_vars = inner_flow_var_dict[arc]
        inner_flow_per_commodity = np.array([0 if vars is 0 else vars.getValue() for vars in inner_flow_vars])
        inner_overload_value = sum(var.X * pattern_cost for pattern, var, pattern_cost in inner_pattern_var_and_cost_per_arc[arc])

        t[0] += time.time() - temp
        temp = time.time()

        # call to the separation problem for one arc : a cut and some pattern are returned
        in_out_separation, preprocessing, iterative_separation = separation_options # in_out_separation decides whether a directionnal normalisation is used or not
        if in_out_separation:
            if preprocessing:
                constraint_coeff, pattern_cost_and_amount_list = in_out_separation_decomposition_with_preprocessing(demand_list, outter_flow_per_commodity, outter_overload_value, inner_flow_per_commodity, inner_overload_value, arc_capacity, iterative_separation=iterative_separation)
            else:
                constraint_coeff, pattern_cost_and_amount_list = in_out_separation_decomposition(demand_list, outter_flow_per_commodity, outter_overload_value, inner_flow_per_commodity, inner_overload_value, arc_capacity)
        else:
            if preprocessing:
                constraint_coeff, pattern_cost_and_amount_list = separation_decomposition_with_preprocessing(demand_list, outter_flow_per_commodity, arc_capacity)
            else:
                constraint_coeff, pattern_cost_and_amount_list = separation_decomposition(demand_list, outter_flow_per_commodity, arc_capacity)

        commodity_coeff_list, overload_coeff, constant_coeff = constraint_coeff
        total_true_overload += sum(pattern_cost * amount for pattern, pattern_cost, amount in pattern_cost_and_amount_list)

        t[1] += time.time() - temp
        temp = time.time()

        # if the created cut cuts the solution of the Fenchel master problem it is added to the Fenchel master problem
        if sum(outter_flow_per_commodity * commodity_coeff_list) > constant_coeff + 10**-7 + overload_coeff * outter_overload_value:
            outter_model.addConstr((sum(outter_flow_var * coefficient for outter_flow_var, coefficient in zip(outter_flow_vars, commodity_coeff_list)) - overload_coeff * outter_overload_vars[arc] <= constant_coeff))
            nb_separated_arc += 1

        # the created patterns are added to the Dantzig-Wolfe master problem
        for pattern, pattern_cost, amount in pattern_cost_and_amount_list:
            column = gurobipy.Column()
            column.addTerms(1, knapsack_convexity_constraint_dict[arc])

            for commodity_index in pattern:
                if capacity_constraint_dict[arc][commodity_index] is not None:
                    column.addTerms(-1, capacity_constraint_dict[arc][commodity_index])

            new_var = inner_model.addVar(obj=pattern_cost, column=column)
            inner_pattern_var_and_cost_per_arc[arc].append((pattern, new_var, pattern_cost))

        t[2] += time.time() - temp

    if verbose : print(t)
    global ttt
    if verbose : print(ttt)
    ttt *= 0

    return nb_separated_arc


def remove_var_from_model(model, path_and_var_per_commodity, path_and_var, path, var, commodity_index):

    for index, path_var in enumerate(path_and_var_per_commodity[commodity_index]):
        if var is path_var[1]:
            path_and_var_per_commodity[commodity_index].pop(index)
            break

    for node_index in range(len(path) - 1):
        node, neighbor = path[node_index], path[node_index + 1]

        for index, path_var in enumerate(path_and_var[node, neighbor][commodity_index]):
            if var is path_var[1]:
                path_and_var[node, neighbor][commodity_index].pop(index)
                break

        if path_and_var[node, neighbor][commodity_index] == []:
            path_and_var[node, neighbor].pop(commodity_index)

    model.remove(var)


def compute_possible_paths_per_commodity(graph, commodity_list, nb_initial_path_created):
    # creates a list of allowed paths for each commodity which contains the k-shortest paths for this commodity
    shortest_paths_per_origin = {}
    possible_paths_per_commodity = []

    for commodity_index, commodity in enumerate(commodity_list):
        origin, destination, demand = commodity

        if origin not in shortest_paths_per_origin:
            shortest_paths_per_origin[origin] = k_shortest_path_all_destination(graph, origin, nb_initial_path_created)

        path_and_cost_list = shortest_paths_per_origin[origin][destination]
        possible_paths_per_commodity.append(set(tuple(remove_cycle_from_path(path)) for path, path_cost in path_and_cost_list))

        possible_paths_per_commodity[commodity_index] = [list(path_tuple) for path_tuple in possible_paths_per_commodity[commodity_index]]

    return possible_paths_per_commodity


def remove_cycle_from_path(path):
    is_in_path = set()
    new_path = []

    for node in path:
        if node in is_in_path:
            while new_path[-1] != node:
                node_to_remove = new_path.pop()
                is_in_path.remove(node_to_remove)

        else:
            is_in_path.add(node)
            new_path.append(node)

    return new_path


if __name__ == "__main__":
    pass
