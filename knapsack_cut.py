import random
import numpy as np
import time
import gurobipy
# from numba import jit

import sys
# sys.path = ["/home/francois/Desktop/knapsacksolver/bazel-bin/python"] + sys.path
import knapsacksolver # the code must be able to import knapsacksolver.so which is the result of the compilation of the library made by fontanf : https://github.com/fontanf/knapsacksolver.git, place knapsacksolver.so in the main folder



def compute_all_lifted_coefficients(demand_list, variable_pattern, coeff_list, fixed_pattern, RHS, remaining_arc_capacity, approximation=False):
    # this function take a cut valid for a polyhedron on a lower dimension and "lifts" it to become a cut valid for a polyhedron on a higher dimension
     # for more details, see the concept of cut lifting (can be found on the literature on knapsack or Fencel cuts)
    lifted_demand_list = [demand_list[commodity_index] for commodity_index in variable_pattern]
    lifted_commodity_list = list(variable_pattern)
    commodity_to_lift_list = list(fixed_pattern)
    coeff_list = list(coeff_list)
    new_pattern_and_cost_list = []

    while commodity_to_lift_list:
        commodity_index = commodity_to_lift_list.pop(0)
        remaining_arc_capacity += demand_list[commodity_index]

        if approximation:
            pre_pattern, lifted_coeff_part = relaxed_penalized_knapsack_optimizer(lifted_demand_list, remaining_arc_capacity, coeff_list)
        else:
            pre_pattern, lifted_coeff_part = penalized_knapsack_optimizer(lifted_demand_list, remaining_arc_capacity, coeff_list)
            # pre_pattern, lifted_coeff_part = gurobi_penalized_knapsack_optimizer(lifted_demand_list, remaining_arc_capacity, coeff_list)

        pattern = [lifted_commodity_list[index] for index in pre_pattern] + commodity_to_lift_list
        pattern_cost = max(0, sum(demand_list[commodity_index] for commodity_index in pattern) - remaining_arc_capacity)
        new_pattern_and_cost_list.append((pattern, pattern_cost))

        RHS, lifted_coeff = lifted_coeff_part, lifted_coeff_part - RHS

        lifted_demand_list.append(demand_list[commodity_index])
        lifted_commodity_list.append(commodity_index)
        coeff_list.append(lifted_coeff)
        # print(lifted_coeff, lifted_coeff_part, RHS)

    commodity_all_coeffs = np.zeros(len(demand_list))
    for index, commodity_index in enumerate(variable_pattern + fixed_pattern):
        commodity_all_coeffs[commodity_index] = coeff_list[index]

    return commodity_all_coeffs, RHS, new_pattern_and_cost_list


def relaxed_penalized_knapsack_optimizer(demand_list, arc_capacity, objective_coeff_per_commodity, overload_objective_coeff=1):
    nb_commodities = len(demand_list)
    order_list = [(objective_coeff_per_commodity[commodity_index] / demand_list[commodity_index], commodity_index) for commodity_index in range(nb_commodities)]
    order_list.sort()
    remaining_arc_capacity = arc_capacity
    lifted_coeff_part = 0
    pattern = []

    while order_list != []:
        ratio, commodity_index = order_list.pop()

        if ratio <= 0:
            break

        elif demand_list[commodity_index] <= remaining_arc_capacity or ratio > overload_objective_coeff:
            lifted_coeff_part += objective_coeff_per_commodity[commodity_index]
            remaining_arc_capacity = max(0, remaining_arc_capacity - demand_list[commodity_index])
            pattern.append(commodity_index)

        else:
            lifted_coeff_part += objective_coeff_per_commodity[commodity_index] * remaining_arc_capacity / demand_list[commodity_index]
            break

    return pattern, lifted_coeff_part


def approximate_penalized_knapsack_optimizer(demand_list, arc_capacity, objective_coeff_per_commodity, overload_objective_coeff=1):
    nb_commodities = len(demand_list)
    order_list = [(objective_coeff_per_commodity[commodity_index] / demand_list[commodity_index], commodity_index) for commodity_index in range(nb_commodities)]
    order_list.sort()
    remaining_arc_capacity = max(0, arc_capacity)
    value = min(0, arc_capacity)
    pattern = []

    while order_list != []:
        ratio, commodity_index = order_list[-1]

        if ratio <= 0:
            break

        elif demand_list[commodity_index] <= remaining_arc_capacity:
            value += objective_coeff_per_commodity[commodity_index]
            remaining_arc_capacity -= demand_list[commodity_index]
            pattern.append(commodity_index)
            order_list.pop()

        else:
            break

    gained_value_list = order_list

    while gained_value_list != []:

        l = []
        for _, commodity_index in gained_value_list:
            objective_coeff, demand = objective_coeff_per_commodity[commodity_index], demand_list[commodity_index]
            l.append((objective_coeff - overload_objective_coeff * max(0, demand - remaining_arc_capacity), commodity_index))
        gained_value_list = l

        gained_value, commodity_index = max(gained_value_list)

        if gained_value <= 0:
            break

        else:
            value += gained_value
            remaining_arc_capacity = max(0, remaining_arc_capacity - demand_list[commodity_index])
            pattern.append(commodity_index)
            gained_value_list.remove((gained_value, commodity_index))

    return pattern, value


# @jit(nopython=True)
def compute_all_lifted_coefficients2(demand_list, variable_pattern, variable_demand_list, coeff_list, fixed_pattern, RHS, remaining_arc_capacity):
    fixed_demand_list = np.array([demand_list[commodity_index] for commodity_index in fixed_pattern])
    fixed_demand = np.sum(fixed_demand_list)
    value_matrix = -np.ones((len(variable_pattern) + len(fixed_pattern) + 1, remaining_arc_capacity + fixed_demand + 1))
    lifted_demand_list = list(variable_demand_list)

    for index, commodity_index in enumerate(fixed_pattern):
        remaining_arc_capacity += demand_list[commodity_index]

        lifted_coeff_part = compute_lifted_coeff_part2(value_matrix, coeff_list, lifted_demand_list, remaining_arc_capacity, len(coeff_list))

        RHS, lifted_coeff = lifted_coeff_part, lifted_coeff_part - RHS
        lifted_demand_list.append(demand_list[commodity_index])
        coeff_list.append(lifted_coeff)

    return coeff_list, RHS


# @jit(nopython=True)
def compute_lifted_coeff_part2(value_matrix, coeff_list, demand_list, remaining_arc_capacity, nb_commodities):
    if nb_commodities == 0:
        return 0

    if remaining_arc_capacity == 0:
        return np.sum(np.array([max(0, coeff_list[commodity_index] - demand_list[commodity_index]) for commodity_index in range(nb_commodities)]))

    last_demand = demand_list[nb_commodities-1]
    new_remaining_arc_capacity = max(0, remaining_arc_capacity - last_demand)
    overflow = max(0, last_demand - remaining_arc_capacity)

    if value_matrix[nb_commodities-1, remaining_arc_capacity] == -1:
        value_0 = compute_lifted_coeff_part2(value_matrix, coeff_list, demand_list, remaining_arc_capacity, nb_commodities-1)
        value_matrix[nb_commodities-1, remaining_arc_capacity] = value_0
    else:
        value_0 = value_matrix[nb_commodities-1, remaining_arc_capacity]

    if value_matrix[nb_commodities-1, new_remaining_arc_capacity] == -1:
        value_1 = compute_lifted_coeff_part2(value_matrix, coeff_list, demand_list, new_remaining_arc_capacity, nb_commodities-1)
        value_matrix[nb_commodities-1, new_remaining_arc_capacity] = value_1
    else:
        value_1 = value_matrix[nb_commodities-1, new_remaining_arc_capacity]

    lifted_coeff_part = max(value_0, value_1 + coeff_list[nb_commodities-1] - overflow)
    value_matrix[nb_commodities, remaining_arc_capacity] = lifted_coeff_part

    return lifted_coeff_part


def compute_approximate_decomposition(demand_list, flow_per_commodity, arc_capacity, order_of_commodities="sorted"):
    # compute a decomposition of a flow distribution on an arc as a convex combination of commodity patterns. This decompostion is approximately optimal in the sense of pattern costs
    nb_commodities = len(demand_list)
    cost_pattern_and_amount_list = [[max(0, sum(demand_list) - arc_capacity), list(range(nb_commodities)), 1]]

    commodity_order = list(range(nb_commodities))
    if order_of_commodities == "sorted":
        commodity_order.sort(key=lambda x:demand_list[x], reverse=True)
    elif order_of_commodities == "random":
        random.shuffle(commodity_order)

    for commodity_index in commodity_order:
        current_flow = 1
        while current_flow > flow_per_commodity[commodity_index] + 10**-5:
            # print(current_flow, flow_per_commodity[commodity_index])
            cost_pattern_and_amount = max([x for x in cost_pattern_and_amount_list if commodity_index in x[1]])
            pattern_cost, pattern, amount = cost_pattern_and_amount
            new_pattern = list(pattern)
            new_pattern.remove(commodity_index)
            new_pattern_cost = max(0, pattern_cost - demand_list[commodity_index])
            new_amount = min(amount, current_flow - flow_per_commodity[commodity_index])
            cost_pattern_and_amount_list.append([new_pattern_cost, new_pattern, new_amount])
            current_flow -= new_amount

            if new_amount == amount:
                cost_pattern_and_amount_list.remove(cost_pattern_and_amount)
            else:
                cost_pattern_and_amount[2] -= new_amount

    pattern_cost_and_amount_list = [(pattern, pattern_cost, amount) for pattern_cost, pattern, amount in cost_pattern_and_amount_list]
    return pattern_cost_and_amount_list


def compute_approximate_decomposition2(demand_list, flow_per_commodity, arc_capacity):
    # compute a decomposition of a flow distribution on an arc as a convex combination of commodity patterns. This decompostion is approximately optimal in the sense of pattern costs
    nb_commodities = len(demand_list)
    cost_pattern_and_amount_list = [[max(0, sum(demand_list) - arc_capacity), list(range(nb_commodities)), 1]]
    done = [False] * nb_commodities
    current_flow_per_commodity = [1] * nb_commodities
    final_pattern_list = []

    while sum(done) != nb_commodities:

        while True:
            a = max(cost_pattern_and_amount_list)
            pattern_cost, pattern, amount = a
            l = [(abs(pattern_cost - demand_list[commodity_index]), commodity_index) for commodity_index in pattern if not done[commodity_index]]
            if l == []:
                final_pattern_list.append((pattern, pattern_cost, amount))
                cost_pattern_and_amount_list.remove(a)
            else:
                break

        _, chosen_commodity_index = min(l)
        new_pattern = list(pattern)
        new_pattern.remove(chosen_commodity_index)
        new_pattern_cost = max(0, pattern_cost - demand_list[chosen_commodity_index])
        new_amount = min(amount, current_flow_per_commodity[chosen_commodity_index] - flow_per_commodity[chosen_commodity_index])
        cost_pattern_and_amount_list.append([new_pattern_cost, new_pattern, new_amount])
        current_flow_per_commodity[chosen_commodity_index] -= new_amount

        if new_amount == amount:
            cost_pattern_and_amount_list.remove(a)
        else:
            a[2] -= new_amount
            done[chosen_commodity_index] = True

    pattern_cost_and_amount_list = [(pattern, pattern_cost, amount) for pattern_cost, pattern, amount in cost_pattern_and_amount_list] + final_pattern_list

    return pattern_cost_and_amount_list



def separation_decomposition_norme_2(demand_list, flow_per_commodity, overload_value, arc_capacity, verbose=1):
    nb_commodities = len(demand_list)
    # print(nb_commodities, flow_per_commodity)

    # Create optimization model
    model = gurobipy.Model()
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = 0
    # model.Params.NonConvex = 2

    pattern_cost_and_amount_list = compute_approximate_decomposition(demand_list, flow_per_commodity, arc_capacity)

    # Create variables
    pattern_cost_and_var_list = [(pattern, pattern_cost, model.addVar()) for pattern, pattern_cost, amount in pattern_cost_and_amount_list] # pattern choice variables
    pattern_cost_and_var_list.extend([(pattern, 10**5, model.addVar()) for pattern, pattern_cost, amount in pattern_cost_and_amount_list])

    convexity_constraint = model.addConstr(sum(var for pattern, pattern_cost, var in pattern_cost_and_var_list) == 1)

    flow_var_per_commodity = [None]*nb_commodities
    overload_var = sum(var * pattern_cost for pattern, pattern_cost, var in pattern_cost_and_var_list)
    objective_function = (overload_var - overload_value) * (overload_var - overload_value)
    for commodity_index in range(nb_commodities):
        flow_var_per_commodity[commodity_index] = gurobipy.LinExpr(sum(var for pattern, pattern_cost, var in pattern_cost_and_var_list if commodity_index in pattern))
        objective_function += (flow_var_per_commodity[commodity_index] - flow_per_commodity[commodity_index]) * (flow_var_per_commodity[commodity_index] - flow_per_commodity[commodity_index])

    model.setObjective(objective_function)

    print("aaaaaaaa")
    print(flow_per_commodity)
    print(overload_value)
    print([pattern_cost for pattern, pattern_cost, var in pattern_cost_and_var_list])

    i = 0
    use_heuristic = False
    while True:
        model.update()
        model.optimize()
        i+=1

        if verbose:
            print(i, model.ObjVal, end='        \r')

        commodity_dual_value_list =  2 * np.array([(flow_var_per_commodity[commodity_index].getValue() - flow_per_commodity[commodity_index]) for commodity_index in range(nb_commodities)])
        overload_dual_value = 2 * (overload_var.getValue() - overload_value)
        assert overload_dual_value >= -10**-10
        overload_dual_value = max(0, overload_dual_value)
        convexity_dual_value = convexity_constraint.Pi
        # print("##########")
        # print(flow_per_commodity)
        # print([flow_var.getValue() for flow_var in flow_var_per_commodity])
        # print(commodity_dual_value_list)
        # print(overload_value, overload_var.getValue(), overload_dual_value, convexity_dual_value)


        if sum(demand for demand, dual_value in zip(demand_list, commodity_dual_value_list) if dual_value != 0) <= arc_capacity:
            pattern = [commodity_index for commodity_index, dual_value in enumerate(commodity_dual_value_list) if dual_value != 0]
            subproblem_objective_value = -sum(commodity_dual_value_list)

        elif use_heuristic:
            pattern, subproblem_objective_value = approximate_penalized_knapsack_optimizer(demand_list, arc_capacity, -commodity_dual_value_list, overload_objective_coeff=overload_dual_value)

        else:
            pattern, subproblem_objective_value = penalized_knapsack_optimizer(demand_list, arc_capacity, -commodity_dual_value_list, overload_objective_coeff=overload_dual_value)

        reduced_cost = -subproblem_objective_value - convexity_dual_value
        pattern_cost = max(0, sum(demand_list[commodity_index] for commodity_index in pattern) - arc_capacity)
        # print(pattern, pattern_cost)
        # print(reduced_cost, subproblem_objective_value)

        if reduced_cost < -10**-5:
            use_heuristic = True
            column = gurobipy.Column()
            column.addTerms(1, convexity_constraint)

            new_var = model.addVar(column=column)
            pattern_cost_and_var_list.append((pattern, pattern_cost, new_var))

            overload_var += new_var * pattern_cost
            objective_function = (overload_var - overload_value) * (overload_var - overload_value)
            for commodity_index in range(nb_commodities):
                if commodity_index in pattern:
                    flow_var_per_commodity[commodity_index] += new_var
                objective_function += (flow_var_per_commodity[commodity_index] - flow_per_commodity[commodity_index]) * (flow_var_per_commodity[commodity_index] - flow_per_commodity[commodity_index])
            model.setObjective(objective_function)

        else:
            if use_heuristic:
                use_heuristic = False
            else:
                break

    return (-commodity_dual_value_list, overload_dual_value, -convexity_dual_value), [(pattern, pattern_cost, var.X) for pattern, pattern_cost, var in pattern_cost_and_var_list if var.X > 10**-6]


def separation_decomposition_aggregation(demand_list, flow_per_commodity, arc_capacity, verbose=0):
    nb_commodities = len(demand_list)
    max_commodity_subset_size = 3

    # Create optimization model
    model = gurobipy.Model()
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = 0

    pattern_cost_and_amount_list = compute_approximate_decomposition(demand_list, flow_per_commodity, arc_capacity)

    # Create variables
    pattern_cost_and_var_list = [(pattern, pattern_cost, model.addVar(obj=pattern_cost)) for pattern, pattern_cost, amount in pattern_cost_and_amount_list] # pattern choice variables

    convexity_constraint = model.addConstr(sum(var for pattern, pattern_cost, var in pattern_cost_and_var_list) == 1)

    # knapsack_constraint_dict = {}
    # flow_var = gurobipy.LinExpr(sum(var * len(pattern) for pattern, pattern_cost, var in pattern_cost_and_var_list))
    # knapsack_constraint_dict[tuple(range(nb_commodities))] = model.addConstr(-flow_var <= -sum(flow_per_commodity))

    flow_var_list = [gurobipy.LinExpr(0) for commodity_index in range(nb_commodities)]
    for pattern, pattern_cost, var in pattern_cost_and_var_list:
        for commodity_index in pattern:
            flow_var_list[commodity_index] += var

    knapsack_constraint_dict = {}
    commodity_subset = []
    for commodity_index in range(nb_commodities):
        commodity_subset.append(commodity_index)

        if len(commodity_subset) == max_commodity_subset_size or commodity_index == nb_commodities-1:
            LHS = sum(flow_var_list[commodity_index] for commodity_index in commodity_subset)
            RHS = sum(flow_per_commodity[commodity_index] for commodity_index in commodity_subset)
            knapsack_constraint_dict[tuple(commodity_subset)] = model.addConstr(-LHS <= -RHS)
            commodity_subset = []

    i = 0
    while True:
        model.update()
        model.optimize()
        i+=1

        convexity_dual_value = convexity_constraint.Pi
        commodity_dual_value_list = np.zeros(nb_commodities)
        for commodity_subset in knapsack_constraint_dict:
            for commodity_index in commodity_subset:
                commodity_dual_value_list[commodity_index] += knapsack_constraint_dict[commodity_subset].Pi

        if sum(demand for demand, dual_value in zip(demand_list, commodity_dual_value_list) if dual_value != 0) <= arc_capacity:
            pattern = [commodity_index for commodity_index, dual_value in enumerate(commodity_dual_value_list) if dual_value != 0]
            subproblem_objective_value = -sum(commodity_dual_value_list)

        else:
            pattern, subproblem_objective_value = penalized_knapsack_optimizer(demand_list, arc_capacity, -commodity_dual_value_list)

        reduced_cost = -subproblem_objective_value - convexity_dual_value
        pattern_cost = max(0, sum(demand_list[commodity_index] for commodity_index in pattern) - arc_capacity)

        if reduced_cost < -10**-5:
            column = gurobipy.Column()
            column.addTerms(1, convexity_constraint)

            for commodity_index in pattern:
                for commodity_subset in knapsack_constraint_dict:
                    if commodity_index in commodity_subset:
                        column.addTerms(-1, knapsack_constraint_dict[commodity_subset])

            new_var = model.addVar(obj=pattern_cost, column=column)
            pattern_cost_and_var_list.append((pattern, pattern_cost, new_var))
            commodity_subset = "####"

        else:
            flow_var_list = [gurobipy.LinExpr(0) for commodity_index in range(nb_commodities)]
            for pattern, pattern_cost, var in pattern_cost_and_var_list:
                for commodity_index in pattern:
                    flow_var_list[commodity_index] += var

            commodity_subset = []
            added_constraint = False
            for commodity_index in range(nb_commodities):
                if flow_var_list[commodity_index].getValue() < flow_per_commodity[commodity_index] - 10**-5:
                    commodity_subset.append(commodity_index)

                    if len(commodity_subset) == max_commodity_subset_size:
                        LHS = sum(flow_var_list[commodity_index] for commodity_index in commodity_subset)
                        RHS = sum(flow_per_commodity[commodity_index] for commodity_index in commodity_subset)
                        knapsack_constraint_dict[tuple(commodity_subset)] = model.addConstr(-LHS <= -RHS)
                        added_constraint = True
                        commodity_subset = []

            if len(commodity_subset) > 0:
                LHS = sum(flow_var_list[commodity_index] for commodity_index in commodity_subset)
                RHS = sum(flow_per_commodity[commodity_index] for commodity_index in commodity_subset)
                knapsack_constraint_dict[tuple(commodity_subset)] = model.addConstr(-LHS <= -RHS)
                added_constraint = True

            if not added_constraint:
                break

        if verbose:
            print(i, model.ObjVal, len(demand_list), reduced_cost, commodity_subset, pattern, end='        \r')
            if i > 600:
                model.update()
                model.optimize()
                print()
                print(demand_list, arc_capacity)
                print(flow_per_commodity)
                print(commodity_dual_value_list)
                print(convexity_dual_value)
                print(pattern)
                print(reduced_cost)
                print([(pattern, pattern_cost, var.X) for pattern, pattern_cost, var in pattern_cost_and_var_list])
                print({commodity_subset : knapsack_constraint_dict[commodity_subset].Pi for commodity_subset in knapsack_constraint_dict})
                assert False

    return (-commodity_dual_value_list, 1, -convexity_dual_value), [(pattern, pattern_cost, var.X) for pattern, pattern_cost, var in pattern_cost_and_var_list if var.Vbasis == 0]



def separation_decomposition(demand_list, flow_per_commodity, arc_capacity, initial_pattern_and_cost_list=None, verbose=0):
    # compute a decomposition of a flow distribution on an arc as a convex combination of commodity patterns. This decompostion is optimal in the sense of pattern costs
    # the last optimal dual variables represent the coefficients of a cut
    # it uses a colum generation process (see the subproblem of the Fenchel decomposotion for more details)
    nb_commodities = len(demand_list)

    # Create optimization model
    model = gurobipy.Model()
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = 0

    # starts with an approximately optimal decomposition
    pattern_cost_and_amount_list = compute_approximate_decomposition(demand_list, flow_per_commodity, arc_capacity)

    if initial_pattern_and_cost_list is not None:
        pattern_cost_and_amount_list.extend([(pattern, pattern_cost, 0) for pattern, pattern_cost in initial_pattern_and_cost_list])

    # Create pattern variables
    pattern_cost_and_var_list = [(pattern, pattern_cost, model.addVar(obj=pattern_cost)) for pattern, pattern_cost, amount in pattern_cost_and_amount_list] # pattern choice variables

    convexity_constraint = model.addConstr(sum(var for pattern, pattern_cost, var in pattern_cost_and_var_list) == 1)

    knapsack_constraint_dict = {}
    for commodity_index in range(nb_commodities):
        flow_var = gurobipy.LinExpr(sum(var for pattern, pattern_cost, var in pattern_cost_and_var_list if commodity_index in pattern))
        knapsack_constraint_dict[commodity_index] = model.addConstr((-flow_var <= -flow_per_commodity[commodity_index]))

    # main loop
    i = 0
    use_heuristic = True
    while True:
        model.update()
        model.optimize()
        i+=1

        # extracting dual values from the model
        commodity_dual_value_list = np.array([knapsack_constraint_dict[commodity_index].Pi for commodity_index in range(nb_commodities)])
        convexity_dual_value = convexity_constraint.Pi

        # resolution of the subproblem of the column generation process
        if sum(demand for demand, dual_value in zip(demand_list, commodity_dual_value_list) if dual_value != 0) <= arc_capacity:
            pattern = [commodity_index for commodity_index, dual_value in enumerate(commodity_dual_value_list) if dual_value != 0]
            subproblem_objective_value = -sum(commodity_dual_value_list)

        elif use_heuristic:
            pattern, subproblem_objective_value = approximate_penalized_knapsack_optimizer(demand_list, arc_capacity, -commodity_dual_value_list)

        else:
            pattern, subproblem_objective_value = penalized_knapsack_optimizer(demand_list, arc_capacity, -commodity_dual_value_list)
            # pattern2, subproblem_objective_value2 = gurobi_penalized_knapsack_optimizer(demand_list, arc_capacity, -commodity_dual_value_list)
            # assert abs(subproblem_objective_value - subproblem_objective_value2) < 10**-3, (pattern, pattern2, demand_list, arc_capacity, commodity_dual_value_list, subproblem_objective_value, subproblem_objective_value2)

        reduced_cost = -subproblem_objective_value - convexity_dual_value
        pattern_cost = max(0, sum(demand_list[commodity_index] for commodity_index in pattern) - arc_capacity)

        if verbose:
            print(i, model.ObjVal, len(demand_list), convexity_dual_value, end='        \r')
            if i > 600 and use_heuristic == False:
                model.update()
                model.optimize()
                print()
                print(demand_list, arc_capacity)
                print(flow_per_commodity)
                print(commodity_dual_value_list)
                print(convexity_dual_value)
                print(pattern)
                print(reduced_cost)
                print(subproblem_objective_value)
                print([(pattern, pattern_cost, var.X) for pattern, pattern_cost, var in pattern_cost_and_var_list if var.VBasis == 0])
                print([(pattern, pattern_cost, var.X) for pattern, pattern_cost, var in pattern_cost_and_var_list])
                assert False

        # if a pattern with a negative reduced cost has been computed, it is added to the model
        if reduced_cost < -10**-4:
            use_heuristic = True
            column = gurobipy.Column()
            column.addTerms(1, convexity_constraint)

            for commodity_index in pattern:
                column.addTerms(-1, knapsack_constraint_dict[commodity_index])

            new_var = model.addVar(obj=pattern_cost, column=column)
            pattern_cost_and_var_list.append((pattern, pattern_cost, new_var))

        else:
            if use_heuristic:
                use_heuristic = False
            elif model.Params.Method == 2:
                model.Params.Method = -1
            else:
                break

    return (-commodity_dual_value_list, 1, -convexity_dual_value), [(pattern, pattern_cost, var.X) for pattern, pattern_cost, var in pattern_cost_and_var_list if var.Vbasis == 0]


def separation_decomposition_with_preprocessing(demand_list, flow_per_commodity, arc_capacity, initial_pattern_and_cost_list=None, verbose=0):
    # makes some preprocessing then calls the method that will make the decomposition and compute the cut
    # afterwards the coefficient of the cuts are lifted
    nb_commodities = len(demand_list)

    fixed_pattern = [commodity_index for commodity_index, flow_value in enumerate(flow_per_commodity) if flow_value == 1]
    variable_pattern = [commodity_index for commodity_index, flow_value in enumerate(flow_per_commodity) if flow_value != 1 and flow_value != 0]

    variable_demand_list = [demand_list[commodity_index] for commodity_index in variable_pattern]
    remaining_arc_capacity = arc_capacity - sum(demand_list[commodity_index] for commodity_index in fixed_pattern)
    variable_flow_per_commodity = [flow_per_commodity[commodity_index] for commodity_index in variable_pattern]

    variable_initial_pattern_and_cost_list = []
    if initial_pattern_and_cost_list is not None:
        for pattern, pattern_cost in initial_pattern_and_cost_list:
            partial_pattern = []
            for commodity_index in pattern:
                if commodity_index in variable_pattern:
                    partial_pattern.append(variable_pattern.index(commodity_index))
                else:
                    break
            else:
                variable_initial_pattern_and_cost_list.append((partial_pattern, max(0, sum(variable_demand_list[commodity_index] for commodity_index in partial_pattern) - remaining_arc_capacity)))

    # calling the separation/decomposition method
    if len(variable_flow_per_commodity) == 0:
        constraint_coeff, pre_pattern_cost_and_amount_list = separation_decomposition(variable_demand_list, variable_flow_per_commodity, remaining_arc_capacity, verbose=verbose)
    else:
        # constraint_coeff, pre_pattern_cost_and_amount_list = separation_decomposition_aggregation(variable_demand_list, variable_flow_per_commodity, remaining_arc_capacity, verbose=verbose)
        constraint_coeff, pre_pattern_cost_and_amount_list = separation_decomposition(variable_demand_list, variable_flow_per_commodity, remaining_arc_capacity, initial_pattern_and_cost_list=variable_initial_pattern_and_cost_list, verbose=verbose)

    variable_commodity_coeff_list, overload_coeff, constant_coeff = constraint_coeff
    pattern_cost_and_amount_list = [([variable_pattern[index] for index in pattern] + fixed_pattern, pattern_cost, amount) for pattern, pattern_cost, amount in pre_pattern_cost_and_amount_list]

    if overload_coeff == 0:
        return (np.zeros(nb_commodities), 0, 0), pattern_cost_and_amount_list

    # lifting the coefficients of the cut
    commodity_coeff_list, constant_coeff, lifting_pattern_and_cost_list = compute_all_lifted_coefficients(demand_list, variable_pattern, variable_commodity_coeff_list, fixed_pattern, constant_coeff, remaining_arc_capacity)

    for pattern, pattern_cost in lifting_pattern_and_cost_list:
        pattern_cost_and_amount_list.append((pattern, pattern_cost, 0))

    return (commodity_coeff_list, overload_coeff, constant_coeff), pattern_cost_and_amount_list


def in_out_separation_decomposition(demand_list, outter_flow_per_commodity, outter_overload_value, inner_flow_per_commodity, inner_overload_value, arc_capacity, initial_pattern_cost_and_amount_list=[], verbose=0):
    # compute a decomposition of a flow distribution on an arc as a convex combination of commodity patterns. This decompostion is optimal in the sense of a normalisation
    # the last optimal dual variables represent the coefficients of a cut
    # it uses a colum generation process (see the subproblem of the Fenchel decomposotion for more details)
    nb_commodities = len(demand_list)
    outter_flow_per_commodity = np.maximum(0, outter_flow_per_commodity)

    # Create optimization model
    model = gurobipy.Model()
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = 0

    # Create variables
    initial_pattern_var = model.addVar(obj=0)
    pattern_var_and_cost_list = [([], initial_pattern_var, inner_overload_value)] # pattern choice variables
    # pattern_var_and_cost_list = [(pattern, model.addVar(), pattern_cost) for pattern, pattern_cost, amount in pattern_cost_and_amount_list] # pattern choice variables
    # pattern_var_and_cost_list.extend([(pattern, model.addVar(), pattern_cost + 10**5) for pattern, pattern_cost, amount in pattern_cost_and_amount_list]) # pattern choice variables
    penalisation_var_plus = model.addVar(obj=1) # positive part of the penalisation var
    penalisation_var_minus = model.addVar(obj=1) # negative part of the penalisation var
    penalisation_var = penalisation_var_plus - penalisation_var_minus

    convexity_constraint = model.addConstr(sum(var for pattern, var, pattern_cost in pattern_var_and_cost_list) == 1)
    overload_constraint = model.addConstr(sum(var * pattern_cost for pattern, var, pattern_cost in pattern_var_and_cost_list) - penalisation_var * (inner_overload_value - outter_overload_value) <= outter_overload_value)

    knapsack_constraint_dict = {}
    for commodity_index in range(nb_commodities):
        inner_flow, outter_flow = inner_flow_per_commodity[commodity_index], outter_flow_per_commodity[commodity_index]
        knapsack_constraint_dict[commodity_index] = model.addConstr(-initial_pattern_var * inner_flow + penalisation_var * (inner_flow - outter_flow) <= -outter_flow)

    # knapsack_constraint_dict = {}
    # for commodity_index in range(nb_commodities):
    #     inner_flow, outter_flow = inner_flow_per_commodity[commodity_index], outter_flow_per_commodity[commodity_index]
    #     flow_var = sum(var for pattern, var, pattern_cost in pattern_var_and_cost_list if commodity_index in pattern)
    #     knapsack_constraint_dict[commodity_index] = model.addConstr(-flow_var + penalisation_var * (inner_flow - outter_flow) <= -outter_flow)

    # main loop of the column generation process
    i = 0
    while True:
        i += 1
        model.update()
        model.optimize()
        if model.Status != 2 :
            print()
            print(i, ", status : ", model.Status)
            print(outter_overload_value, inner_overload_value, arc_capacity)
            print(demand_list)
            print(outter_flow_per_commodity)
            print(inner_flow_per_commodity)
            print(pattern_cost_and_amount_list)
            flow = np.zeros(nb_commodities)
            for pattern, pattern_cost, amount in pattern_cost_and_amount_list:
                for commodity_index in pattern:
                    flow[commodity_index] += amount
            print(flow)

            print(model.Params.FeasibilityTol)
            model.write("model.lp")
            model.computeIIS()
            model.write("modelIIS.ilp")

        # getting the dual variables
        commodity_dual_value_list = -np.array([knapsack_constraint_dict[commodity_index].Pi for commodity_index in range(nb_commodities)])
        overload_dual_value = -overload_constraint.Pi
        convexity_dual_value = -convexity_constraint.Pi

        # solving the subproblem of the column generation process
        pattern, subproblem_objective_value = penalized_knapsack_optimizer(demand_list, arc_capacity, commodity_dual_value_list, overload_dual_value)

        reduced_cost = -subproblem_objective_value + convexity_dual_value
        pattern_cost = max(0, sum(demand_list[commodity_index] for commodity_index in pattern) - arc_capacity)
        if verbose : print(i, model.ObjVal, reduced_cost, end='          \r')

        #  if the pattern with a negative reduced cost is computed it is added to the model
        if reduced_cost < -10**-5:
            column = gurobipy.Column()
            column.addTerms(1, convexity_constraint)
            column.addTerms(pattern_cost, overload_constraint)

            for commodity_index in pattern:
                column.addTerms(-1, knapsack_constraint_dict[commodity_index])

            new_var = model.addVar(obj=0, column=column)
            pattern_var_and_cost_list.append((pattern, new_var, pattern_cost))

        else:
            break

    # normalise the coefficients of the cut
    if overload_dual_value != 0:
        commodity_dual_value_list = commodity_dual_value_list / overload_dual_value
        convexity_dual_value = convexity_dual_value / overload_dual_value
        overload_dual_value = 1

    return (commodity_dual_value_list, overload_dual_value, convexity_dual_value), [(pattern, pattern_cost, var.X) for pattern, var, pattern_cost in pattern_var_and_cost_list[1:] if var.VBasis == 0]


def in_out_separation_decomposition_iterative(demand_list, outter_flow_per_commodity, outter_overload_value, inner_flow_per_commodity, inner_overload_value, arc_capacity, verbose=0):
    # compute a decomposition of a flow distribution on an arc as a convex combination of commodity patterns. This decompostion is optimal in the sense of a normalisation
    # the last optimal dual variables represent the coefficients of a cut (see the subproblem of the Fenchel decomposotion for more details)
    # this is a new method to solve the Fenchel subproblem when a directionnal normalisation is used
    # it uses another normalisation to make the computation
    nb_commodities = len(demand_list)
    in_out_convex_coeff = 0
    outter_flow_per_commodity = np.array(outter_flow_per_commodity)
    inner_flow_per_commodity = np.array(inner_flow_per_commodity)
    current_flow_per_commodity = outter_flow_per_commodity
    current_overload_value = outter_overload_value
    commodity_coeff_list = np.zeros(nb_commodities)
    constant_coeff = 0

    # Create optimization model
    model = gurobipy.Model()
    model.modelSense = gurobipy.GRB.MINIMIZE
    model.Params.OutputFlag = 0

    # Create variables
    pattern_cost_and_amount_list = compute_approximate_decomposition(demand_list, outter_flow_per_commodity, arc_capacity)
    pattern_cost_and_amount_list.extend(compute_approximate_decomposition(demand_list, inner_flow_per_commodity, arc_capacity))
    pattern_cost_and_var_list = [(pattern, pattern_cost, model.addVar(obj=pattern_cost)) for pattern, pattern_cost, amount in pattern_cost_and_amount_list] # pattern choice variables
    # pattern_cost_and_var_list.append((list(range(nb_commodities)), model.addVar(obj=10**5), 10**5))


    # Constraints :
    convexity_constraint = model.addConstr(sum(var for pattern, pattern_cost, var in pattern_cost_and_var_list) == 1)
    knapsack_constraint_dict = {}
    for commodity_index in range(nb_commodities):
        flow_var = gurobipy.LinExpr(sum(var for pattern, pattern_cost, var in pattern_cost_and_var_list if commodity_index in pattern))
        knapsack_constraint_dict[commodity_index] = model.addConstr((-flow_var <= -outter_flow_per_commodity[commodity_index]))

    # main loop
    i = 0
    use_heuristic = True
    while True:
        model.update()
        model.optimize()
        i+=1

        if current_overload_value > model.ObjVal - 10**-5:
            break

        commodity_dual_value_list = np.array([knapsack_constraint_dict[commodity_index].Pi for commodity_index in range(nb_commodities)])
        convexity_dual_value = convexity_constraint.Pi

        if sum(demand for demand, dual_value in zip(demand_list, commodity_dual_value_list) if dual_value != 0) <= arc_capacity:
            pattern = [commodity_index for commodity_index, dual_value in enumerate(commodity_dual_value_list) if dual_value != 0]
            subproblem_objective_value = -sum(commodity_dual_value_list)
        elif use_heuristic:
            pattern, subproblem_objective_value = approximate_penalized_knapsack_optimizer(demand_list, arc_capacity, -commodity_dual_value_list)

        else:
            pattern, subproblem_objective_value = penalized_knapsack_optimizer(demand_list, arc_capacity, -commodity_dual_value_list)

        reduced_cost = -subproblem_objective_value - convexity_dual_value
        pattern_cost = max(0, sum(demand_list[commodity_index] for commodity_index in pattern) - arc_capacity)
        if verbose :
            print(i, model.ObjVal, reduced_cost, current_overload_value, in_out_convex_coeff, end='         \r')


        if use_heuristic == True:
            if reduced_cost >= -10**-6:
                use_heuristic = False

        elif sum(-commodity_dual_value_list * current_flow_per_commodity) - current_overload_value > subproblem_objective_value:
            outter_value = sum(-commodity_dual_value_list * outter_flow_per_commodity) - outter_overload_value
            inner_value = sum(-commodity_dual_value_list * inner_flow_per_commodity) - inner_overload_value
            in_out_convex_coeff = (subproblem_objective_value - outter_value) / (inner_value - outter_value)
            current_flow_per_commodity = in_out_convex_coeff * inner_flow_per_commodity + (1 - in_out_convex_coeff) * outter_flow_per_commodity
            current_overload_value = in_out_convex_coeff * inner_overload_value + (1 - in_out_convex_coeff) * outter_overload_value
            commodity_coeff_list = -commodity_dual_value_list
            constant_coeff = -convexity_dual_value
            use_heuristic = True

            for commodity_index in range(nb_commodities):
                knapsack_constraint_dict[commodity_index].RHS = -current_flow_per_commodity[commodity_index]

        if reduced_cost < -10**-6:
            use_heuristic = True

        column = gurobipy.Column()
        column.addTerms(1, convexity_constraint)

        for commodity_index in pattern:
            column.addTerms(-1, knapsack_constraint_dict[commodity_index])

        new_var = model.addVar(obj=pattern_cost, column=column)
        pattern_cost_and_var_list.append((pattern, pattern_cost, new_var))

    return (commodity_coeff_list, 1, constant_coeff), [(pattern, pattern_cost, var.X) for pattern, pattern_cost, var in pattern_cost_and_var_list if var.Vbasis == 0]


def in_out_separation_decomposition_iterative2(demand_list, outter_flow_per_commodity, outter_overload_value, inner_flow_per_commodity, inner_overload_value, arc_capacity):
    # compute a decomposition of a flow distribution on an arc as a convex combination of commodity patterns. This decompostion is optimal in the sense of a normalisation
    # the last optimal dual variables represent the coefficients of a cut (see the subproblem of the Fenchel decomposotion for more details)
    # this is a new method to solve the Fenchel subproblem when a directionnal normalisation is used
    # it uses another normalisation to make the computation
    nb_commodities = len(demand_list)
    in_out_convex_coeff = 0
    outter_flow_per_commodity = np.array(outter_flow_per_commodity)
    inner_flow_per_commodity = np.array(inner_flow_per_commodity)
    current_flow_per_commodity = outter_flow_per_commodity
    current_overload_value = outter_overload_value
    old_constraint_coeff = (np.zeros(nb_commodities), 1, 0)
    old_pattern_and_cost_list = []
    inner_overload_value += 0.1

    _, inner_pattern_cost_and_amount_list = separation_decomposition_with_preprocessing(demand_list, inner_flow_per_commodity, arc_capacity, verbose=0)
    inner_pattern_and_cost_list = [(pattern, pattern_cost) for pattern, pattern_cost, amount in inner_pattern_cost_and_amount_list]

    i = 0
    use_heuristic = True
    while True:
        i+=1
        constraint_coeff, pattern_cost_and_amount_list = separation_decomposition_with_preprocessing(demand_list, current_flow_per_commodity, arc_capacity, initial_pattern_and_cost_list=inner_pattern_and_cost_list + old_pattern_and_cost_list, verbose=0)
        # constraint_coeff, pattern_cost_and_amount_list = separation_decomposition_norme_2(demand_list, current_flow_per_commodity, current_overload_value, arc_capacity, verbose=1)

        decomposition_overload = sum(pattern_cost * amount for pattern, pattern_cost, amount in pattern_cost_and_amount_list)
        if current_overload_value > decomposition_overload - 10**-5:
            break

        commodity_coeff_list, overload_coeff, constant_coeff = constraint_coeff
        old_pattern_and_cost_list = [(pattern, pattern_cost) for pattern, pattern_cost, amount in pattern_cost_and_amount_list]

        if sum(commodity_coeff_list * current_flow_per_commodity) > overload_coeff * current_overload_value + constant_coeff:
            outter_value = sum(commodity_coeff_list * outter_flow_per_commodity) - overload_coeff * outter_overload_value - constant_coeff
            inner_value = sum(commodity_coeff_list * inner_flow_per_commodity) - overload_coeff * inner_overload_value - constant_coeff
            in_out_convex_coeff = max(0, min(1, (- outter_value) / (inner_value - outter_value)))
            current_flow_per_commodity = in_out_convex_coeff * inner_flow_per_commodity + (1 - in_out_convex_coeff) * outter_flow_per_commodity
            current_overload_value = in_out_convex_coeff * inner_overload_value + (1 - in_out_convex_coeff) * outter_overload_value
            old_constraint_coeff = constraint_coeff


    return old_constraint_coeff, pattern_cost_and_amount_list


ttt = np.zeros((5,))

def in_out_separation_decomposition_with_preprocessing(demand_list, outter_flow_per_commodity, outter_overload_value, inner_flow_per_commodity,
                                                        inner_overload_value, arc_capacity, iterative_separation=False):
    # makes some preprocessing then calls the method that will make the decomposition and compute the cut
    # afterwards the coefficient of the cuts are lifted
    nb_commodities = len(demand_list)
    fixed_pattern = []
    variable_pattern = []

    temp = time.time()

    for commodity_index in range(nb_commodities):
        outter_flow, inner_flow = outter_flow_per_commodity[commodity_index], inner_flow_per_commodity[commodity_index]

        if outter_flow == 0 and inner_flow == 0:
            pass

        elif outter_flow == 1 and inner_flow == 1:
            fixed_pattern.append(commodity_index)

        else:
            # if abs(outter_flow - inner_flow) < 10**-3:
            #     outter_flow_per_commodity[commodity_index] = inner_flow_per_commodity[commodity_index]
            variable_pattern.append(commodity_index)


    variable_demand_list = [demand_list[commodity_index] for commodity_index in variable_pattern]
    variable_outter_flow_per_commodity = [outter_flow_per_commodity[commodity_index] for commodity_index in variable_pattern]
    variable_inner_flow_per_commodity = [inner_flow_per_commodity[commodity_index] for commodity_index in variable_pattern]
    remaining_arc_capacity = arc_capacity - sum(demand_list[commodity_index] for commodity_index in fixed_pattern)


    ttt[0] += time.time() - temp
    temp = time.time()

    # call to the separation/decomposition algorithm
    if iterative_separation == False:
        constraint_coeff, pre_pattern_cost_and_amount_list = in_out_separation_decomposition(variable_demand_list, variable_outter_flow_per_commodity, outter_overload_value, variable_inner_flow_per_commodity, inner_overload_value, remaining_arc_capacity)
    else:
        constraint_coeff, pre_pattern_cost_and_amount_list = in_out_separation_decomposition_iterative2(variable_demand_list, variable_outter_flow_per_commodity, outter_overload_value, variable_inner_flow_per_commodity, inner_overload_value, remaining_arc_capacity)

    ttt[1] += time.time() - temp
    temp = time.time()

    variable_commodity_coeff_list, overload_coeff, constant_coeff = constraint_coeff
    pattern_cost_and_amount_list = [([variable_pattern[index] for index in pattern] + fixed_pattern, pattern_cost, amount) for pattern, pattern_cost, amount in pre_pattern_cost_and_amount_list]

    if overload_coeff == 0:
        return (np.zeros(nb_commodities), 0, 0), pattern_cost_and_amount_list

    # lifting of the cut's coefficients
    commodity_coeff_list, constant_coeff, lifting_pattern_and_cost_list = compute_all_lifted_coefficients(demand_list, variable_pattern, variable_commodity_coeff_list, fixed_pattern, constant_coeff, remaining_arc_capacity)

    for pattern, pattern_cost in lifting_pattern_and_cost_list:
        pattern_cost_and_amount_list.append((pattern, pattern_cost, 0))

    ttt[2] += time.time() - temp

    return (commodity_coeff_list, overload_coeff, constant_coeff), pattern_cost_and_amount_list


def knapsack_solver(value_list, weight_list, capacity, precision=10**-7):
    # this function solves a classical knapsack problem by calling a MINKNAP algorithm coded in c++
    nb_objects = len(value_list)

    if capacity <= 0:
        return [0] * nb_objects, -10**10

    value_list_rounded = (value_list/ precision).astype(int)

    instance = knapsacksolver.Instance()
    instance.set_capacity(capacity)

    for object_index in range(nb_objects):
        instance.add_item(weight_list[object_index], value_list_rounded[object_index])

    solution = knapsacksolver.solve(instance, algorithm = "minknap", verbose = False)

    return [solution.contains(object_index) for object_index in range(nb_objects)], solution.profit() * precision



def gurobi_knapsack_solver(value_list, weight_list, capacity):
    # this function solves a classical knapsack problem by solving a MILP model with gurobi
    nb_commodities = len(weight_list)

    # Create optimization model
    model = gurobipy.Model()
    model.modelSense = gurobipy.GRB.MAXIMIZE
    model.Params.OutputFlag = 1

    # Create variables
    choice_var = model.addVars(nb_commodities, obj=value_list, vtype='B')

    constraint = model.addConstr((sum(choice_var[commodity_index] * weight_list[commodity_index] for commodity_index in range(nb_commodities)) <= capacity))

    model.update()
    model.optimize()

    return [choice_var[commodity_index].X > 0.5 for commodity_index in range(nb_commodities)], model.ObjVal


def penalized_knapsack_optimizer(demand_list, arc_capacity, objective_coeff_per_commodity, overload_objective_coeff=1, verbose=0):
    # this function solves a special knapsack problem where over-capacitating the knapsack is allowed but penalised
    # this problem can be solved by solving two classical knapsack problem (this is what is done here)
    nb_commodities = len(demand_list)

    first_solution, first_solution_value = knapsack_solver(np.array(objective_coeff_per_commodity), demand_list, arc_capacity)

    value_array = overload_objective_coeff * np.array(demand_list) - np.array(objective_coeff_per_commodity)
    value_list = np.array(value_array)
    weight_list = np.array(demand_list)
    for commodity_index in range(nb_commodities):
        if value_list[commodity_index] <= 0:
            value_list[commodity_index] = 0
            weight_list[commodity_index] = 2*(sum(demand_list) - arc_capacity)

    second_solution, second_solution_value = knapsack_solver(value_list, weight_list, sum(demand_list) - arc_capacity)
    second_solution_value = second_solution_value + overload_objective_coeff * arc_capacity - sum(value_array)

    if first_solution_value >= second_solution_value:
        return [commodity_index for commodity_index in range(nb_commodities) if first_solution[commodity_index] and objective_coeff_per_commodity[commodity_index] !=0], first_solution_value

    else:
        return [commodity_index for commodity_index in range(nb_commodities) if not second_solution[commodity_index] and objective_coeff_per_commodity[commodity_index] !=0], second_solution_value


def gurobi_penalized_knapsack_optimizer(demand_list, arc_capacity, objective_coeff_per_commodity, overload_objective_coeff=1, verbose=0):
    # this function solves a special knapsack problem where over-capacitating the knapsack is allowed but penalised
    # this is done by solving a MILP model with gurobi
    nb_commodities = len(demand_list)

    # Create optimization model
    model = gurobipy.Model()
    model.modelSense = gurobipy.GRB.MAXIMIZE
    model.Params.OutputFlag = verbose

    # Create variables
    choice_var = model.addVars(list(range(nb_commodities)), obj=objective_coeff_per_commodity, vtype='B')
    delta_var = model.addVar(obj=-overload_objective_coeff)

    model.addConstr((sum(choice_var[commodity_index] * demand_list[commodity_index] for commodity_index in range(nb_commodities)) - delta_var <= arc_capacity))

    model.update()
    model.optimize()

    return [commodity_index for commodity_index in range(nb_commodities) if choice_var[commodity_index].X > 0.5], model.ObjVal

if __name__ == "__main__":
    for i in range(10**2):
        demand_list = np.random.randint(1, 100, size=10)
        commodity_dual_value_list = np.random.random(size=10) * 100
        # commodity_dual_value_list = (commodity_dual_value_list/ 10**-3).astype(int)
        arc_capacity = sum(demand_list)//2
        # assert knapsack_solver(commodity_dual_value_list, demand_list, arc_capacity)[1] == gurobi_knapsack_solver(commodity_dual_value_list, demand_list, arc_capacity)[1]
        print(demand_list)
        print(commodity_dual_value_list)
        print(arc_capacity)
        a = penalized_knapsack_optimizer(demand_list, arc_capacity, commodity_dual_value_list)
        b = gurobi_penalized_knapsack_optimizer(demand_list, arc_capacity, commodity_dual_value_list)
        print(a)
        print(b)
        c = sum(commodity_dual_value_list[commodity_index] for commodity_index in a[0]) - max(0, sum(demand_list[commodity_index] for commodity_index in a[0]) - arc_capacity)
        d = sum(commodity_dual_value_list[commodity_index] for commodity_index in b[0]) - max(0, sum(demand_list[commodity_index] for commodity_index in b[0]) - arc_capacity)
        e = max(0, sum(demand_list[commodity_index] for commodity_index in a[0]) - arc_capacity)
        f = max(0, sum(demand_list[commodity_index] for commodity_index in b[0]) - arc_capacity)
        assert a[0] == b[0], (a,b, i, c, d, e, f)

if False and __name__ == "__main__":
    arc_capacity = 10**4
    nb_commodities = 200
    demand_list = []

    for commodity_index in range(nb_commodities):
        demand_list.append(random.randint(arc_capacity/nb_commodities, 3*arc_capacity/nb_commodities))

    outter_flow_per_commodity = [0]*nb_commodities
    inner_flow_per_commodity = [0]*nb_commodities
    outter_remaining_arc_capacity = arc_capacity
    inner_remaining_arc_capacity = arc_capacity
    for commodity_index in range(nb_commodities // 3):
        outter_flow_per_commodity[commodity_index] = 10
        outter_remaining_arc_capacity -= demand_list[commodity_index]
        inner_flow_per_commodity[commodity_index] = 10
        inner_remaining_arc_capacity -= demand_list[commodity_index]

    while outter_remaining_arc_capacity > 0:
        commodity_index = random.randint(0, nb_commodities-1)
        if outter_flow_per_commodity[commodity_index] < 10:
            outter_flow_per_commodity[commodity_index] += 1
            outter_remaining_arc_capacity -= 0.1 * demand_list[commodity_index]

    outter_overload_value = 0

    while inner_remaining_arc_capacity > 0:
        commodity_index = random.randint(0, nb_commodities-1)
        if inner_flow_per_commodity[commodity_index] < 10:
            inner_flow_per_commodity[commodity_index] += 1
            inner_remaining_arc_capacity -= 0.1 * demand_list[commodity_index]

    inner_flow_per_commodity, outter_flow_per_commodity = np.array(inner_flow_per_commodity) / 10, np.array(outter_flow_per_commodity) / 10
    # inner_flow_per_commodity = [min(1, max(0, outter_flow + (random.random()-0.5) * 10**-5)) for outter_flow in outter_flow_per_commodity]


    print(inner_flow_per_commodity, outter_flow_per_commodity)
    pattern_cost_and_amount_list = compute_approximate_decomposition(demand_list, inner_flow_per_commodity, arc_capacity)
    inner_overload_value = sum(pattern_cost * amount for pattern, pattern_cost, amount in pattern_cost_and_amount_list)

    # _, pattern_cost_and_amount_list = separation_decomposition_with_preprocessing(demand_list, inner_flow_per_commodity, arc_capacity)
    # inner_overload_value = sum(pattern_cost * amount for pattern, pattern_cost, amount in pattern_cost_and_amount_list) + 10**-5
    # _, pattern_cost_and_amount_list = separation_decomposition_with_preprocessing(demand_list, outter_flow_per_commodity, arc_capacity)
    # outter_overload_value = sum(pattern_cost * amount for pattern, pattern_cost, amount in pattern_cost_and_amount_list) - 10**-5
    # print(inner_overload_value, outter_overload_value)


    # constraint_coeff, pattern_cost_and_amount_list = in_out_separation_decomposition_with_preprocessing(demand_list, outter_flow_per_commodity, outter_overload_value, inner_flow_per_commodity, inner_overload_value, arc_capacity, iterative_separation=True)
    constraint_coeff, pattern_cost_and_amount_list = separation_decomposition(demand_list, outter_flow_per_commodity, arc_capacity)
    # constraint_coeff, pattern_cost_and_amount_list = separation_decomposition_with_preprocessing(demand_list, outter_flow_per_commodity, arc_capacity)
    print("\n", constraint_coeff)
