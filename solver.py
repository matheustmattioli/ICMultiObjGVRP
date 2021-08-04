import sys
import time
import math
import random
from queue import Queue
from collections import namedtuple

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

N_ITE_MS_CLUST = 10 # Number of iterations multistart cluster.
MAX_ITER_W_NO_IMPROV = 50
N_ITE_GRASP = 100 # Number of iterations GRASP.
ALPHA_MAX = 0.5
MAX_DEPTH = 4
NUMERICAL_ERROR = 0.00001
DEBUG = 0

# Calculate distance between two nodes.
def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

# Calculate pollution factor between two nodes.
def factor(customer1, customer2, pollution_factors):
    i = customer1
    j = customer2
    if i < j:
        i, j = j, i
    return pollution_factors[i][j]

# Calculate length of a tour.
def calc_obj(vehicle_tours, customers):
    obj = 0
    if len(vehicle_tours) > 0:
        for v in range(len(vehicle_tours)):
            obj += length(customers[vehicle_tours[v - 1]], customers[vehicle_tours[v]])
    return obj

# Calculate factor of a tour.
def calc_factorobj(vehicle_tours, pollution_factors):
    obj = 0
    if len(vehicle_tours) > 0:
        for v in range(len(vehicle_tours)):
            obj += factor(vehicle_tours[v - 1], vehicle_tours[v], pollution_factors)
    return obj

# Making clusters of customers with greedy path technique. In
# which we select a random customer to be the center of cluster and 
# do a nearest neighbour addiction until we fulfill the capacity of the vehicle. 
def greedyPathClusterization(vehicle_capacity, dictofcustomers, customers, customer_count, customers_visited, pollution_factors, weights):
    cluster = [] # Subset of customers
    aux = 1
    cluster.append(int(0))
    capacity_remaining = vehicle_capacity

    flag_time = 0
    if len(dictofcustomers.keys()) > 0: # If we still have customers to attend.
        j = random.choice(list(dictofcustomers.keys())) # Choose a random customer to be the center.
        
        capacity_remaining -= customers[j].demand
        cluster.append(int(j))
        aux += 1 # aux is an iterator who appoint to the next slot in cluster list.
        customers_visited += 1

        remove = dictofcustomers.pop(j) # For debug purposes, we pop the value in "remove".
        if DEBUG > 1:
            print(remove)
        while(capacity_remaining > 0 and customers_visited <= customer_count - 1 and (time.time() - start) < 1800):
            best_value = float('inf')
            flag_capacity = 1
            for v in range(1, customer_count + 1):
                if v in dictofcustomers and capacity_remaining - customers[v].demand >= 0:
                    minl = length(customers[cluster[aux - 1]], customers[v])
                    minf = factor(cluster[aux - 1], v, pollution_factors)
                    minv = minl*weights[0] + minf*weights[1]
                    flag_capacity = 0
                    if minv < best_value: # Select nearest neighbour
                        best_value = minv
                        customer_candidate = v
            # If no one neighbour is considered, then capacity remaining in vehicle is greater than 0, but isn't enough to attend any customer.
            if flag_capacity == 1:
                break
            # Insert selected customer in cluster
            cluster.append(int(customer_candidate))
            aux += 1
            capacity_remaining -= customers[customer_candidate].demand 

            remove = dictofcustomers.pop(customer_candidate)
            if DEBUG > 1:
                print(remove)

            customers_visited += 1

    cluster.append(int(0))
    if time.time() - start >= 1800:
        flag_time = 1
    return cluster, customers_visited, flag_time

# Constuctive heuristic for Traveling Salesman Problem (TSP).
# It's a nearest neighbour (NN) algorithm with Restricted Candidate List (RCL).
def greedypath_RCL(circuit, customers, ALPHA, pollution_factors, weights):
    # Do hamiltonian circuit with greedy choices in selected subset
    # Selected by clusters function
    # using RCL
    nodeCount = len(circuit)
    obj_BS = float('inf')
    flag_time = 0
    #for v in circuit:
    for v in range(1):
        dictofpositions = {circuit[i] : circuit[i] for i in range(nodeCount)}
        solution_greedy = [0 for i in range(nodeCount)]
        k = 0
        solution_greedy[k] = dictofpositions.pop(v)
        k += 1
        # greedy choices
        while k < nodeCount and (time.time() - start) < 1800: 
            nearest_value = float('inf')
            farthest_value = 0
            # deciding nearest and farthest customers from  k - 1 customer
            for n in circuit:
                if n in dictofpositions:
                    lengthNL = length(customers[solution_greedy[k - 1]], customers[n])
                    lengthNF = factor(solution_greedy[k - 1], n, pollution_factors)
                    lengthN = lengthNL*weights[0] + lengthNF*weights[1]
                    if lengthN < nearest_value:
                        nearest_value = lengthN
                    if lengthN > farthest_value:
                        farthest_value = lengthN
            RCL = []
            # filling in RCL
            for n in circuit:
                if n in dictofpositions:
                    lengthNL = length(customers[solution_greedy[k - 1]], customers[n])
                    lengthNF = factor(solution_greedy[k - 1], n, pollution_factors)
                    lengthN = lengthNL*weights[0] + lengthNF*weights[1]
                    if lengthN <= (nearest_value + (farthest_value - nearest_value)*ALPHA): # Condition to insert neighbours in RCL
                        RCL.append(n)
            solution_greedy[k] = random.choice(RCL)
            remove = dictofpositions.pop(solution_greedy[k])
            if DEBUG > 1:
                print("Vertice escolhido para a posicao k = %d", k)
                print(remove)
            k += 1
        # Decide best solution found
        curr_objL = calc_obj(solution_greedy, customers)
        curr_objF = calc_factorobj(solution_greedy, pollution_factors)
        curr_obj = curr_objL*weights[0] + curr_objF*weights[1]
        if curr_obj < obj_BS:
            obj_BS = curr_obj
            best_solution = solution_greedy
        dictofpositions.clear()
    if time.time() - start >= 1800:
        flag_time = 1
    return best_solution, flag_time

# Local Search for TSP with 2OPT neighbour structure.
# Adaptated for multiobj optimization
def localSearch2OPT(cluster, customers, pollution_factors, weights):
    # Small improvements in solutions through analyze of neighborhoods
    # Cost of the initial solution
    objL = calc_obj(cluster, customers)
    objF = calc_factorobj(cluster, pollution_factors)
    obj = objL*weights[0] + objF*weights[1]
    customer_count = len(cluster)
    # Initialize variables with datas from the initial solution
    lenght_BS = obj 
    lenght_solution = lenght_BS
    best_solution = list(cluster)
    count_iteration = 0 # for debug
    # Main loop
    while (time.time() - start) < 1800: 
        try:
            if DEBUG >= 2:
                start_while = time.time()
            lenght_BF = lenght_solution
            best_x = customer_count - 1
            best_y = 0
            # 2-OPT NEIGHBORHOOD
            for x in range(0, customer_count - 2):
                for y in range(x + 1, customer_count - 1):
                    edgeAL = length(customers[cluster[x]], customers[cluster[x - 1]])
                    edgeAF = factor(cluster[x], cluster[x - 1], pollution_factors)
                    edgeA = edgeAL*weights[0] + edgeAF*weights[1]

                    edgeBL = length(customers[cluster[y]], customers[cluster[(y + 1) % customer_count]])
                    edgeBF = factor(cluster[y], cluster[(y + 1) % customer_count], pollution_factors)
                    edgeB = edgeBL*weights[0] + edgeBF*weights[1]

                    edgeCL = length(customers[cluster[x]], customers[cluster[(y + 1) % customer_count]])
                    edgeCF = factor(cluster[x], cluster[(y + 1) % customer_count], pollution_factors)
                    edgeC = edgeCL*weights[0] + edgeCF*weights[1]

                    edgeDL = length(customers[cluster[y]], customers[cluster[(x - 1)]])
                    edgeDF = factor(cluster[y], cluster[x - 1], pollution_factors)
                    edgeD = edgeDL*weights[0] + edgeDF*weights[1]

                    lenght_PS = lenght_solution - (edgeA + edgeB) 
                    lenght_PS = lenght_PS + (edgeC + edgeD)
                    if lenght_PS < lenght_BF:
                        best_x = x
                        best_y = y
                        lenght_BF = lenght_PS
            cluster[best_x:best_y + 1] =  cluster[best_x:best_y + 1][::-1]
            lenght_solution = lenght_BF
            # Update solution
            if lenght_solution < lenght_BS:
                best_solution = list(cluster)
                lenght_BS = lenght_solution
                if DEBUG >= 2:
                    print("--------------------------------------------------------")
                    print(lenght_BS)
                    end_while = time.time()
                    count_iteration += 1
                    print("tempo do loop", end_while - start_while)
                    print("iteracao numero ", count_iteration)
            else:
                break                
        except KeyboardInterrupt:
            break
    return best_solution, lenght_BS


# Greedy Randomized Adaptative Search Procedure (GRASP) implementation with 
# Local Search 2-OPT as "Search Procedure".
def GRASP(cluster_vehicle, customers, pollution_factors, weights):
    best_value = float('inf')
    ALPHA = 0
    n_iter_w_no_improv = 0
    random.seed(2020 + 0)
    i = 1
    while ALPHA <= ALPHA_MAX + NUMERICAL_ERROR and n_iter_w_no_improv < MAX_ITER_W_NO_IMPROV and (time.time() - start) < 1800:
        #print("iteracao ", i, " tempo ", time.time() - start)
        solution, flag = greedypath_RCL(cluster_vehicle[:-1], customers, ALPHA, pollution_factors, weights)
        if flag == 1:
            break
        # solution_obj = calc_lenght(solution, nodeCount, points)
        #solution, solution_obj = local_searchVNS(solution, points, nodeCount)
        #print("solution obj", solution_obj)
        solution, solution_obj = localSearch2OPT(solution, customers, pollution_factors, weights)
        n_iter_w_no_improv += 1

        if solution_obj < best_value:
            best_value = solution_obj
            best_solution = list(solution)
            n_iter_w_no_improv = 0
        if N_ITE_GRASP <= 1:
            break
        ALPHA += ALPHA_MAX/(N_ITE_GRASP - 1)
        if n_iter_w_no_improv == MAX_ITER_W_NO_IMPROV and DEBUG >= 3:
            print("patinou e saiu na iteracao ", i)
        i += 1
    return best_solution, best_value

# Receive input data of an instance and call the solver.
def solve_it(input_data):
    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    for i in range(1, customer_count + 1):
        line = lines[i]
        parts = line.split()
        customers.append(
            Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))
    
    pollution_factors = [[int(0) for i in range(customer_count)] for i in range(customer_count)]
    i = 1
    for k in range(customer_count + 2, 2*customer_count + 1):
        j = 0
        parts = lines[k].split()
        while i > j:
            pollution_factors[i][j] = float(parts[j])
            j += 1
        i += 1

    if DEBUG >= 1:
        print(f"Numero de clientes = {customer_count}")
        print(f"Numero de veiculos = {vehicle_count}")
        print(f"Capacidade dos veiculos = {vehicle_capacity}")

    if DEBUG >= 2:
        print("Lista de clientes:")
        for customer in customers:
            print(
                f"index do cliente = {customer.index}, demanda do cliente = {customer.demand}, ({customer.x}, {customer.y})")
        print()
    
    if DEBUG >= 3:
        for i in range(customer_count):
            for j in range(customer_count):
                if i > j:
                    print(pollution_factors[i][j])

    return weight_generator(customer_count, vehicle_count, vehicle_capacity, customers, pollution_factors)

# Solver for VRP.
# Divided in two steps:
#     1º: Clusterization
#     2º: Draw routes for each cluster (TSP)
def VRPsolver(customer_count, vehicle_count, vehicle_capacity, customers, pollution_factors, weights):
    best_obj = float('inf')
    best_objL = float('inf')
    best_objF = float('inf')
    flag_time = 0 # Stop the program when our time run out.
    #best_solution = []
    for k in range(N_ITE_MS_CLUST):
        if flag_time == 1:
            break
        flag_viability = 1
        solution = []
        dictofcustomers = {i : customers[i] for i in range(1, customer_count)}
        customers_visited = 0
        # Choose type of clusterization
        for i in range(vehicle_count):
            # Choose type of clusterization
            cluster_vehicle, customers_visited, flag_time = greedyPathClusterization(vehicle_capacity, dictofcustomers, customers, customer_count, customers_visited, pollution_factors, weights)
            if flag_time == 1:
                break
                
            # GRASP
            best_solution_vehicle, flag_time = GRASP(cluster_vehicle, customers, pollution_factors, weights)
            if flag_time == 1:
                break

            for j in range(len(best_solution_vehicle)):
                if best_solution_vehicle[j] == 0:
                    best_solution_vehicle = best_solution_vehicle[j:] + best_solution_vehicle[:j + 1]
                    break
            solution.append(best_solution_vehicle) 

        # Verify if are customers unvisited
        if customers_visited < customer_count - 1 and flag_time == 0:
            flag_viability = 0 # Flag to check viability of solution found
        
        # Calculate cost of VRP
        obj = 0
        objL = 0
        objF = 0
        for i in range(vehicle_count):
            objL += calc_obj(solution[i], customers)
            objF += calc_factorobj(solution[i], pollution_factors)
        obj = objF + objL
        #print("1ª it = ", obj)
        if obj < best_obj and flag_viability == 1:
            best_obj = obj
            best_objL = objL
            best_objF = objF
            # best_solution = solution
            if DEBUG >= 3:
                print("Best solution in k = ", k, "loop")
        dictofcustomers.clear()

    # prepare the solution in the specified output format
    outputData = [best_objL, best_objF]
    return outputData


# To guide ours meta-heuristics in the pareto frontier, 
# we generate weights through this method.
def weight_generator(customer_count, vehicle_count, vehicle_capacity, customers, pollution_factors):
    # Solutions that we obtain in each iteration of the program.
    solution = []
    # The first and second solution ignore one of the objetive function 
    solution.append(VRPsolver(customer_count, vehicle_count, vehicle_capacity, customers, pollution_factors, [1, 0]))
    solution.append(VRPsolver(customer_count, vehicle_count, vehicle_capacity, customers, pollution_factors, [0, 1]))
    # Weights are calculate with a par calculation and put in that Queue named pairs_of_points.
    pairs_of_points = Queue(maxsize = 0)
    pairs_of_points.put([[1, 0], [0, 1]])
    number_points = 2
    # Thats define the number of solutions we obtain
    while number_points < pow(2, MAX_DEPTH + 1) - 1 + 2:
        pair = pairs_of_points.get()
        pointL = pair[0]
        pointR = pair[1]
        # Par calculation
        pointM = [(pointL[0] + pointR[0])/2, (pointL[1] + pointR[1])/2] 
        # Obtain new solution with weight calculate
        solution.append(VRPsolver(customer_count, vehicle_count, vehicle_capacity, customers, pollution_factors, pointM)) 
        pairs_of_points.put([pointL, pointM])
        pairs_of_points.put([pointM, pointR])
        number_points += 1
    return solution

start = time.time()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        output_data = solve_it(input_data)
        solution_file = open(file_location + ".csv", "w")
        for i in range(len(output_data)):
            solution_file.write('%.2f' % output_data[i][0] + ', ' + '%.2f' % output_data[i][1])
            solution_file.write('\n')
        solution_file.close()
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./basic/vrp_5_4_1)')


end = time.time()
print(end - start)
