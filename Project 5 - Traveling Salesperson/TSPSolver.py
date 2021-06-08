#!/usr/bin/python3
import copy
import heapq
import sys

from TSPClasses import *


# Reducing / Updating the Cost Matrix
# Time Complexity = O(num_row_cities * num_col_cities) : must iterate through all row_cites and all col_cities 2x
# Space Complexity = O(num_cities^2) : must store a num_cities x num_cities matrix
def ReduceMatrix(matrix, row_cities, col_cities):
    min_row = np.min(matrix, axis=1)
    bound = 0

    # Reduce so that there is a 0 in every row
    for icity in row_cities:
        i = icity._index
        min_row_i = min_row[i]
        if min_row_i == math.inf:
            return math.inf
        bound += min_row_i
        for jcity in col_cities:
            matrix[i][jcity._index] -= min_row_i

    # Reduce so that there is a 0 in every column
    min_col = np.min(matrix, axis=0)
    for jcity in col_cities:
        j = jcity._index
        min_col_j = min_col[j]
        if min_col_j == math.inf:
            return math.inf
        bound += min_col_j
        for icity in row_cities:
            matrix[icity._index][j] -= min_col_j

    return bound


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    # Generate a Matrix from the List of Cities
    # Time Complexity = O(num_cities^2) : must iterate through each city for each city 2x
    # Space Complexity = O(num_cities^2) : must store a num_cities x num_cities matrix
    def Matrix(self, cities):
        M = [[math.inf for x in cities] for x in cities]
        for icity in cities:
            i = icity._index
            for jcity in cities:
                j = jcity._index
                if self._scenario._edge_exists[i][j]:
                    M[i][j] = cities[i].costTo(cities[j])
        return M

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution, 
        time spent to find solution, number of permutations tried during search, the 
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    # Execute Random Tour Algorithm to Find Any Traveling Salesperson Tour
    # Time Complexity = Omega(num_cities) : must iterate through all the cities at least once.
    # The while loop will likely run a constant number of times, so Omega time complexity makes sense here.
    # Space Complexity = O(num_cities) : must store a list of cities and equally long list for the route.
    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for 
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def expand_greedy(self, matrix, row_num, visited, num_cities, start_time, time_allowance):
        sys.setrecursionlimit(100000000)
        if len(visited) == num_cities:
            if matrix[row_num][0] != math.inf:
                return visited
            else:
                return None
        Q = []
        heapq.heapify(Q)
        row = matrix[row_num]
        for i in range(num_cities):
            if row[i] != math.inf and i not in visited:
                heapq.heappush(Q, (row[i], i))
        while time.time() - start_time < time_allowance:
            if len(Q):
                topValue, topIndex = heapq.heappop(Q)
                new_visited = copy.deepcopy(visited)
                new_visited.append(topIndex)
                path = self.expand_greedy(matrix, topIndex, new_visited, num_cities, start_time, time_allowance)
                if path is not None:
                    return path
            else:
                return None
        return None

    def greedy(self, time_allowance=60.0):
        start_time = time.time()

        BSSF = None
        foundTour = False

        num_updates = 0

        cities = self._scenario.getCities()
        num_cities = len(cities)
        M = self.Matrix(cities)

        path = self.expand_greedy(M, 0, [0], num_cities, start_time, time_allowance)

        if path is not None:
            route = []
            for i in range(num_cities):
                route.append(cities[path[i]])
            BSSF = TSPSolution(route)
            num_updates += 1
            foundTour = True

        return {'cost': BSSF.cost if foundTour else math.inf, 'time': time.time() - start_time, 'count': num_updates,
                'soln': BSSF, 'max': None, 'total': None, 'pruned': None}

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''

    # Execute Branch and Bound Algorithm to Find Shortest Traveling Salesperson Tour
    # Time Complexity = O((num_cities + 3)!)
    # Space Complexity = O(num_cities!)
    def branchAndBound(self, time_allowance=60.0):
        start_time = time.time()

        # Initialize BSSF
        # Time Complexity = Omega(num_cities)
        # Space Complexity = O(num_cities)
        greedy = self.greedy(time_allowance)
        lowest_cost = greedy['cost']
        BSSF = greedy['soln']

        # Generate Initial Search State
        # Time Complexity = O(num_cities^2) : must run the Matrix function to generate a matrix
        # Space Complexity = O(num_cities^2) : must store a num_cities x num_cities matrix
        max_queue_size = 1
        pruned_states = 0
        num_updates = 0
        total_states = 1

        cities = self._scenario.getCities()
        num_cities = len(cities)
        MR = self.Matrix(cities)

        # Reducing the Cost Matrix
        # Time Complexity = O(num_cities^2)
        # Space Complexity = O(num_cities^2)
        lower_bound = ReduceMatrix(MR, cities, cities)

        path = [0]

        # Initialize Priority Queue
        # Time Complexity = O(1)
        # Space Complexity = O(1)
        heap = []
        heapq.heapify(heap)
        heapq.heappush(heap, State(current_city=cities[0], RCM=copy.deepcopy(MR), bound=lower_bound,
                                   remaining_cities=copy.deepcopy(cities), path=path, depth=0))

        # num_states = O(num_cities!)
        # num_states empirical analysis = polynomial
        while len(heap) and time.time() - start_time < time_allowance:

            # Expanding a Search State
            # Time Complexity = O(num_cities^3)
            # Space Complexity = O(num_cities^4)
            topState = heapq.heappop(heap)
            topState_cost = topState.bound
            if topState_cost >= lowest_cost:
                pruned_states += 1
                continue

            top_remaining_cities = topState.remaining_cities
            row_cities = copy.deepcopy(top_remaining_cities)
            row_cities.remove(row_cities[0])

            for city in top_remaining_cities:

                # Generate a Search State
                # Time Complexity = O(num_cities^2) : must iterate through each city for each city 2x
                # Space Complexity = O(num_cities^2) : must store a num_cities x num_cities matrix
                total_states += 1

                current_city_index = topState.current_city._index
                next_city_index = city._index

                path = copy.deepcopy(topState.path)
                if next_city_index == 0:
                    continue

                path.append(next_city_index)

                i = top_remaining_cities.index(city)
                col_cities = copy.deepcopy(top_remaining_cities)
                col_cities.remove(col_cities[i])

                MR = np.array(copy.deepcopy(topState.RCM))
                lower_bound = topState_cost + MR[current_city_index, next_city_index]
                MR[current_city_index, :] = math.inf
                MR[:, next_city_index] = math.inf
                MR[next_city_index, current_city_index] = math.inf

                # Updating the Cost Matrix
                # Time Complexity = O(num_row_cities * num_col_cities)
                # Space Complexity = O(num_cities^2)
                lower_bound += ReduceMatrix(MR, row_cities, col_cities)

                if lower_bound >= lowest_cost:
                    pruned_states += 1
                    continue
                else:

                    # The Priority Queue
                    # Time Complexity = Insertions / Deletions = O(num_cities)
                    # Space Complexity = O(num_states)
                    heapq.heappush(heap, State(current_city=city, RCM=copy.deepcopy(MR), bound=lower_bound,
                                               remaining_cities=copy.deepcopy(col_cities), path=path,
                                               depth=topState.depth + 1))
                    if len(heap) > max_queue_size:
                        max_queue_size = len(heap)

            # Update BSSF
            if len(row_cities) == 0:  # Reached the end of the tour.
                route = []
                for i in range(num_cities):
                    route.append(cities[path[i]])
                BSSF = TSPSolution(route)
                lowest_cost = topState_cost
                num_updates += 1

        return {'cost': BSSF.cost, 'time': time.time() - start_time, 'count': num_updates, 'soln': BSSF,
                'max': max_queue_size, 'total': total_states, 'pruned': pruned_states + len(heap)}

    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''

    def fancy(self, time_allowance=60.0):
        pass


class State:
    def __init__(self, current_city=None, RCM=None, bound=None, remaining_cities=None, path=None, depth=None):
        self.current_city = current_city
        self.RCM = RCM
        self.bound = bound
        self.remaining_cities = remaining_cities
        self.path = path
        self.depth = depth

    def __lt__(self, obj2):
        if self.depth == obj2.depth:
            return self.bound < obj2.bound
        else:
            return self.depth > obj2.depth
