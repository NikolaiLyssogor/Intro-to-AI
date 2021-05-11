from makeRandomExpressions import generate_random_expr
from fitnessAndValidityFunctions import is_viable_expr, compute_fitness
from random import choices 
import math 
from crossOverOperators import random_expression_mutation, random_subtree_crossover
from geneticAlgParams import GAParams
from matplotlib import pyplot as plt


class GASolver: 
    def __init__(self, params, lst_of_identifiers, n):
        # Parameters for GA: see geneticAlgParams
        # Also includes test data for regression and checking validity
        self.params = params
        # The population size 
        self.N = n
        # Store the actual population (you can use other data structures if you wish)
        self.pop = []
        while(len(self.pop) < 500):
            expr = generate_random_expr(self.params.depth, lst_of_identifiers, self.params)
            if(is_viable_expr(expr, lst_of_identifiers, self.params)):
                self.pop.append(expr)
        # A list of identifiers for the expressions
        self.identifiers = lst_of_identifiers
        # Maintain statistics on best fitness in each generation
        self.population_stats = []
        # Store best solution so far across all generations
        self.best_solution_so_far = None
        # Store the best fitness so far across all generations
        self.best_fitness_so_far = -float('inf')

    # Please add whatever helper functions you wish.

    # handles overflow errors when using the probability function
    def weight_catch(self, fitness):
        try:
            return math.exp(-fitness/self.params.temperature)
        except OverflowError:
            return -float('inf')

    # TODO: Implement the genetic algorithm as described in the
    # project instructions.
    # This function need not return anything. However, it should
    # update the fields best_solution_so_far, best_fitness_so_far and
    # population_stats
    def run_ga_iterations(self, n_iter=1000):
        # each iteration is one generation
        for i in range(n_iter):
            # print("Generation:", i)
            # keep elite fitness expressions
            k = math.floor(self.N * self.params.elitism_fraction) # number of elite expressions each generation
            sorted_pop = sorted(self.pop, key=lambda expr: compute_fitness(expr, self.identifiers, self.params), reverse=True)
            new_pop = sorted_pop[:k]

            # update statistics
            self.population_stats.append(compute_fitness(new_pop[0], self.identifiers, self.params))
            if(compute_fitness(new_pop[0], self.identifiers, self.params) > self.best_fitness_so_far):
                self.best_fitness_so_far = compute_fitness(new_pop[0], self.identifiers, self.params)
                self.best_solution_so_far = new_pop[0]
                # print("-->Best fitness:", self.best_fitness_so_far)

            # compute weights for selecting expressions in crossover
            fitness_lst = [compute_fitness(expr, self.identifiers, self.params) for expr in sorted_pop]
            weights = [1/self.weight_catch(fitness) for fitness in fitness_lst]
            # print("orig lst:", fitness_lst[:15])
            # print("weights:", weights[:15])

            # get rest of population through crossover and mutation
            while(len(new_pop) < 500):
                selected = choices(sorted_pop, weights=weights, k=2)

                (ea_cross, eb_cross) = random_subtree_crossover(selected[0], selected[1])
                ea_mut = random_expression_mutation(ea_cross, self.identifiers, self.params)
                eb_mut = random_expression_mutation(eb_cross, self.identifiers, self.params)

                if(is_viable_expr(ea_mut, self.identifiers, self.params)):
                    new_pop.append(ea_mut)
                if(is_viable_expr(eb_mut, self.identifiers, self.params)):
                    new_pop.append(eb_mut)

            self.pop = new_pop


## Function: curve_fit_using_genetic_algorithms
# Run curvefitting using given parameters and return best result, best fitness and population statistics.
# DO NOT MODIFY
def curve_fit_using_genetic_algorithm(params, lst_of_identifiers, pop_size, num_iters):
    solver = GASolver(params, lst_of_identifiers, pop_size)
    solver.run_ga_iterations(num_iters)
    return (solver.best_solution_so_far, solver.best_fitness_so_far, solver.population_stats)


# Run test on a toy problem.
if __name__ == '__main__':
    params = GAParams()
    params.regression_training_data = [
       ([-2.0 + 0.02*j], 5.0 * math.cos(-2.0 + 0.02*j) - math.sin((-2.0 + 0.02*j)/10.0)) for j in range(201)
    ]
    params.test_points = list([ [-4.0 + 0.02 * j] for j in range(401)])
    solver = GASolver(params,['x'],500)
    # solver.bozo_test()
    solver.run_ga_iterations(100)
    print('Done!')
    print(f'Best solution found: {solver.best_solution_so_far.simplify()}, fitness = {solver.best_fitness_so_far}')
    stats = solver.population_stats
    niters = len(stats)
    plt.plot(range(niters), [st for st in stats] , 'b-')
    plt.show()



