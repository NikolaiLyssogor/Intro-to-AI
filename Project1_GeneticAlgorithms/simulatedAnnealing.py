from makeRandomExpressions import generate_random_expr
from fitnessAndValidityFunctions import is_viable_expr, compute_fitness
from random import choices, random 
import math 
from crossOverOperators import random_expression_mutation, random_subtree_crossover
from geneticAlgParams import GAParams
from geneticSearchAlgorithms import GASolver
import numpy as np


# Implement simulated annealing: this is not compulsory but
# one of the ways you can go "above and beyond" is to compare GA
# to simulated annealing.
# This is not the main assignment however -- please attempt this only
#  if you have solved GA with plenty of time left over.
# Function: run_simulated_annealing
# Inputs: n_steps -- number of steps of SA to run.
#        lst_of_identifiers -- list of identifiers
#        params -- parameters for the problem (same as geneticAlgParams)
#        Useful params for simulated annealing are
#        params.simulated_annealing_cool_steps=100
#        params.simulated_annealing_cool_frac = 0.8
#        params.simulated_annealing_start_temp = 100
#  Feel free to reuse ideas from assignment 4.
def simulated_annealing_single_step(expr, identifiers, params):
    while(True):
        new_expr = random_expression_mutation(expr, identifiers, params)
        if(is_viable_expr(new_expr, identifiers, params)):
            fit_diff = compute_fitness(new_expr, identifiers, params) - compute_fitness(expr, identifiers, params)
            if(fit_diff >= 0):
                return new_expr
            else:
                r = math.exp(fit_diff/params.temperature)
                if(r >= random()):
                    return new_expr
                else:
                    return expr

def run_simulated_annealing(n_steps, identifiers, params):
    expr = generate_random_expr(params.depth, identifiers, params)
    stats = [compute_fitness(expr, identifiers, params)]
    best_fitness = -float('inf')
    best_expr = None

    for i in range(n_steps):
        if(i > 0 and i % params.simulated_annealing_cool_steps == 0):
            params.temperature = params.simulated_annealing_cool_frac * params.temperature
        expr = simulated_annealing_single_step(expr, identifiers, params)
        current_fitness = compute_fitness(expr, identifiers, params)
        stats.append(current_fitness)
        if current_fitness > best_fitness:
            best_expr = expr
            best_fitness = current_fitness

    return (best_expr, best_fitness, stats)

# Run test on a toy problem.
if __name__ == '__main__':
    params = GAParams()
    params.regression_training_data = [
        ([-2.0 + 0.02*j], 5.0 * math.cos(-2.0 + 0.02*j) - math.sin((-2.0 + 0.02*j)/10.0)) for j in range(201)
    ]
    params.test_points = list([ [-4.0 + 0.02 * j] for j in range(401)])
    run_simulated_annealing(30000, ['x'], params)
