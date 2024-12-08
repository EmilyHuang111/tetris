# train.py

import numpy as np
from genetic import Genetic_AI
from game import Game
import pandas as pd
import random
from genetic_helpers import (
    bool_to_np,
    get_peaks,
    get_holes,
    get_wells,
    get_bumpiness,
    get_row_transition,
    get_col_transition
)
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cross(a1, a2, aggregate='lin'):
    """
    Compute crossover of two agents, returning a new agent
    """
    new_genotype = []
    # Avoid division by zero
    a1_prop = a1.fit_rel / a2.fit_rel if a2.fit_rel != 0 else 0
    for i in range(len(a1.genotype)):
        rand = random.uniform(0, 1)
        if rand > a1_prop:
            new_genotype.append(a1.genotype[i])
        else:
            new_genotype.append(a2.genotype[i])
    return Genetic_AI(genotype=np.array(new_genotype), aggregate=aggregate, mutate=True)

def compute_fitness(agent, num_trials):
    """
    Given an agent and a number of trials, computes fitness as
    arithmetic mean of pieces dropped over the trials
    """
    fitness = []
    for trial in range(num_trials):
        game = Game('genetic', agent=agent)
        pieces_dropped, rows_cleared = game.run_no_visual()
        fitness.append(pieces_dropped)
        logging.debug(f"    Trial: {trial+1}/{num_trials}")
    return np.average(np.array(fitness))

def compute_fitness_parallel(args):
    """
    Helper function for parallel fitness computation.
    """
    agent, num_trials = args
    fitness = []
    for trial in range(num_trials):
        game = Game('genetic', agent=agent)
        pieces_dropped, rows_cleared = game.run_no_visual()
        fitness.append(pieces_dropped)
    return np.average(np.array(fitness))

def run_X_epochs(num_epochs=10, num_trials=5, pop_size=100, aggregate='lin',
                num_elite=5, survival_rate=.35, logging_file='default.csv',
                save_path='data/best_genotype.npy', use_parallel=True):
    # Initialize data collection
    headers = ['epoch', 'avg_fit', 'avg_gene', 'top_fit', 'top_gene', 'elite_fit', 'elite_gene']
    df = pd.DataFrame(columns=headers)
    df.to_csv(f'data/{logging_file}.csv', index=False)
    
    # Create initial population
    population = [Genetic_AI(aggregate=aggregate) for _ in range(pop_size)]
    
    # Record the overall start time
    overall_start_time = time.time()
    
    # Initialize the outer progress bar for epochs
    for epoch in tqdm(range(num_epochs), desc='Total Epochs', unit='epoch'):
        epoch_start_time = time.time()
        total_fitness = 0
        top_agent = None
        gene_sum = np.zeros_like(population[0].genotype)
        
        if use_parallel:
            # Prepare arguments for parallel processing
            args = [(agent, num_trials) for agent in population]
            
            # Initialize a multiprocessing Pool
            with Pool(processes=cpu_count()) as pool:
                fitness_scores = list(tqdm(pool.imap(compute_fitness_parallel, args),
                                           total=len(args),
                                           desc=f'Epoch {epoch+1} Agents',
                                           unit='agent',
                                           leave=False))
        else:
            # Serial processing with progress bar
            fitness_scores = []
            for agent in tqdm(population, desc=f'Epoch {epoch+1} Agents', unit='agent', leave=False):
                fitness = compute_fitness(agent, num_trials=num_trials)
                fitness_scores.append(fitness)
        
        # Assign fitness scores and find the top agent
        for agent, fitness in zip(population, fitness_scores):
            agent.fit_score = fitness
            total_fitness += agent.fit_score
            gene_sum += agent.genotype
            if top_agent is None or agent.fit_score > top_agent.fit_score:
                top_agent = agent
        
        # Compute relative fitness
        for agent in population:
            agent.fit_rel = agent.fit_score / total_fitness if total_fitness > 0 else 0
        
        # Selection
        next_gen = []
        
        # Sort population by descending fitness
        sorted_pop = sorted(population, reverse=True)
        
        # Elite selection: copy top-performing agents directly to the next generation
        elite_fit_score = 0
        elite_genes = np.zeros_like(population[0].genotype)
        for i in range(num_elite):
            elite_fit_score += sorted_pop[i].fit_score
            elite_genes += sorted_pop[i].genotype
            # Deep copy to ensure no mutation affects the elite
            next_gen.append(Genetic_AI(genotype=np.copy(sorted_pop[i].genotype), mutate=False))
        
        # Save the best genotype of the current epoch
        best_genotype = sorted_pop[0].genotype
        np.save(save_path, best_genotype)
        
        # Selection based on survival rate
        num_parents = max(2, round(pop_size * survival_rate))  # Ensure at least two parents
        parents = sorted_pop[:num_parents]
        
        # Crossover to create the rest of the next generation
        while len(next_gen) < pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child = cross(parent1, parent2, aggregate=aggregate)
            next_gen.append(child)
        
        # Data collection for logging
        avg_fit = total_fitness / pop_size if pop_size > 0 else 0
        avg_gene = gene_sum / pop_size if pop_size > 0 else np.zeros_like(population[0].genotype)
        elite_fit = elite_fit_score / num_elite if num_elite > 0 else 0
        elite_gene = elite_genes / num_elite if num_elite > 0 else np.zeros_like(population[0].genotype)
        
        data = [[epoch+1, avg_fit, avg_gene.tolist(),
                 top_agent.fit_score, top_agent.genotype.tolist(),
                 elite_fit, elite_gene.tolist()]]
        df_epoch = pd.DataFrame(data, columns=headers)
        df_epoch.to_csv(f'data/{logging_file}.csv', mode='a', index=False, header=False)
        
        # Calculate elapsed time for the epoch
        epoch_elapsed = time.time() - epoch_start_time
        
        # Calculate total elapsed time and estimate remaining time
        overall_elapsed = time.time() - overall_start_time
        epochs_completed = epoch + 1
        epochs_remaining = num_epochs - epochs_completed
        average_epoch_time = overall_elapsed / epochs_completed if epochs_completed > 0 else 0
        estimated_remaining = average_epoch_time * epochs_remaining
        
        # Print epoch summary with timing
        print(f'\nEpoch {epoch+1}/{num_epochs} completed:')
        print(f"  Average Fitness: {avg_fit}")
        print(f"  Top Fitness: {top_agent.fit_score}")
        print(f"  Epoch Time: {epoch_elapsed:.2f} seconds")
        print(f"  Estimated Remaining Time: {estimated_remaining/60:.2f} minutes\n")
        
        # Prepare for next generation
        population = next_gen
    
    overall_total_time = time.time() - overall_start_time
    print(f"Training completed in {overall_total_time/60:.2f} minutes.")
    return data

if __name__ == '__main__':
    run_X_epochs(num_epochs=10, num_trials=5, pop_size=50, num_elite=5,
                save_path='data/best_genotype.npy', use_parallel=True)
