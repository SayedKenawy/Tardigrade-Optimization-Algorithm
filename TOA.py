# -*- coding: utf-8 -*-
"""
Tardigrade Optimization Algorithm (TOA)
Inspired by the behavior of Tardigrades
Author: Ahmed Mohamed Zaki & El-Sayed M. El-kenawy
"""

import random
import numpy as np
from solution import solution
import time
import math

def safe_float(value):
    """
    Safely convert any value to float for printing
    """
    try:
        if hasattr(value, '__len__') and not isinstance(value, str):
            if hasattr(value, 'flatten'):
                flat_val = value.flatten()
                return float(flat_val[0]) if len(flat_val) > 0 else 0.0
            else:
                return float(value[0]) if len(value) > 0 else 0.0
        else:
            return float(value)
    except (ValueError, TypeError, IndexError):
        return 0.0

def TOA(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    """
    Tardigrade Optimization Algorithm (TOA)
    
    Parameters:
    objf: objective function
    lb: lower bound
    ub: upper bound  
    dim: number of variables
    SearchAgents_no: number of search agents (tardigrades)
    Max_iter: maximum iterations
    """
    
    # Initialize parameters inspired by scientific research
    beta_A1 = 0.5   # reduced for faster convergence
    beta_A2 = 1.2   # reduced to avoid oscillation
    cryptobiosis_threshold = 0.6  # reduced for more activity
    
    levy_probability = 0.1  # probability of Levy flight
    local_search_probability = 0.2  # probability of local search
    elite_ratio = 0.2  # elite ratio
    
    # Convert bounds
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    #  initialization of positions
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        # Mixed initialization: random + center-based
        random_part = np.random.uniform(lb[i], ub[i], SearchAgents_no)
        center_part = np.random.normal((lb[i] + ub[i])/2, (ub[i] - lb[i])/6, SearchAgents_no)
        center_part = np.clip(center_part, lb[i], ub[i])
        
        # Blend random and center-based initialization
        mix_factor = np.random.random(SearchAgents_no)
        Positions[:, i] = np.where(mix_factor < 0.7, random_part, center_part)
    
    #  variables
    hunger_levels = np.random.uniform(0.3, 1.0, SearchAgents_no)  # initial variation
    search_times = np.random.uniform(5, 15, SearchAgents_no)  # search time variation
    velocity = np.zeros((SearchAgents_no, dim))  # velocity for continuity
    personal_best = Positions.copy()  # personal best position for each agent
    personal_best_score = np.full(SearchAgents_no, float('inf'))
    
    # Best solution
    Best_pos = np.zeros(dim)
    Best_score = float("inf")
    prev_best_score = float("inf")
    stagnation_counter = 0  # stagnation counter
    
    # Convergence curve
    Convergence_curve = np.zeros(Max_iter)
    
    # Solution object
    s = solution()
    
    print('TOA (Tardigrade Optimization Algorithm) is optimizing "' + objf.__name__ + '"')
    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    # =====  Main Loop =====
    for l in range(Max_iter):
        
        # Adaptive parameters
        convergence_factor = l / Max_iter  # convergence factor [0,1]
        exploration_factor = 2 * (1 - convergence_factor)  # decreases with time
        exploitation_factor = 2 * convergence_factor  # increases with time
        
        # ===== Evaluation and Update =====
        fitness_values = np.zeros(SearchAgents_no)
        
        for i in range(SearchAgents_no):
            # Ensure bounds
            for j in range(dim):
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])
            
            # Calculate fitness
            try:
                fitness = objf(Positions[i, :])
                
                # Ensure fitness is always a scalar
                if hasattr(fitness, '__len__') and not isinstance(fitness, str):
                    fitness = float(fitness.flatten()[0]) if hasattr(fitness, 'flatten') else float(fitness[0])
                else:
                    fitness = float(fitness)
                    
                fitness_values[i] = fitness
                
                # Update personal best
                if fitness < personal_best_score[i]:
                    personal_best_score[i] = fitness
                    personal_best[i] = Positions[i, :].copy()
                
                # Update global best
                if fitness < Best_score:
                    Best_score = fitness
                    Best_pos = Positions[i, :].copy()
                    
            except Exception as e:
                print(f"Error calculating fitness for agent {i}: {e}")
                fitness_values[i] = float("inf")
        
        # Ensure Best_score is always scalar
        Best_score = safe_float(Best_score)
        
        # Stagnation detection and escape mechanism
        if abs(Best_score - prev_best_score) < 1e-10:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        prev_best_score = Best_score
        
        # Escape mechanism from Local Minima
        if stagnation_counter > 10:
            print(f"Applying escape mechanism at iteration {l}")
            worst_agents = np.argsort(fitness_values)[-SearchAgents_no//4:]  # worst 25%
            for idx in worst_agents:
                for j in range(dim):
                    if random.random() < 0.5:
                        Positions[idx, j] = random.uniform(lb[j], ub[j])  # random repositioning
            stagnation_counter = 0
        
        # ===== Update biological parameters =====
        elite_count = int(SearchAgents_no * elite_ratio)
        elite_indices = np.argsort(fitness_values)[:elite_count]
        
        for i in range(SearchAgents_no):
            
            # Update search time adaptively
            if i in elite_indices:
                alpha_i = beta_A2 * (0.5 + 0.5 * convergence_factor)  # elite slower
            else:
                alpha_i = beta_A2 * (1 + exploration_factor)  # others faster
            search_times[i] = max(0.5, 3600 / alpha_i)
            
            # Update hunger level intelligently
            if fitness_values[i] < np.mean(fitness_values):  # good performance
                hunger_levels[i] = max(0.1, hunger_levels[i] - 0.15)
            else:  # poor performance
                hunger_levels[i] = min(1.0, hunger_levels[i] + 0.1)
            
            #  optimization model
            t = max(0.001, convergence_factor)
            y_t = 6.9349 * t / (1 + 0.0754 * t)
            ts_normalized = search_times[i] / 3600
            optimization_factor = y_t / (t + ts_normalized)
            
            # =====  Position Update =====
            for j in range(dim):
                
                r1, r2 = random.random(), random.random()
                old_pos = Positions[i, j]
                
                # Behavior 1: Intensive Exploitation - 40%
                if r1 < 0.4:
                    if i in elite_indices:
                        # Elite: precise local search around personal best
                        movement = (random.gauss(0, 1) * (ub[j] - lb[j]) * 0.02 * 
                                  (1 - convergence_factor) * hunger_levels[i])
                        Positions[i, j] = personal_best[i, j] + movement
                    else:
                        # Non-elite: movement toward global best
                        movement = (hunger_levels[i] * random.uniform(-1, 1) * 
                                  (ub[j] - lb[j]) * 0.05 * optimization_factor * exploitation_factor)
                        Positions[i, j] = Best_pos[j] + movement
                
                # Behavior 2: Smart Exploration - 35%
                elif r1 < 0.75:
                    if random.random() < levy_probability:
                        # Levy flight for long-range exploration
                        levy_step = np.random.standard_cauchy() * (ub[j] - lb[j]) * 0.01
                        Positions[i, j] = Positions[i, j] + levy_step
                    else:
                        #  traditional exploration
                        exploration_range = 0.1 * exploration_factor * (1 + search_times[i] / 1000)
                        movement = (random.uniform(-1, 1) * (ub[j] - lb[j]) * exploration_range * 0.1)
                        Positions[i, j] = Positions[i, j] + movement
                
                # Behavior 3:  Cryptobiosis or Cooperation - 25%
                else:
                    if hunger_levels[i] > cryptobiosis_threshold:
                        #  cryptobiosis state
                        if random.random() < 0.3:  # chance to move
                            movement = (random.uniform(-0.005, 0.005) * (ub[j] - lb[j]))
                            Positions[i, j] = Positions[i, j] + movement
                        # else stay in place (complete freezing)
                    else:
                        # Intelligent cooperation with best solutions
                        if r2 < 0.5:
                            # Cooperation with global best
                            cooperation_factor = (random.uniform(0, 1) * hunger_levels[i] * 0.2)
                            movement = (Best_pos[j] - Positions[i, j]) * cooperation_factor
                        else:
                            # Cooperation with random elite agent
                            if len(elite_indices) > 0:
                                elite_agent = random.choice(elite_indices)
                                cooperation_factor = (random.uniform(-0.5, 0.5) * hunger_levels[i] * 0.15)
                                movement = (Positions[elite_agent, j] - Positions[i, j]) * cooperation_factor
                            else:
                                movement = 0
                        Positions[i, j] = Positions[i, j] + movement
                
                # Apply velocity for continuity
                velocity[i, j] = 0.7 * velocity[i, j] + 0.3 * (Positions[i, j] - old_pos)
                Positions[i, j] = Positions[i, j] + 0.1 * velocity[i, j]
                
                # Ensure bounds
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])
        
        # ===== Local Search for Elite =====
        if random.random() < local_search_probability:
            for idx in elite_indices[:3]:  # best 3 agents
                for j in range(dim):
                    # Precise local improvement
                    for _ in range(2):  # two attempts
                        candidate_pos = Positions[idx].copy()
                        candidate_pos[j] += random.gauss(0, 1) * (ub[j] - lb[j]) * 0.001
                        candidate_pos[j] = np.clip(candidate_pos[j], lb[j], ub[j])
                        
                        try:
                            candidate_fitness = objf(candidate_pos)
                            
                            # Ensure candidate_fitness is scalar
                            if hasattr(candidate_fitness, '__len__') and not isinstance(candidate_fitness, str):
                                candidate_fitness = float(candidate_fitness.flatten()[0]) if hasattr(candidate_fitness, 'flatten') else float(candidate_fitness[0])
                            else:
                                candidate_fitness = float(candidate_fitness)
                            
                            if candidate_fitness < fitness_values[idx]:
                                Positions[idx] = candidate_pos
                                fitness_values[idx] = candidate_fitness
                                if candidate_fitness < Best_score:
                                    Best_score = candidate_fitness
                                    Best_pos = candidate_pos.copy()
                                break
                        except:
                            continue
        
        # =====  Satiation Model =====
        satiation_factor = max(0.005, 0.5 / (1 + convergence_factor * 2))
        for i in elite_indices:  # apply only to elite
            if random.random() < satiation_factor:
                for j in range(dim):
                    # Very precise movements for final optimization
                    fine_movement = (random.gauss(0, 1) * (ub[j] - lb[j]) * 0.002 * 
                                   (1 - convergence_factor)**2)
                    candidate_pos = Best_pos[j] + fine_movement
                    candidate_pos = np.clip(candidate_pos, lb[j], ub[j])
                    
                    # Test improvement
                    test_position = Positions[i].copy()
                    test_position[j] = candidate_pos
                    try:
                        test_fitness = objf(test_position)
                        
                        # Ensure test_fitness is scalar
                        if hasattr(test_fitness, '__len__') and not isinstance(test_fitness, str):
                            test_fitness = float(test_fitness.flatten()[0]) if hasattr(test_fitness, 'flatten') else float(test_fitness[0])
                        else:
                            test_fitness = float(test_fitness)
                        
                        if test_fitness < fitness_values[i]:
                            Positions[i, j] = candidate_pos
                    except:
                        pass
        
        # Save best result (ensure it's scalar)
        Convergence_curve[l] = safe_float(Best_score)
        
        #  progress printing with safe formatting
        if l % 10 == 0:
            active_agents = sum(1 for h in hunger_levels if h < cryptobiosis_threshold)
            elite_fitness = np.mean([fitness_values[i] for i in elite_indices]) if len(elite_indices) > 0 else 0
            print(f"Iteration {l}: Best = {safe_float(Best_score):.6f}, Elite Avg = {safe_float(elite_fitness):.6f}, "
                  f"Active = {active_agents}, Stagnation = {stagnation_counter}")
    
    # ===== Completion =====
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "TOA"
    s.objfname = objf.__name__
    s.bestIndividual = Best_pos
    s.best = safe_float(Best_score)  # Ensure scalar value
    
    # Final statistics with safe formatting
    final_active = sum(1 for h in hunger_levels if h < cryptobiosis_threshold)
    print(f"TOA completed: Best fitness = {safe_float(Best_score):.6f}")
    print(f"Final stats: Active agents = {final_active}, Total evaluations = {Max_iter * SearchAgents_no}")
    
    return s
