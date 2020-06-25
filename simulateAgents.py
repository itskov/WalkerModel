import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import poisson
from itertools import product
import matplotlib.pyplot as plt
import pandas as pd



from RandomWalker import RandomWalker


def simulate_prior_vs_no_prior(lam_ratio, basal_speed):
    GRID_SIZE = 1000
    EXPERIMENT_STEPS = 50000
    PROXIMITY = GRID_SIZE / 10

    # Creating the grid
    grid = np.zeros((GRID_SIZE, GRID_SIZE))

    # Sampling food patches.
    food_patches_num = np.random.poisson(GRID_SIZE * lam_ratio)


    # Food patches coordinates
    # Ordinary
    patches_x = np.random.uniform(0, GRID_SIZE, food_patches_num)
    patches_y = np.random.uniform(0, GRID_SIZE, food_patches_num)
    food_patches_ord = np.array([patches_x, patches_y]).T

    # Prior
    patches_x = np.random.uniform(0, GRID_SIZE, food_patches_num)
    patches_y = np.random.uniform(0, GRID_SIZE, food_patches_num)
    food_patches_prior = np.array([patches_x, patches_y]).T


    ord_walker = RandomWalker((500, 500), 1 * 0.5, GRID_SIZE, GRID_SIZE)
    prior_walker = RandomWalker((500, 500), 1 * 0.5, GRID_SIZE, GRID_SIZE)


    ord_found_num = 0
    prior_found_num = 0
    ord_traj = 0
    prior_traj = 0

    for i in range(1, EXPERIMENT_STEPS):
        ord_walker.step()
        prior_walker.step()

        ord_distances = cdist(ord_walker.current_cords(), food_patches_ord)
        prior_distances = cdist(prior_walker.current_cords(), food_patches_prior)

        found_patches_ord = ord_distances < PROXIMITY
        found_patches_prior = prior_distances < PROXIMITY

        # Removing visited patches.
        food_patches_ord[np.ravel(found_patches_ord), :] = np.Inf
        food_patches_prior[np.ravel(found_patches_prior), :] = np.Inf

        ord_found_num += np.sum(found_patches_ord)
        prior_found_num += np.sum(found_patches_prior)

        # Reevaluate prior speed.
        estimated_rate = (prior_found_num / i) * np.floor(GRID_SIZE**2 / i)
        prior_for_food = 1 - poisson.cdf(prior_found_num, estimated_rate)

        #prior_walker._speed = np.max((basal_speed, prior_for_food))
        prior_walker._speed = basal_speed + (1 - basal_speed) * prior_for_food

        ord_traj += ord_walker._speed
        prior_traj += prior_walker._speed

    ord_efficiency = ord_found_num / ord_traj
    prior_efficiency = prior_found_num / prior_traj
    print("Ordinary agent. Found %d, ratio %f, Mean Speed: %f" % (ord_found_num, ord_efficiency, ord_traj / EXPERIMENT_STEPS))
    print("Prior agent. Found %d, ratio %f, Mean Speed: %f" % (prior_found_num, prior_efficiency, prior_traj / EXPERIMENT_STEPS))

    return(ord_found_num , ord_efficiency, prior_found_num, prior_efficiency)

def batch_run(count, lam, basal_speed, plot=False):
    df = pd.DataFrame({'Count' : [], 'Efficiency' : [], 'Type' : [], 'Lambda' : [], 'BasalSpeed' : []})

    all_count_ord = []
    all_eff_ord = []
    all_count_prior = []
    all_eff_prior = []

    for i in range(count):
        (ord_found_num, ord_efficiency, prior_found_num, prior_efficiency) = simulate_prior_vs_no_prior(lam, basal_speed)

        all_count_ord.append(ord_found_num)
        all_eff_ord.append(ord_efficiency)
        all_count_prior.append(prior_found_num)
        all_eff_prior.append(prior_efficiency)

    df_ord = pd.DataFrame({'Count' : all_count_ord, 'Efficiency' : all_eff_ord, 'Type' : 'ord', 'Lambda' : lam, 'BasalSpeed' : basal_speed})
    df_prior = pd.DataFrame({'Count' : all_count_prior, 'Efficiency' : all_eff_prior, 'Type' : 'prior', 'Lambda' : lam, 'BasalSpeed' : basal_speed})

    df = pd.concat((df_ord, df_prior))

    return df



def bath_run_conc(comb):
    print('Starting. Lambda: %f. Basal Speed: %f' % (comb[0], comb[1]))
    df = batch_run(1, *comb)
    print('Done with: Lambda: %f. Basal Speed: %f' % (comb[0], comb[1]))
    return df



if __name__ == "__main__":
    from multiprocessing import Pool

    lams = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.2, 0.3, 0.4, 0.5]
    basal_speeds = [0.01, 0.03, 0.05, 0.07, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25]

    all_combs = product(lams, basal_speeds)

    p = Pool(4)

    all_dfs = p.map(bath_run_conc, all_combs)
    all_dfs = pd.concat(all_dfs)

    all_dfs.to_pickle('./ModelResults.pkl')












