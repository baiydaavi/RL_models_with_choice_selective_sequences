import random
import numpy as np
from tqdm import tqdm

from utils import gaussian, gaussian_normalised

def run_td_model(
        data_id="PL",
        activity_type="sequential",
        num_trials=50000,
        elig_time=0.8,
        discount_time=0.7,
        learning_rate=0.009,
        stay_bias=0.15,
        temperature=7000.0,
        frac_decision_neuron=21,
        is_stimulated=False,
        fraction_stimulated=0.7,
        stimulation_level=0.2,
        start_time=-3.0,
        end_time=3.0,
        step_time=0.01,
        synaptic_delay_time=0.01,
        reward_offset=0.2,
        seed=4,
        save_NAc_activity=False
):

    choice_activity = np.load(f"recorded_data/{data_id}_{activity_type}.npz")
    left_choice_activity = choice_activity["left_choice_activity"]
    right_choice_activity = choice_activity["right_choice_activity"]

    synaptic_delay_steps = int(synaptic_delay_time / step_time)

    times = np.arange(start_time, end_time, step_time)[1:None]

    time_steps = np.arange(0, len(times), 1)  # indexing the time array

    learning_rate = learning_rate  # learning rate

    # eligibility trace decay rate (0.9 = 0.6 s), (0.99 = 6s)
    lamb_decay = np.exp(-step_time / elig_time)

    discount_factor = np.exp(
        -synaptic_delay_time / discount_time
    )  # temporal discounting

    gauss_width = 0.3

    use_synaptic_eligibility = True  # do we use eligibility trace? - True,
    # False

    # background_pl = 0.05  #background activity for PL - any number
    # higher
    # than 0 turns it on
    # background activity for PL - any number higher than 0 turns
    # it on
    background_mean = 0.05
    # background activity for PL - any number higher than 0 turns
    # it on
    background_width = 0.05

    num_neurons = left_choice_activity.shape[1]

    num_left_neurons = num_right_neurons = num_neurons // 2

    stimulated_trials = np.zeros(num_trials)

    if is_stimulated:
        stimulated_trials = choose_stimulated_trials(stimulated_trials)

    # define the reward time for each trial (varies between 0-1 s from lever
    # press time)
    reward_start_time = int(np.where(times >= 0)[0][0])  # + 0.2/step_time
    reward_end_time = int(np.where(times >= 1)[0][0])  # +0.2/step_time

    np.random.seed(seed)
    peak_reward_times = [
        np.random.randint(reward_start_time, reward_end_time) for _ in
        range(num_trials)
    ]

    # define the reward time for the non-variable delay case (rewarded at 1 s
    # from lever press time)
    # peak_reward_times = [int(len(t)/2-1) for l in range(num_trials)]

    # given that lever press happens at 0 s define the probability for when
    # nose poke starts (mean start time is -1.5 s)

    nose_poke_time_distribution = gaussian(times, -2.5, 0.1) / np.sum(
        gaussian(times, -2.5, 0.1)
    )  # normalise the probability

    # find the index for the start time of nose poke given the probability
    # for various nose poke start time
    np.random.seed(seed)
    nose_poke_times = np.random.choice(
        time_steps, num_trials, p=nose_poke_time_distribution
    )

    # Initialise the PL-NAc weights

    right_weights = np.zeros(
        (num_trials, num_right_neurons)
    )  # weight matrix for right sequence
    left_weights = np.zeros(
        (num_trials, num_left_neurons)
    )  # weight matrix for left sequence

    # initialise the prediction error matrix
    RPEs = np.zeros((num_trials, len(times)))

    # initialise the vector telling which side gets rewarded on each trial -
    # 1: left, 0: right
    rewarded_sides = np.zeros(num_trials)
    rewarded_trials = np.zeros(num_trials)

    # initialise the vector telling which side is chosen on each trial - 1:
    # left, 0: right
    choices = np.zeros(num_trials)

    # initialise the matrix calculating the right action values function at
    # all time for each num_trials
    values = np.zeros((num_trials, len(times)))

    # initialise the right values function vector for each trial that is used
    # to make the choice(values function at nose poke)
    right_decision_value = np.zeros(num_trials)

    # initialise the left values function vector for each trial that is used
    # to make the choice(values function at nose poke)
    left_decision_value = np.zeros(num_trials)

    if save_NAc_activity:
        NAcs = np.zeros((num_trials, num_neurons, len(times)))

    # Trial 0 doesn't involve updating the values function. It's a proxy trial
    # for the code to use the initial values

    choices[0] = -1

    rewarded_sides[0] = -1

    # Initialize block parameters
    reward_count_in_blocks = 0
    high_prob_blocks = np.zeros(num_trials)

    high_prob_blocks[0] = np.random.randint(2)
    if high_prob_blocks[0] == 0:
        high_prob_blocks[0] = -1

    high_prob_blocks[0] = 1

    trial_count_in_blocks = 1

    num_nrn_list = list(np.arange(0, (num_left_neurons + num_right_neurons)))

    for current_trial in tqdm(range(1, num_trials)):

        # I_r/I_l = 1 if choice on last trial was right/left and 0 otherwise.
        I_r = int(-1 == choices[current_trial - 1])
        I_l = int(1 == choices[current_trial - 1])

        # update the decision vector (values function when nose poke happens)
        back_noise = np.abs(
            ((background_width * background_mean) / np.sqrt(step_time))
            * np.random.randn(num_left_neurons + num_right_neurons, len(times))
            + background_mean
        )

        # update the decision vector for right choice to use for next trial
        right_decision_value[current_trial] = np.matmul(
            right_weights[current_trial - 1,
            0: num_neurons // frac_decision_neuron],
            np.mean(
                back_noise[
                num_left_neurons: num_left_neurons
                                  + num_neurons // frac_decision_neuron,
                0:5,
                ],
                axis=1,
            ),
        )
        # update the decision vector for left
        left_decision_value[current_trial] = np.matmul(
            left_weights[current_trial - 1,
            0: num_neurons // frac_decision_neuron],
            np.mean(back_noise[0: num_neurons // frac_decision_neuron, 0:5],
                    axis=1),
        )

        # calculate the probability to choose left or right
        P_right = np.exp(
            temperature * right_decision_value[current_trial] + stay_bias * I_r
        )
        P_left = np.exp(
            temperature * left_decision_value[current_trial] + stay_bias * I_l
        )

        # normalising the probabilities
        total_p = P_right + P_left
        P_right /= total_p
        P_left /= total_p

        # make the action given the prob
        if np.random.random() < P_left:
            choices[current_trial] = 1  # left choice
        else:
            choices[current_trial] = -1  # right choice

        # Determines if there should be a block switch
        if reward_count_in_blocks > 9 and np.random.randint(10) < 4:
            reward_count_in_blocks = 0
            high_prob_blocks[current_trial] = high_prob_blocks[
                                                  current_trial - 1] * -1
            trial_count_in_blocks = 1
        else:
            high_prob_blocks[current_trial] = high_prob_blocks[
                current_trial - 1]
            trial_count_in_blocks += 1

        # determine whether reward was received
        if choices[current_trial] == high_prob_blocks[current_trial]:
            if np.random.random() > 0.3:
                rewarded_trials[current_trial] = 1
                rewarded_sides[current_trial] = choices[current_trial]
                reward_count_in_blocks = reward_count_in_blocks + 1
            else:
                rewarded_sides[current_trial] = -choices[current_trial]
        else:
            if np.random.random() > 0.9:
                rewarded_trials[current_trial] = 1
                rewarded_sides[current_trial] = choices[current_trial]
                reward_count_in_blocks = reward_count_in_blocks + 1
            else:
                rewarded_sides[current_trial] = -choices[current_trial]

        if rewarded_trials[current_trial] == 1:
            rewards = gaussian_normalised(
                times,
                times[peak_reward_times[current_trial]] + reward_offset,
                gauss_width,
            )
            rewards[times < times[peak_reward_times[current_trial]]] = 0
        else:
            rewards = np.zeros(times.shape)

        if choices[current_trial] == 1:  # left choice
            current_trial_activity = left_choice_activity[
                                     np.random.randint(
                                         left_choice_activity.shape[0]), :, :
                                     ].copy()
        else:
            current_trial_activity = right_choice_activity[
                                     np.random.randint(
                                         right_choice_activity.shape[0]), :, :
                                     ].copy()

        if is_stimulated:
            chr_nrn_list = random.sample(
                num_nrn_list.copy(),
                int(fraction_stimulated * (
                            num_left_neurons + num_right_neurons)),
            )
            chr_nrn_list = np.sort(chr_nrn_list)

            if stimulated_trials[current_trial] == 1:
                current_trial_activity[chr_nrn_list, :] = stimulation_level

        eligibility_traces = np.zeros(
            (num_right_neurons + num_left_neurons, len(times))
        )

        # define a temporary vector to update weights through time for a
        # given trial
        temporary_weights = np.concatenate(
            [left_weights[current_trial - 1, :],
             right_weights[current_trial - 1, :]]
        )

        for current_time_step in np.arange(
                nose_poke_times[current_trial], len(times) - 1
        ):
            eligibility_traces[:, current_time_step] = (
                    use_synaptic_eligibility
                    * eligibility_traces[:, current_time_step - 1]
                    * lamb_decay
                    + step_time * current_trial_activity[:, current_time_step]
            )

            # calculate the right values function at the current time step
            values[current_trial, current_time_step] = np.matmul(
                temporary_weights, current_trial_activity[:, current_time_step]
            )

            if save_NAc_activity:
                # calculate right choice NAc firing for time step j using the
                # updated weights
                NAcs[current_trial, :, current_time_step] = np.multiply(
                    temporary_weights,
                    current_trial_activity[:, current_time_step])

            # Calculate prediction error
            RPE = (
                    rewards[current_time_step]
                    + (
                            discount_factor * values[
                        current_trial, current_time_step]
                            - values[
                                current_trial, current_time_step -
                                synaptic_delay_steps]
                    )
                    / synaptic_delay_time
            )
            RPEs[current_trial, current_time_step] = RPE

            # update the weights for the current time step
            temporary_weights = (
                    temporary_weights
                    + step_time
                    * learning_rate
                    * RPE
                    * eligibility_traces[:, current_time_step]
            )
            temporary_weights[temporary_weights < 0] = 0

        # keep the final updated weights to use for the next trial
        left_weights[current_trial, :] = temporary_weights[0:num_left_neurons]
        right_weights[current_trial, :] = temporary_weights[
                                          num_left_neurons:None]

    return {
        "times": times,
        "high_prob_blocks": high_prob_blocks,
        "choices": choices,
        "rewarded_sides": rewarded_sides,
        "rewarded_trials": rewarded_trials,
        "RPEs": RPEs,
        "values": values,
        "stimulated_trials": stimulated_trials,
        "right_decision_value": right_decision_value,
        "left_decision_value": left_decision_value,
        "peak_reward_times": peak_reward_times,
        "step_time": step_time,
        "NAc_activity": NAcs if save_NAc_activity else 'not_available'
    }


def choose_stimulated_trials(stimulated_trials, fraction_trials_stimulated=0.1):
    stimulated_trials[
    0: int(fraction_trials_stimulated * len(stimulated_trials))] = 1
    stimulated_trials = np.random.permutation(stimulated_trials)
    return stimulated_trials
