import glob
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

from utils import softmax


class GenerateTestPlots:
    # Initializing the class
    def __init__(
            self,
            testing_data_path="testing_data/normal/",
            num_actions=3,
            num_states=42,
            num_hidden_units=128,
            gamma=0.96,
    ):

        testing_files = [f for f in
                         glob.glob(testing_data_path + "*.npz", recursive=True)]

        self.current_trial_times = []
        self.trial_count_in_blocks = []
        self.actions = []
        self.choices = []
        self.actor_logits = []
        self.reward_probs = []
        self.rewards = []
        self.is_rewardeds = []
        self.values = []
        self.rpes = []
        self.critic_hs = []
        self.critic_cs = []
        self.actor_hs = []
        self.actor_cs = []
        self.is_stimulateds = []

        for file in tqdm.tqdm(testing_files):
            testing_data = np.load(file)
            self.current_trial_times.append(testing_data['current_trial_times'])
            self.trial_count_in_blocks.append(
                testing_data['trial_count_in_blocks'])
            self.actions.append(testing_data['actions'])
            self.choices.append(testing_data['choices'])
            self.actor_logits.append(np.float32(testing_data['actor_logits']))
            self.reward_probs.append(np.float32(testing_data['reward_probs']))
            self.rewards.append(np.float32(testing_data['rewards']))
            self.is_rewardeds.append(testing_data['is_rewardeds'])
            self.values.append(np.float32(testing_data['values']))
            self.rpes.append(np.float32(testing_data['rpes']))
            self.critic_hs.append(np.float32(testing_data['critic_hs']))
            self.critic_cs.append(np.float32(testing_data['critic_cs']))
            self.actor_hs.append(np.float32(testing_data['actor_hs']))
            self.actor_cs.append(np.float32(testing_data['actor_cs']))

        self.current_trial_times = np.array(self.current_trial_times).reshape(
            -1)
        self.trial_count_in_blocks = np.array(
            self.trial_count_in_blocks).reshape(-1)
        self.actions = np.array(self.actions).reshape(-1)
        self.choices = np.array(self.choices).reshape(-1)
        self.actor_logits = np.array(self.actor_logits).reshape(-1, num_actions)
        self.reward_probs = np.array(self.reward_probs).reshape(-1)
        self.rewards = np.array(self.rewards).reshape(-1)
        self.is_rewardeds = np.array(self.is_rewardeds).reshape(-1)
        self.values = np.array(self.values).reshape(-1)
        self.rpes = np.array(self.rpes).reshape(-1)
        self.critic_hs = np.array(self.critic_hs).reshape(-1, num_hidden_units)
        self.critic_cs = np.array(self.critic_cs).reshape(-1, num_hidden_units)
        self.actor_hs = np.array(self.actor_hs).reshape(-1, num_hidden_units)
        self.actor_cs = np.array(self.actor_cs).reshape(-1, num_hidden_units)
        self.is_stimulateds = np.array(self.is_stimulateds).reshape(-1)
        self.num_states = num_states
        self.num_hidden_units = num_hidden_units
        self.gamma = gamma

    ####################################
    # pca activity plot

    ####################################
    # Behavior plot

    def behavior(
            self, start_num_trials=500, num_tr=200, plt_val=None,
            save=None
    ):

        print(
            f"reward rate = "
            f"{np.mean(self.is_rewardeds[self.current_trial_times == 1])}")

        end_num_trials = start_num_trials + num_tr

        high_prob = self.reward_probs[self.current_trial_times == 2]
        rewarded_trials = self.is_rewardeds[self.current_trial_times == 2]
        chosen_side = 2 * self.choices[self.current_trial_times == 2] - 1
        values = np.mean(
            np.array(
                [
                    self.values[
                        self.current_trial_times == (i + 1) % self.num_states]
                    for i in range(self.num_states)
                ]
            ),
            axis=0,
        )

        rew_side = chosen_side * rewarded_trials

        max_val = np.max(values)
        min_val = np.min(values[1:-1])
        mean_val = (max_val + min_val) / 2

        plt.figure(figsize=(16, 5))

        plt.scatter(
            np.arange(start_num_trials, end_num_trials),
            high_prob[start_num_trials:end_num_trials] * (
                    max_val - mean_val) * 1.5
            + mean_val,
            s=40,
            color="black",
            label="high probability side",
        )
        plt.scatter(
            np.arange(start_num_trials, end_num_trials),
            chosen_side[start_num_trials:end_num_trials] * (
                    max_val - mean_val) * 1.3
            + mean_val,
            s=20,
            color="blue",
            label="chosen side",
        )
        plt.scatter(
            np.arange(start_num_trials, end_num_trials),
            rew_side[start_num_trials:end_num_trials] * (
                    max_val - mean_val) * 1.1
            + mean_val,
            s=30 * np.abs(rew_side[start_num_trials:end_num_trials]),
            color="green",
            label="rewarded side",
        )

        if plt_val:
            plt.plot(
                np.arange(start_num_trials, end_num_trials),
                values[start_num_trials:end_num_trials],
                color="aqua",
                label="value averaged over time",
            )

        fontP = FontProperties()
        fontP.set_size("xx-large")
        plt.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            ncol=3,
            mode="expand",
            borderaxespad=0.0,
            prop=fontP,
        )
        plt.xlabel("Trial number", fontsize=20)
        # plt.savefig('behav.pdf', bbox_inches='tight')
        # plt.yticks([])
        for i in np.where(np.diff(high_prob))[0] + 1:
            if start_num_trials <= i < end_num_trials:
                plt.axvline(x=i - 0.5, linestyle="dotted", color="k")

        if save:
            plt.savefig(save, bbox_inches="tight")

    ####################################
    # Stay probability plot

    def stay_prob(self, plt_adap=None, save=None):

        prob_rew_return = 0.0
        prob_unrew_return = 0.0

        high_prob = self.reward_probs[self.current_trial_times == 2]
        rewarded_trials = self.is_rewardeds[self.current_trial_times == 2]
        chosen_side = 2 * self.choices[self.current_trial_times == 2] - 1

        switch_high = np.where(np.diff(high_prob) != 0)[0] + 1
        unrewarded_trials = (rewarded_trials - 1) * -1
        next_choice = np.roll(chosen_side, -1)
        return_rew = rewarded_trials * ((next_choice == chosen_side) * 1)
        prob_rew_return += np.sum(return_rew) / np.sum(rewarded_trials)
        return_unrew = unrewarded_trials * ((next_choice == chosen_side) * 1)
        prob_unrew_return += np.sum(return_unrew) / np.sum(unrewarded_trials)

        early_trials_5 = []
        late_trials_5 = []
        for i in range(len(switch_high) - 2):
            for j in range(5):
                early_trials_5.append(switch_high[i] + j)
                late_trials_5.append(switch_high[i + 1] + j - 5)

        next_choice = np.roll(chosen_side, -1)
        early_return_rew = rewarded_trials[early_trials_5] * (
                (next_choice[early_trials_5] == chosen_side[early_trials_5]) * 1
        )
        prob_early_return_rew = np.sum(early_return_rew) / np.sum(
            rewarded_trials[early_trials_5]
        )
        early_return_unrew = unrewarded_trials[early_trials_5] * (
                (next_choice[early_trials_5] == chosen_side[early_trials_5]) * 1
        )
        prob_early_return_unrew = np.sum(early_return_unrew) / np.sum(
            unrewarded_trials[early_trials_5]
        )

        late_return_rew = rewarded_trials[late_trials_5] * (
                (next_choice[late_trials_5] == chosen_side[late_trials_5]) * 1
        )
        prob_late_return_rew = np.sum(late_return_rew) / np.sum(
            rewarded_trials[late_trials_5]
        )
        late_return_unrew = unrewarded_trials[late_trials_5] * (
                (next_choice[late_trials_5] == chosen_side[late_trials_5]) * 1
        )
        prob_late_return_unrew = np.sum(late_return_unrew) / np.sum(
            unrewarded_trials[late_trials_5]
        )

        # fig = plt.figure(figsize=(6, 8))
        plt.subplot(1, 2, 1)
        plt.bar(
            [1, 1.5],
            [prob_rew_return, prob_unrew_return],
            width=0.2,
            color=["green", "red"],
        )
        plt.xticks([1, 1.5],
                   ["previously\n rewarded", "previously\n unrewarded"])
        plt.ylabel("probability of return")
        plt.ylim([0.0, 1.1])

        if plt_adap:
            plt.subplot(1, 2, 2)
            plt.bar(
                [1, 1.9],
                [prob_early_return_rew, prob_early_return_unrew],
                width=0.2,
                edgecolor="grey",
                color="grey",
                label="First 5",
            )
            plt.bar(
                [1.3, 2.2],
                [prob_late_return_rew, prob_late_return_unrew],
                width=0.2,
                edgecolor="grey",
                color="white",
                label="second 5",
            )
            plt.xticks(
                [1.15, 2.05],
                ["previously\n rewarded", "previously\n unrewarded"]
            )
            plt.ylabel("stay probability")
            plt.ylim([0, 1.1])
            plt.legend()

        if save:
            plt.savefig(save, bbox_inches="tight")

    ####################################
    # reward regression plot

    def choice_reg(self, stim=None, trials_back=11, save=None):

        if stim:

            rewarded_trials = self.is_rewardeds[
                (self.current_trial_times == 2) & (self.choices != -1)
                ]
            chosen_side = (
                    2 * self.choices[
                (self.current_trial_times == 2) & (self.choices != -1)] - 1
            )
            stim_trials = self.is_stimulateds[
                (self.current_trial_times == 2) & (self.choices != -1)
                ]

            reward_mat = np.zeros((trials_back - 1, len(rewarded_trials)))
            reward_vect = chosen_side * rewarded_trials
            for i in np.arange(1, trials_back):
                reward_mat[i - 1, :] = np.roll(reward_vect, i)

            # makes unreward matrix
            unrewarded_trials = np.abs(rewarded_trials - 1)
            unreward_mat = np.zeros((trials_back - 1, len(unrewarded_trials)))
            unreward_vec = chosen_side * unrewarded_trials
            for i in np.arange(1, trials_back):
                unreward_mat[i - 1, :] = np.roll(unreward_vec, i)

            # makes laser matrix
            laser_mat = np.zeros((trials_back - 1, len(rewarded_trials)))
            for i in np.arange(1, trials_back):
                laser_mat[i - 1, :] = np.roll(stim_trials, i)

            y = chosen_side
            x = np.concatenate(
                (
                    np.ones([1, len(y)]),
                    reward_mat,
                    unreward_mat,
                    reward_mat * laser_mat,
                    unreward_mat * laser_mat,
                    laser_mat,
                ),
                axis=0,
            )

            y_new = np.asarray((y + 1) / 2, dtype=int)

            log_reg = sm.Logit(y_new, x.T).fit()

            # Plots regression
            reward_coefs = log_reg.params[1:trials_back]
            unreward_coefs = log_reg.params[
                             trials_back: int(trials_back * 2 - 1)]
            rewlaser_coefs = log_reg.params[
                             int(trials_back * 2 - 1): int(trials_back * 3 - 2)
                             ]
            norewlaser_coefs = log_reg.params[
                               int(trials_back * 3 - 2): int(
                                   trials_back * 4 - 3)
                               ]
            # laser_coefs = log_reg.params[
            #               int(trials_back * 4 - 3): int(trials_back * 5 - 4)
            #               ]

            fig = plt.figure(figsize=(12, 6))
            plt.plot(reward_coefs, "b", label="rewarded trials no stimulation")
            plt.plot(unreward_coefs, "r",
                     label="unrewarded trials no stimulation")
            plt.plot(
                reward_coefs + rewlaser_coefs,
                linestyle="dotted",
                color="b",
                label="rewarded trials with stimulation",
            )
            plt.plot(
                unreward_coefs + norewlaser_coefs,
                linestyle="dotted",
                color="r",
                label="unrewarded trials with stimulation",
            )
            # plt.plot(laser_coefs,'k')
            plt.axhline(y=0, linestyle="dotted", color="gray")
            plt.xticks(
                np.arange(0, trials_back - 1, 2),
                [str(i) for i in np.arange(-1, -trials_back, -2)],
            )
            plt.xlabel("trials back")
            plt.ylabel("regression coefficients")
            plt.legend()

        else:

            rewarded_trials = self.is_rewardeds[
                (self.current_trial_times == 2) & (self.choices != -1)
                ]
            chosen_side = (
                    2 * self.choices[
                (self.current_trial_times == 2) & (self.choices != -1)] - 1
            )

            reward_mat = np.zeros((trials_back - 1, len(rewarded_trials)))
            reward_vect = chosen_side * rewarded_trials
            for i in np.arange(1, trials_back):
                reward_mat[i - 1, :] = np.roll(reward_vect, i)

            # makes unreward matrix
            unrewarded_trials = np.abs(rewarded_trials - 1)
            unreward_mat = np.zeros((trials_back - 1, len(unrewarded_trials)))
            unreward_vec = chosen_side * unrewarded_trials
            for i in np.arange(1, trials_back):
                unreward_mat[i - 1, :] = np.roll(unreward_vec, i)

            y = chosen_side
            x = np.concatenate((np.ones([1, len(y)]), reward_mat, unreward_mat),
                               axis=0)

            y_new = np.asarray((y + 1) / 2, dtype=int)

            log_reg = sm.Logit(y_new, x.T).fit()

            fig = plt.figure(figsize=(10, 3))
            plt.plot(
                np.arange(0, trials_back - 1),
                log_reg.params[1:trials_back],
                color="blue",
                linewidth=1,
                label="model - rewarded trials",
            )
            plt.plot(
                np.arange(0, trials_back - 1),
                log_reg.params[trials_back:None],
                color="red",
                linewidth=1,
                label="model - unrewarded trials",
            )
            plt.axhline(y=0, linestyle="dotted", color="gray")
            plt.legend()
            plt.xticks(
                np.arange(0, trials_back - 1, 2),
                [str(i) for i in np.arange(-1, -trials_back, -2)],
            )
            plt.xlabel("trials back")
            plt.ylabel("regression coefficients")

        if save:
            plt.savefig(save, bbox_inches="tight")

    ####################################
    # value plot

    def block_value_plot(self, save=None):

        high_prob = self.reward_probs[(self.current_trial_times == 2)]
        rewarded_trials = self.is_rewardeds[(self.current_trial_times == 2)]
        chosen_side = 2 * self.choices[(self.current_trial_times == 2)] - 1
        value = np.array(
            [
                self.values[
                    (self.current_trial_times == (i + 1) % self.num_states)]
                for i in range(self.num_states)
            ]
        ).T

        switch_high = np.where(np.diff(high_prob) != 0)[0] + 1

        early_trials = switch_high[0: len(switch_high) - 2]
        late_trials = early_trials + 14  # switch_high[1:-1] - 1
        # middle_trials=(early_trials+late_trials)/2
        middle_trials = early_trials + 4

        print(
            f"reward rate on last trial of a block = "
            f"{np.round(np.mean(rewarded_trials[early_trials - 1]), 2)}"
        )
        print(
            f"reward rate on first trial of a block = "
            f"{np.round(np.mean(rewarded_trials[early_trials]), 2)}"
        )
        # block identity of each trial
        block_iden = high_prob[early_trials]

        # indeces for block and actual choice
        left_left_early = early_trials[
            (block_iden == 1) & (chosen_side[early_trials] == 1)
            ]
        left_right_early = early_trials[
            (block_iden == 1) & (chosen_side[early_trials] == -1)
            ]
        right_right_early = early_trials[
            (block_iden == -1) & (chosen_side[early_trials] == -1)
            ]
        right_left_early = early_trials[
            (block_iden == -1) & (chosen_side[early_trials] == 1)
            ]

        left_left_middle = middle_trials[
            (block_iden == 1) & (chosen_side[middle_trials] == 1)
            ]
        left_right_middle = middle_trials[
            (block_iden == 1) & (chosen_side[middle_trials] == -1)
            ]
        right_right_middle = middle_trials[
            (block_iden == -1) & (chosen_side[middle_trials] == -1)
            ]
        right_left_middle = middle_trials[
            (block_iden == -1) & (chosen_side[middle_trials] == 1)
            ]

        left_left_late = late_trials[
            (block_iden == 1) & (chosen_side[late_trials] == 1)
            ]
        left_right_late = late_trials[
            (block_iden == 1) & (chosen_side[late_trials] == -1)
            ]
        right_right_late = late_trials[
            (block_iden == -1) & (chosen_side[late_trials] == -1)
            ]
        right_left_late = late_trials[
            (block_iden == -1) & (chosen_side[late_trials] == 1)
            ]

        left_left_early_val = value[left_left_early, :]
        left_right_early_val = value[left_right_early, :]
        right_right_early_val = value[right_right_early, :]
        right_left_early_val = value[right_left_early, :]

        left_left_middle_val = value[left_left_middle, :]
        left_right_middle_val = value[left_right_middle, :]
        right_right_middle_val = value[right_right_middle, :]
        right_left_middle_val = value[right_left_middle, :]

        left_left_late_val = value[left_left_late, :]
        left_right_late_val = value[left_right_late, :]
        right_right_late_val = value[right_right_late, :]
        right_left_late_val = value[right_left_late, :]

        fig = plt.figure(figsize=(8, 8))

        y_lim_min = -0.5
        y_lim_max = np.max(value) - 2

        t = np.arange(-2.1, 2.1, 0.1)  # range(self.num_states+1)

        plt.subplot(3, 2, 1)
        plt.title("Left block")
        val_trace_l = np.mean(left_left_early_val, axis=0)
        val_trace_r = np.mean(left_right_early_val, axis=0)
        val_sem_l = stats.sem(left_left_early_val, axis=0)
        val_sem_r = stats.sem(left_right_early_val, axis=0)
        plt.errorbar(
            t, val_trace_l, val_sem_l, color="blue", ecolor="skyblue",
            linewidth=1.0
        )
        plt.errorbar(
            t, val_trace_r, val_sem_r, color="green", ecolor="lime",
            linewidth=1.0
        )
        plt.ylim([y_lim_min, y_lim_max])

        plt.subplot(3, 2, 2)
        plt.title("Right block")
        val_trace_l = np.mean(right_left_early_val, axis=0)
        val_trace_r = np.mean(right_right_early_val, axis=0)
        val_sem_l = stats.sem(right_left_early_val, axis=0)
        val_sem_r = stats.sem(right_right_early_val, axis=0)
        plt.errorbar(
            t, val_trace_l, val_sem_l, color="blue", ecolor="skyblue",
            linewidth=1.0
        )
        plt.errorbar(
            t, val_trace_r, val_sem_r, color="green", ecolor="lime",
            linewidth=1.0
        )
        plt.ylim([y_lim_min, y_lim_max])
        plt.text(4, y_lim_max / 2, "1st trial", fontsize=15, color="k")

        plt.subplot(3, 2, 3)
        val_trace_l = np.mean(left_left_middle_val, axis=0)
        val_trace_r = np.mean(left_right_middle_val, axis=0)
        val_sem_l = stats.sem(left_left_middle_val, axis=0)
        val_sem_r = stats.sem(left_right_middle_val, axis=0)
        plt.errorbar(
            t, val_trace_l, val_sem_l, color="blue", ecolor="skyblue",
            linewidth=1.0
        )
        plt.errorbar(
            t, val_trace_r, val_sem_r, color="green", ecolor="lime",
            linewidth=1.0
        )
        plt.ylim([y_lim_min, y_lim_max])
        plt.ylabel("Value", fontsize=15)

        plt.subplot(3, 2, 4)
        val_trace_l = np.mean(right_left_middle_val, axis=0)
        val_trace_r = np.mean(right_right_middle_val, axis=0)
        val_sem_l = stats.sem(right_left_middle_val, axis=0)
        val_sem_r = stats.sem(right_right_middle_val, axis=0)
        plt.errorbar(
            t, val_trace_l, val_sem_l, color="blue", ecolor="skyblue",
            linewidth=1.0
        )
        plt.errorbar(
            t, val_trace_r, val_sem_r, color="green", ecolor="lime",
            linewidth=1.0
        )
        plt.ylim([y_lim_min, y_lim_max])
        plt.text(4, y_lim_max / 2, "5th trial", fontsize=15, color="k")

        plt.text(6, y_lim_max / 3, "Left Press", fontsize=15, color="blue")
        plt.text(6, y_lim_max / 9, "Right Press", fontsize=15, color="green")

        plt.subplot(3, 2, 5)
        val_trace_l = np.mean(left_left_late_val, axis=0)
        val_trace_r = np.mean(left_right_late_val, axis=0)
        val_sem_l = stats.sem(left_left_late_val, axis=0)
        val_sem_r = stats.sem(left_right_late_val, axis=0)
        plt.errorbar(
            t, val_trace_l, val_sem_l, color="blue", ecolor="skyblue",
            linewidth=1.0
        )
        plt.errorbar(
            t, val_trace_r, val_sem_r, color="green", ecolor="lime",
            linewidth=1.0
        )
        plt.ylim([y_lim_min, y_lim_max])

        plt.subplot(3, 2, 6)
        val_trace_l = np.mean(right_left_late_val, axis=0)
        val_trace_r = np.mean(right_right_late_val, axis=0)
        val_sem_l = stats.sem(right_left_late_val, axis=0)
        val_sem_r = stats.sem(right_right_late_val, axis=0)
        plt.errorbar(
            t, val_trace_l, val_sem_l, color="blue", ecolor="skyblue",
            linewidth=1.0
        )
        plt.errorbar(
            t, val_trace_r, val_sem_r, color="green", ecolor="lime",
            linewidth=1.0
        )
        plt.ylim([y_lim_min, y_lim_max])
        plt.text(4, y_lim_max / 2, "15th trial", fontsize=15, color="k")

        plt.text(-3, -2.5, "Time(s)", fontsize=15)

        if save:
            plt.savefig(save, bbox_inches="tight")

    def da_reg(self, plt_sec=None, trials_back=6, save=None):

        rewarded_trials = self.is_rewardeds[self.current_trial_times == 2]
        chosen_side = 2 * self.choices[self.current_trial_times == 2] - 1
        if plt_sec:
            y_vec_cs = np.mean(
                np.array(
                    [self.values[self.current_trial_times == i] for i in
                     range(2, 7)]),
                axis=0,
            )
        else:
            y_vec_cs = np.mean(
                np.array(
                    [
                        self.rpes[self.current_trial_times == i]
                        for i in
                        range(self.num_states // 2, self.num_states // 2 + 10)
                    ]
                ),
                axis=0,
            )

        # makes x matrix of reward identity on previous trials
        x_mat = np.zeros((trials_back, len(rewarded_trials)))
        for i in np.arange(0, trials_back):
            x_mat[i, :] = np.roll(rewarded_trials, i)

        y = np.reshape(y_vec_cs, [len(y_vec_cs), 1])
        x = np.concatenate((np.ones([1, len(y_vec_cs)]), x_mat), axis=0)

        results = sm.OLS(y, x.T).fit()

        plt.figure(figsize=(11, 4))
        if plt_sec:
            plt.suptitle(
                "Regressing DA activity at beginning against outcome",
                fontsize=20
            )
        else:
            plt.suptitle(
                "Regressing DA activity at reward time against outcome",
                fontsize=20
            )

        plt.scatter(np.arange(trials_back), results.params[1:None])
        plt.axhline(y=0, linestyle="dashed", color="k")
        plt.xlabel("trials back", fontsize=15)
        plt.ylabel("regression coefficients", fontsize=15)
        print(results.params[1:None])

        if save:
            plt.savefig(save, bbox_inches="tight")

    ####################################
    # block switch plot

    def block_switch(self, trial_back=10, save=None):

        high_prob = self.reward_probs[
            (self.current_trial_times == 2) & (self.choices != -1)]
        chosen_side = (
                2 * self.choices[
            (self.current_trial_times == 2) & (self.choices != -1)] - 1
        )

        switch_high = np.where(np.diff(high_prob) != 0)[0] + 1
        early_trials = switch_high[0: len(switch_high) - 1]
        block_iden = high_prob[early_trials]

        # finds times of left to right
        block_switch = early_trials[block_iden == -1] - 1

        time_window = np.arange(-trial_back, trial_back + 1)
        r_choice_mat = np.zeros([len(block_switch), len(time_window)])
        for i in np.arange(1, len(block_switch)):
            r_choice_mat[i, :] = chosen_side[time_window + block_switch[i]]
            r_choice_mat[i, :] = (r_choice_mat[i, :] + 1) / 2

        # same except right to left
        block_switch = early_trials[block_iden == 1] - 1

        time_window = np.arange(-trial_back, trial_back + 1)
        l_choice_mat = np.zeros([len(block_switch), len(time_window)])
        for i in np.arange(1, len(block_switch)):
            l_choice_mat[i, :] = chosen_side[time_window + block_switch[i]] * -1
            l_choice_mat[i, :] = (l_choice_mat[i, :] + 1) / 2

        final_choice_mat = np.concatenate([l_choice_mat, r_choice_mat], axis=0)

        plot_trace = np.mean(final_choice_mat, axis=0)
        sem_trace = stats.sem(final_choice_mat, axis=0)

        plot_trace2 = np.mean(-1 * (final_choice_mat - 1), axis=0)
        sem_trace2 = stats.sem(-1 * (final_choice_mat - 1), axis=0)

        ax = plt.figure(figsize=(8, 4)).gca()
        ax.axvline(x=0, linestyle="dotted", color="gray")
        ax.errorbar(time_window, plot_trace, sem_trace)
        ax.errorbar(time_window, plot_trace2, sem_trace2)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=10))
        ax.text(
            -10,
            1.05,
            "Blue - pre-switch \nhigh probability\nchoice  ",
            fontsize=15,
            color="dodgerblue",
        )
        ax.text(
            1,
            1.05,
            "Orange - post-switch \nhigh probability\nchoice  ",
            fontsize=15,
            color="orange",
        )
        plt.xlabel("Trials from block switch")
        plt.ylabel("p(choice)")
        # plt.axhline(y=plot_trace[time_window==1],color='k',linestyle='dotted')
        print(plot_trace[time_window == 1])

        if save:
            plt.savefig(save, bbox_inches="tight")

    ####################################
    # rpe rew vs unrew plot

    def rpe_plot(self, save=None):

        rpe_r = np.array(
            [
                np.mean(
                    self.rpes[
                        (self.current_trial_times == (i + 2) % self.num_states)
                        & (self.choices != -1)
                        & (self.is_rewardeds == 1)
                        ]
                )
                for i in range(self.num_states)
            ]
        )
        rpe_ur = np.array(
            [
                np.mean(
                    self.rpes[
                        (self.current_trial_times == (i + 2) % self.num_states)
                        & (self.choices != -1)
                        & (self.is_rewardeds == 0)
                        ]
                )
                for i in range(self.num_states)
            ]
        )
        val_r = np.mean(
            self.values[
                (self.current_trial_times == 1) & (self.choices != -1) & (
                        self.is_rewardeds == 1)
                ]
        )
        val_ur = np.mean(
            self.values[
                (self.current_trial_times == 1) & (self.choices != -1) & (
                        self.is_rewardeds == 0)
                ]
        )

        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        rew_rpe = np.concatenate([[0], [val_r], rpe_r])
        unrew_rpe = np.concatenate([[0], [val_ur], rpe_ur])
        plt.plot(np.arange(-2.2, 2.1, 0.1), rew_rpe, label="rewarded trials")
        plt.plot(np.arange(-2.2, 2.1, 0.1), unrew_rpe,
                 label="unrewarded trials")
        plt.xlabel("time", fontsize=20)
        plt.ylabel("reward prediction error", fontsize=20)
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(-2.1, 2.1, 0.1), rpe_r, label="rewarded trials")
        plt.plot(np.arange(-2.1, 2.1, 0.1), rpe_ur, label="unrewarded trials")
        plt.xlabel("time", fontsize=20)
        plt.ylabel("reward prediction error", fontsize=20)
        plt.legend()

        if save:
            plt.savefig(save, bbox_inches="tight")

    def prob_rpe(self, save=None):

        mean_rpe_r = np.zeros(30)
        mean_rpe_l = np.zeros(30)
        mean_logit_r = np.zeros(30)
        mean_logit_l = np.zeros(30)

        prob_vec = softmax(self.actor_logits)

        for i in range(1, 31):

            if i == 30:
                mean_logit_r[i - 1] = np.mean(
                    prob_vec[
                        (self.current_trial_times == 1)
                        & (self.reward_probs == -1)
                        & (self.trial_count_in_blocks == 1),
                        1,
                    ]
                )
                mean_logit_l[i - 1] = np.mean(
                    prob_vec[
                        (self.current_trial_times == 1)
                        & (self.reward_probs == 1)
                        & (self.trial_count_in_blocks == 1),
                        1,
                    ]
                )
            else:
                mean_logit_r[i - 1] = np.mean(
                    prob_vec[
                        (self.current_trial_times == 1)
                        & (self.reward_probs == 1)
                        & (self.trial_count_in_blocks == i + 1),
                        1,
                    ]
                )
                mean_logit_l[i - 1] = np.mean(
                    prob_vec[
                        (self.current_trial_times == 1)
                        & (self.reward_probs == -1)
                        & (self.trial_count_in_blocks == i + 1),
                        1,
                    ]
                )

            mean_rpe_r[i - 1] = np.mean(
                self.rpes[
                    (self.current_trial_times > self.num_states // 2)
                    & (self.reward_probs == 1)
                    & (self.trial_count_in_blocks == i)
                    ]
            )
            mean_rpe_l[i - 1] = np.mean(
                self.rpes[
                    (self.current_trial_times > self.num_states // 2)
                    & (self.reward_probs == -1)
                    & (self.trial_count_in_blocks == i)
                    ]
            )

        color_range = np.arange(1, 31)
        cbar_name = "Trials in a left to right block"

        plt.figure(figsize=(10, 10))
        plt.plot(
            mean_rpe_r,
            mean_logit_r,
            color="grey",
        )

        fig1 = plt.scatter(
            mean_rpe_r,
            mean_logit_r,
            c=color_range,
            cmap="cool",
            edgecolors="black",
            s=100,
        )
        plt.plot(
            mean_rpe_l,
            mean_logit_l,
            color="grey",
        )

        fig2 = plt.scatter(
            mean_rpe_l,
            mean_logit_l,
            c=color_range,
            cmap="hot",
            edgecolors="black",
            s=100,
        )
        cbar = plt.colorbar(fig1)
        cbar.set_label("Trials in a left block", fontsize=15)
        cbar2 = plt.colorbar(fig2)
        cbar2.set_label("Trials in a right block", fontsize=15)
        plt.ylabel("prob(right)", fontsize=20)
        plt.xlabel("RPE", fontsize=20)
        plt.title("Choice probability vs RPE plot", fontsize=15)

        if save:
            plt.savefig(save, bbox_inches="tight")

    def logit_rpe(self, save=None):

        mean_rpe_r = np.zeros(30)
        mean_rpe_l = np.zeros(30)
        mean_logit_r = np.zeros(30)
        mean_logit_l = np.zeros(30)

        prob_vec = self.actor_logits[:, 1] - self.actor_logits[:, 2]

        for i in range(1, 31):

            if i == 30:
                mean_logit_r[i - 1] = np.mean(
                    prob_vec[
                        (self.current_trial_times == 1)
                        & (self.reward_probs == -1)
                        & (self.trial_count_in_blocks == 1)
                        ]
                )
                mean_logit_l[i - 1] = np.mean(
                    prob_vec[
                        (self.current_trial_times == 1)
                        & (self.reward_probs == 1)
                        & (self.trial_count_in_blocks == 1)
                        ]
                )
            else:
                mean_logit_r[i - 1] = np.mean(
                    prob_vec[
                        (self.current_trial_times == 1)
                        & (self.reward_probs == 1)
                        & (self.trial_count_in_blocks == i + 1)
                        ]
                )
                mean_logit_l[i - 1] = np.mean(
                    prob_vec[
                        (self.current_trial_times == 1)
                        & (self.reward_probs == -1)
                        & (self.trial_count_in_blocks == i + 1)
                        ]
                )

            mean_rpe_r[i - 1] = np.mean(
                self.rpes[
                    (self.current_trial_times > self.num_states // 2)
                    & (self.reward_probs == 1)
                    & (self.trial_count_in_blocks == i)
                    ]
            )
            mean_rpe_l[i - 1] = np.mean(
                self.rpes[
                    (self.current_trial_times > self.num_states // 2)
                    & (self.reward_probs == -1)
                    & (self.trial_count_in_blocks == i)
                    ]
            )

        color_range = np.arange(1, 31)
        cbar_name = "Trials in a left to right block"

        plt.figure(figsize=(10, 10))
        plt.plot(
            mean_rpe_r,
            mean_logit_r,
            color="grey",
        )

        fig1 = plt.scatter(
            mean_rpe_r,
            mean_logit_r,
            c=color_range,
            cmap="cool",
            edgecolors="black",
            s=100,
        )
        plt.plot(
            mean_rpe_l,
            mean_logit_l,
            color="grey",
        )

        fig2 = plt.scatter(
            mean_rpe_l,
            mean_logit_l,
            c=color_range,
            cmap="hot",
            edgecolors="black",
            s=100,
        )
        cbar = plt.colorbar(fig1)
        cbar.set_label("Trials in a left block", fontsize=15)
        cbar2 = plt.colorbar(fig2)
        cbar2.set_label("Trials in a right block", fontsize=15)
        plt.ylabel("Decision Variable", fontsize=20)
        plt.xlabel("RPE", fontsize=20)
        plt.title("Decision variable vs RPE plot", fontsize=15)

        if save:
            plt.savefig(save, bbox_inches="tight")

    def value_plot(self, save=None):

        val_r = np.array(
            [
                np.mean(
                    self.values[
                        (self.current_trial_times == (i + 1) % self.num_states)
                        & (self.choices == 1)
                        ]
                )
                for i in range(self.num_states)
            ]
        )
        val_l = np.array(
            [
                np.mean(
                    self.values[
                        (self.current_trial_times == (i + 1) % self.num_states)
                        & (self.choices == 0)
                        ]
                )
                for i in range(self.num_states)
            ]
        )

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(-2.1, 2.1, 0.1), val_l)
        plt.xlabel("time")
        plt.ylabel("value left")

        plt.subplot(1, 2, 2)
        plt.plot(np.arange(-2.1, 2.1, 0.1), val_r)
        plt.xlabel("time")
        plt.ylabel("value right")

        if save:
            plt.savefig(save, bbox_inches="tight")

    def value_rpe_plot(self, save=None):

        val = np.array(
            [
                np.mean(
                    self.values[
                        (self.current_trial_times == (i + 1) % self.num_states)
                        & (self.choices != -1)
                        ]
                )
                for i in range(self.num_states)
            ]
        )

        rpe_r = np.array(
            [
                np.mean(
                    self.rpes[
                        (self.current_trial_times == (i + 2) % self.num_states)
                        & (self.choices != -1)
                        & (self.is_rewardeds == 1)
                        ]
                )
                for i in range(self.num_states)
            ]
        )

        rpe_ur = np.array(
            [
                np.mean(
                    self.rpes[
                        (self.current_trial_times == (i + 2) % self.num_states)
                        & (self.choices != -1)
                        & (self.is_rewardeds == 0)
                        ]
                )
                for i in range(self.num_states)
            ]
        )

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(-2.1, 2.1, 0.1), val, label="value")
        plt.plot(
            np.arange(-2.1, 2.1, 0.1),
            rpe_r,
            label="rpe - rewarded trials",
        )
        plt.xlabel("time")
        # plt.ylabel('value left')

        plt.subplot(1, 2, 2)
        plt.plot(np.arange(-2.1, 2.1, 0.1), val, label="value")
        plt.plot(
            np.arange(-2.1, 2.1, 0.1),
            rpe_ur,
            label="rpe - unrewarded trials",
        )
        plt.xlabel("time")
        # plt.ylabel('value right')

        if save:
            plt.savefig(save, bbox_inches="tight")

    def value_switch(self, trial_back=10, save=None):

        high_prob = self.reward_probs[(self.current_trial_times == 2)]
        chosen_side = (
                2 * self.choices[(self.current_trial_times == 2)] - 1
        )
        value = np.mean(
            np.array(
                [
                    self.values[
                        (self.current_trial_times == (i + 1) % self.num_states)
                    ]
                    for i in range(self.num_states - 1)
                ]
            ),
            axis=0,
        )

        switch_high = np.where(np.diff(high_prob) != 0)[0] + 1
        early_trials = switch_high[0: len(switch_high) - 1]
        block_iden = high_prob[early_trials]

        time_window = np.arange(-trial_back, trial_back + 1)

        # finds times of left to right
        block_switch = early_trials[block_iden == 1] - 1

        window_vec = np.zeros(len(high_prob)) - 100

        for i in np.arange(len(block_switch)):
            window_vec[time_window + block_switch[i]] = time_window

        mean_val_same_1 = np.array(
            [np.mean(value[(window_vec == i) & (chosen_side == 1)]) for i in
             time_window])
        mean_val_diff_1 = np.array(
            [np.mean(value[(window_vec == i) & (chosen_side == -1)]) for i in
             time_window])

        # finds times from right to left
        block_switch = early_trials[block_iden == -1] - 1

        window_vec = np.zeros(len(high_prob)) - 100

        for i in np.arange(len(block_switch)):
            window_vec[time_window + block_switch[i]] = time_window

        mean_val_same_2 = np.array(
            [np.mean(value[(window_vec == i) & (chosen_side == -1)]) for i in
             time_window])
        mean_val_diff_2 = np.array(
            [np.mean(value[(window_vec == i) & (chosen_side == 1)]) for i in
             time_window])

        mean_val_same = (mean_val_same_1 + mean_val_same_2) / 2
        mean_val_diff = (mean_val_diff_1 + mean_val_diff_2) / 2

        ax = plt.figure(figsize=(8, 4)).gca()
        ax.axvline(x=0, linestyle="dotted", color="gray")
        ax.plot(time_window, mean_val_same)
        ax.plot(time_window, mean_val_diff)
        # ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=10))
        ax.text(
            -10,
            2.75,
            "Blue - pre-switch \nhigh probability\nchoice  ",
            fontsize=15,
            color="dodgerblue",
        )
        ax.text(
            1,
            2.75,
            "Orange - post-switch \nhigh probability\nchoice  ",
            fontsize=15,
            color="orange",
        )
        plt.xlabel("Trials from block switch")
        plt.ylabel("value(choice)")
        # plt.axhline(y=plot_trace[time_window==1],color='k',linestyle='dotted')
        # print(plot_trace[time_window == 1])

        if save:
            plt.savefig(save, bbox_inches="tight")

    def logit_rpe_reg(self, trials_back=10, save=None):

        rpe_r = (
                np.mean(
                    np.array(
                        [
                            self.rpes[
                                self.current_trial_times == (
                                            i + 2) % self.num_states]
                            for i in
                            range(self.num_states // 2, self.num_states)
                        ]
                    ),
                    axis=0,
                )
                * (self.reward_probs[self.current_trial_times == 2] == -1)
                * 1
        )

        rpe_l = (
                np.mean(
                    np.array(
                        [
                            self.rpes[
                                self.current_trial_times == (
                                            i + 2) % self.num_states]
                            for i in
                            range(self.num_states // 2, self.num_states)
                        ]
                    ),
                    axis=0,
                )
                * (self.reward_probs[self.current_trial_times == 2] == 1)
                * 1
        )

        logit = self.actor_logits[
            (self.current_trial_times == 1), 1
        ]  # - self.actor_logits[(self.current_trial_times==1), 1]

        # makes x matrix of reward identity on previous trials
        r_mat = np.zeros((trials_back, len(rpe_r)))
        l_mat = np.zeros((trials_back, len(rpe_l)))

        for i in np.arange(0, trials_back):
            r_mat[i, :] = np.roll(rpe_r, i)
            l_mat[i, :] = np.roll(rpe_l, i)

        y = np.reshape(logit, [len(logit), 1])
        x = np.concatenate((np.ones([1, len(logit)]), r_mat, l_mat), axis=0)

        results1 = sm.OLS(y, x.T).fit()

        logit = self.actor_logits[
            (self.current_trial_times == 1), 2
        ]  # - self.actor_logits[(self.current_trial_times==1), 1]

        y = np.reshape(logit, [len(logit), 1])
        x = np.concatenate((np.ones([1, len(logit)]), l_mat, r_mat), axis=0)

        results2 = sm.OLS(y, x.T).fit()

        plt.figure(figsize=(11, 4))
        plt.title("Regressing logit against rpe on previous trials",
                  fontsize=20)
        plt.plot(
            np.arange(1, trials_back + 1),
            (
                    results1.params[1: trials_back + 1]
                    + results2.params[1: trials_back + 1]
            )
            / 2,
            color="green",
            label="Same side block",
        )
        plt.plot(
            np.arange(1, trials_back + 1),
            (
                    results1.params[trials_back + 1: None]
                    + results2.params[trials_back + 1: None]
            )
            / 2,
            color="red",
            label="Opposite side Block",
        )
        plt.legend()
        plt.axhline(y=0, linestyle="dashed", color="k")
        plt.xlabel("trials back", fontsize=15)
        plt.ylabel("regression coefficients", fontsize=15)

        if save:
            plt.savefig(save, bbox_inches="tight")
