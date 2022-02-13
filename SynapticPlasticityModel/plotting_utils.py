import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator


class GenerateTestPlots:
    """Generates plots using data collected during model run."""

    def __init__(self, results):
        """Initialize the plot generator.

        Args:
            results (dict): Model run data.
        """

        self.step_time = results["step_time"]
        self.times = results["times"]
        self.high_prob_blocks = results["high_prob_blocks"]
        self.choices = results["choices"]
        self.rewarded_sides = results["rewarded_sides"]
        self.rewarded_trials = results["rewarded_trials"]
        self.values = results["values"]
        self.RPEs = results["RPEs"]
        self.stimulated_trials = results["stimulated_trials"]
        self.right_decision_value = results["right_decision_value"]
        self.left_decision_value = results["left_decision_value"]
        self.NAc_activity = results["NAc_activity"]
        self.peak_reward_times = results["peak_reward_times"]

    def behavior(self, start_num_trials=500, num_tr=200, save=None):

        print(f"reward rate = " f"{np.mean(self.rewarded_trials)}")

        block_switches = np.where(np.diff(self.high_prob_blocks) != 0)[0] + 1

        print(
            f"mean block length ="
            f"{np.mean(np.diff(block_switches))}"
            "\u00B1"
            f"{np.std(np.diff(block_switches))}"
        )

        end_num_trials = start_num_trials + num_tr

        rew_side = self.choices * self.rewarded_trials
        decision_value = self.left_decision_value - self.right_decision_value
        max_decision_value = np.max(
            np.abs(decision_value[start_num_trials:end_num_trials])
        )

        plt.figure(figsize=(16, 5))
        plt.scatter(
            np.arange(start_num_trials, end_num_trials),
            self.high_prob_blocks[start_num_trials:end_num_trials] * 1.5,
            s=40,
            color="black",
            label="high probability side",
        )
        plt.scatter(
            np.arange(start_num_trials, end_num_trials),
            self.choices[start_num_trials:end_num_trials] * 1.3,
            s=20,
            color="blue",
            label="chosen side",
        )
        plt.scatter(
            np.arange(start_num_trials, end_num_trials),
            rew_side[start_num_trials:end_num_trials] * 1.1,
            s=30 * np.abs(rew_side[start_num_trials:end_num_trials]),
            color="green",
            label="rewarded side",
        )
        plt.plot(
            np.arange(start_num_trials, end_num_trials),
            decision_value[start_num_trials:end_num_trials] / max_decision_value,
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

        if save:
            plt.savefig(save, bbox_inches="tight")

    def stay_probability(self, mode=None, save=None):
        if mode == "optogenetic":

            unrewarded_trials = np.abs(self.rewarded_trials - 1)
            next_choice = np.roll(self.choices, -1)
            next_id = np.roll(self.stimulated_trials, -1)
            norm_chosen_side = self.choices[next_id == 0]
            norm_next_choice = next_choice[next_id == 0]
            norm_rewarded_trials = self.rewarded_trials[next_id == 0]
            norm_unrewarded_trials = unrewarded_trials[next_id == 0]
            norm_return_rew = norm_rewarded_trials * (
                (norm_next_choice == norm_chosen_side) * 1
            )
            norm_prob_rew_return = np.sum(norm_return_rew) / np.sum(norm_rewarded_trials)
            norm_return_unrew = norm_unrewarded_trials * (
                (norm_next_choice == norm_chosen_side) * 1
            )
            norm_prob_unrew_return = np.sum(norm_return_unrew) / np.sum(
                norm_unrewarded_trials
            )

            opto_chosen_side = self.choices[next_id == 1]
            opto_next_choice = next_choice[next_id == 1]
            opto_rewarded_trials = self.rewarded_trials[next_id == 1]
            opto_unrewarded_trials = unrewarded_trials[next_id == 1]
            opto_return_rew = opto_rewarded_trials * (
                (opto_next_choice == opto_chosen_side) * 1
            )
            opto_prob_rew_return = np.sum(opto_return_rew) / np.sum(opto_rewarded_trials)
            opto_return_unrew = opto_unrewarded_trials * (
                (opto_next_choice == opto_chosen_side) * 1
            )
            opto_prob_unrew_return = np.sum(opto_return_unrew) / np.sum(
                opto_unrewarded_trials
            )

            fig = plt.figure(figsize=(9, 3))
            plt.subplot(1, 2, 1)
            plt.bar(
                [1, 2, 4, 5],
                [
                    norm_prob_rew_return,
                    opto_prob_rew_return,
                    norm_prob_unrew_return,
                    opto_prob_unrew_return,
                ],
                width=0.6,
                color=["k", "aqua", "k", "aqua"],
            )
            plt.xticks(
                [1.5, 4.5], ["current trial\n reward", "current trial\n no reward"]
            )
            plt.ylim([0.5, 1.0])
            plt.ylabel("probability of return")

            norm_prob_rew_return = 0.0
            norm_prob_unrew_return = 0.0

            opto_prob_rew_return = 0.0
            opto_prob_unrew_return = 0.0

            unrewarded_trials = np.abs(self.rewarded_trials - 1)
            next_choice = np.roll(self.choices, -1)
            next_id = self.stimulated_trials
            norm_chosen_side = self.choices[next_id == 0]
            norm_next_choice = next_choice[next_id == 0]
            norm_rewarded_trials = self.rewarded_trials[next_id == 0]
            norm_unrewarded_trials = unrewarded_trials[next_id == 0]
            norm_return_rew = norm_rewarded_trials * (
                (norm_next_choice == norm_chosen_side) * 1
            )
            norm_prob_rew_return += np.sum(norm_return_rew) / np.sum(norm_rewarded_trials)
            norm_return_unrew = norm_unrewarded_trials * (
                (norm_next_choice == norm_chosen_side) * 1
            )
            norm_prob_unrew_return += np.sum(norm_return_unrew) / np.sum(
                norm_unrewarded_trials
            )

            opto_chosen_side = self.choices[next_id == 1]
            opto_next_choice = next_choice[next_id == 1]
            opto_rewarded_trials = self.rewarded_trials[next_id == 1]
            opto_unrewarded_trials = unrewarded_trials[next_id == 1]
            opto_return_rew = opto_rewarded_trials * (
                (opto_next_choice == opto_chosen_side) * 1
            )
            opto_prob_rew_return += np.sum(opto_return_rew) / np.sum(opto_rewarded_trials)
            opto_return_unrew = opto_unrewarded_trials * (
                (opto_next_choice == opto_chosen_side) * 1
            )
            opto_prob_unrew_return += np.sum(opto_return_unrew) / np.sum(
                opto_unrewarded_trials
            )

            # fig = plt.figure(figsize=(3,3))
            plt.subplot(1, 2, 2)
            plt.bar(
                [1, 2, 4, 5],
                [
                    norm_prob_rew_return,
                    opto_prob_rew_return,
                    norm_prob_unrew_return,
                    opto_prob_unrew_return,
                ],
                width=0.6,
                color=["k", "aqua", "k", "aqua"],
            )
            plt.xticks(
                [1.5, 4.5], ["previous trial\n reward", "previous trial\n no reward"]
            )
            plt.ylim([0.5, 1.0])
            plt.ylabel("probability of return")

        else:
            prob_rew_return = 0.0
            prob_unrew_return = 0.0

            unrewarded_trials = (self.rewarded_trials - 1) * -1
            next_choice = np.roll(self.choices, -1)
            return_rew = self.rewarded_trials * ((next_choice == self.choices) * 1)
            prob_rew_return += np.sum(return_rew) / np.sum(self.rewarded_trials)
            return_unrew = unrewarded_trials * ((next_choice == self.choices) * 1)
            prob_unrew_return += np.sum(return_unrew) / np.sum(unrewarded_trials)

            plt.subplot(1, 2, 1)
            plt.bar(
                [1, 1.5],
                [prob_rew_return, prob_unrew_return],
                width=0.2,
                color=["green", "red"],
            )
            plt.xticks([1, 1.5], ["previously\n rewarded", "previously\n unrewarded"])
            plt.ylabel("probability of return")
            plt.ylim([0.0, 1.1])

        if save:
            plt.savefig(save, bbox_inches="tight")

    def choice_regression(self, mode=None, trials_back=11, save=None):

        if mode == "optogenetic":
            reward_mat = np.zeros((trials_back - 1, len(self.rewarded_trials)))
            reward_vect = self.choices * self.rewarded_trials
            for i in np.arange(1, trials_back):
                reward_mat[i - 1, :] = np.roll(reward_vect, i)

            # makes unreward matrix
            unrewarded_trials = np.abs(self.rewarded_trials - 1)
            unreward_mat = np.zeros((trials_back - 1, len(unrewarded_trials)))
            unreward_vec = self.choices * unrewarded_trials
            for i in np.arange(1, trials_back):
                unreward_mat[i - 1, :] = np.roll(unreward_vec, i)

            # makes laser matrix
            laser_mat = np.zeros((trials_back - 1, len(self.rewarded_trials)))
            for i in np.arange(1, trials_back):
                laser_mat[i - 1, :] = np.roll(self.stimulated_trials, i)

            y = self.choices
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
            unreward_coefs = log_reg.params[trials_back : int(trials_back * 2 - 1)]
            rewlaser_coefs = log_reg.params[
                int(trials_back * 2 - 1) : int(trials_back * 3 - 2)
            ]
            norewlaser_coefs = log_reg.params[
                int(trials_back * 3 - 2) : int(trials_back * 4 - 3)
            ]

            fig = plt.figure(figsize=(12, 6))
            plt.plot(reward_coefs, "b", label="rewarded trials no stimulation")
            plt.plot(unreward_coefs, "r", label="unrewarded trials no stimulation")
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

            reward_mat = np.zeros((trials_back - 1, len(self.rewarded_trials)))
            reward_vect = self.choices * self.rewarded_trials
            for i in np.arange(1, trials_back):
                reward_mat[i - 1, :] = np.roll(reward_vect, i)

            # makes unreward matrix
            unrewarded_trials = np.abs(self.rewarded_trials - 1)
            unreward_mat = np.zeros((trials_back - 1, len(unrewarded_trials)))
            unreward_vec = self.choices * unrewarded_trials
            for i in np.arange(1, trials_back):
                unreward_mat[i - 1, :] = np.roll(unreward_vec, i)

            y = self.choices
            x = np.concatenate((np.ones([1, len(y)]), reward_mat, unreward_mat), axis=0)

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

    def block_value_plot(self, save=None):
        block_switches = np.where(np.diff(self.high_prob_blocks) != 0)[0] + 1
        early_trials = block_switches[0 : len(block_switches) - 1]
        middle_trials = early_trials + 4
        late_trials = early_trials + 14

        # block identity of each trial
        block_iden = self.high_prob_blocks[early_trials]

        # indeces for block and actual choice
        left_left_early = early_trials[
            (block_iden == 1) & (self.choices[early_trials] == 1)
        ]
        left_right_early = early_trials[
            (block_iden == 1) & (self.choices[early_trials] == -1)
        ]
        right_right_early = early_trials[
            (block_iden == -1) & (self.choices[early_trials] == -1)
        ]
        right_left_early = early_trials[
            (block_iden == -1) & (self.choices[early_trials] == 1)
        ]

        left_left_middle = middle_trials[
            (block_iden == 1) & (self.choices[middle_trials] == 1)
        ]
        left_right_middle = middle_trials[
            (block_iden == 1) & (self.choices[middle_trials] == -1)
        ]
        right_right_middle = middle_trials[
            (block_iden == -1) & (self.choices[middle_trials] == -1)
        ]
        right_left_middle = middle_trials[
            (block_iden == -1) & (self.choices[middle_trials] == 1)
        ]

        left_left_late = late_trials[(block_iden == 1) & (self.choices[late_trials] == 1)]
        left_right_late = late_trials[
            (block_iden == 1) & (self.choices[late_trials] == -1)
        ]
        right_right_late = late_trials[
            (block_iden == -1) & (self.choices[late_trials] == -1)
        ]
        right_left_late = late_trials[
            (block_iden == -1) & (self.choices[late_trials] == 1)
        ]

        fig = plt.figure(figsize=(6, 6))

        plt.subplot(3, 2, 1)
        plt.title("Left block")
        val_trace_l = np.mean(self.values[left_left_early, :], axis=0)
        val_trace_r = np.mean(self.values[left_right_early, :], axis=0)
        val_sem_l = stats.sem(self.values[left_left_early, :], axis=0)
        val_sem_r = stats.sem(self.values[left_right_early, :], axis=0)
        plt.errorbar(
            self.times,
            val_trace_l,
            val_sem_l,
            color="blue",
            ecolor="skyblue",
            linewidth=1.0,
        )
        plt.errorbar(
            self.times,
            val_trace_r,
            val_sem_r,
            color="green",
            ecolor="lime",
            linewidth=1.0,
        )
        plt.ylim([-0.03, 0.22])

        plt.subplot(3, 2, 2)
        plt.title("Right block")
        val_trace_l = np.mean(self.values[right_left_early, :], axis=0)
        val_trace_r = np.mean(self.values[right_right_early, :], axis=0)
        val_sem_l = stats.sem(self.values[right_left_early, :], axis=0)
        val_sem_r = stats.sem(self.values[right_right_early, :], axis=0)
        plt.errorbar(
            self.times,
            val_trace_l,
            val_sem_l,
            color="blue",
            ecolor="skyblue",
            linewidth=1.0,
        )
        plt.errorbar(
            self.times,
            val_trace_r,
            val_sem_r,
            color="green",
            ecolor="lime",
            linewidth=1.0,
        )
        plt.ylim([-0.03, 0.22])
        plt.text(4, 0.19, "Trial 1", fontsize=15, color="k")

        plt.subplot(3, 2, 3)
        val_trace_l = np.mean(self.values[left_left_middle, :], axis=0)
        val_trace_r = np.mean(self.values[left_right_middle, :], axis=0)
        val_sem_l = stats.sem(self.values[left_left_middle, :], axis=0)
        val_sem_r = stats.sem(self.values[left_right_middle, :], axis=0)
        plt.errorbar(
            self.times,
            val_trace_l,
            val_sem_l,
            color="blue",
            ecolor="skyblue",
            linewidth=1.0,
        )
        plt.errorbar(
            self.times,
            val_trace_r,
            val_sem_r,
            color="green",
            ecolor="lime",
            linewidth=1.0,
        )
        plt.ylim([-0.03, 0.22])
        plt.ylabel("self.values", fontsize=15)

        plt.subplot(3, 2, 4)
        val_trace_l = np.mean(self.values[right_left_middle, :], axis=0)
        val_trace_r = np.mean(self.values[right_right_middle, :], axis=0)
        val_sem_l = stats.sem(self.values[right_left_middle, :], axis=0)
        val_sem_r = stats.sem(self.values[right_right_middle, :], axis=0)
        plt.errorbar(
            self.times,
            val_trace_l,
            val_sem_l,
            color="blue",
            ecolor="skyblue",
            linewidth=1.0,
        )
        plt.errorbar(
            self.times,
            val_trace_r,
            val_sem_r,
            color="green",
            ecolor="lime",
            linewidth=1.0,
        )
        plt.ylim([-0.03, 0.22])
        plt.text(4, 0.19, "Trial 5", fontsize=15, color="k")

        plt.text(8, 0.15, "Left Press", fontsize=15, color="blue")
        plt.text(8, 0.05, "Right Press", fontsize=15, color="green")

        plt.subplot(3, 2, 5)
        val_trace_l = np.mean(self.values[left_left_late, :], axis=0)
        val_trace_r = np.mean(self.values[left_right_late, :], axis=0)
        val_sem_l = stats.sem(self.values[left_left_late, :], axis=0)
        val_sem_r = stats.sem(self.values[left_right_late, :], axis=0)
        plt.errorbar(
            self.times,
            val_trace_l,
            val_sem_l,
            color="blue",
            ecolor="skyblue",
            linewidth=1.0,
        )
        plt.errorbar(
            self.times,
            val_trace_r,
            val_sem_r,
            color="green",
            ecolor="lime",
            linewidth=1.0,
        )
        plt.ylim([-0.03, 0.22])

        plt.subplot(3, 2, 6)
        val_trace_l = np.mean(self.values[right_left_late, :], axis=0)
        val_trace_r = np.mean(self.values[right_right_late, :], axis=0)
        val_sem_l = stats.sem(self.values[right_left_late, :], axis=0)
        val_sem_r = stats.sem(self.values[right_right_late, :], axis=0)
        plt.errorbar(
            self.times,
            val_trace_l,
            val_sem_l,
            color="blue",
            ecolor="skyblue",
            linewidth=1.0,
        )
        plt.errorbar(
            self.times,
            val_trace_r,
            val_sem_r,
            color="green",
            ecolor="lime",
            linewidth=1.0,
        )
        plt.ylim([-0.03, 0.22])
        plt.text(4, 0.19, "Trial 15", fontsize=15, color="k")

        plt.text(-4.9, -0.12, "Time(s)", fontsize=15)

        if save:
            plt.savefig(save, bbox_inches="tight")

    def dopamine_regression(self, trials_back=6, save=None):

        y_vec_cs = np.zeros(len(self.RPEs))

        for i in range(len(self.RPEs)):
            y_vec_cs[i] = np.mean(
                self.RPEs[
                    i,
                    int(self.peak_reward_times[i]) : int(
                        self.peak_reward_times[i] + 1 / self.step_time
                    ),
                ]
            )

        # makes x matrix of reward identity on previous trials
        x_mat = np.zeros((trials_back, len(self.rewarded_trials)))
        for i in np.arange(0, trials_back):
            x_mat[i, :] = np.roll(self.rewarded_trials, i)

        y = np.reshape(y_vec_cs, [len(y_vec_cs), 1])
        x = np.concatenate((np.ones([1, len(y_vec_cs)]), x_mat), axis=0)

        regresion_results = sm.OLS(y, x.T).fit()

        plt.figure(figsize=(11, 4))

        plt.title("Regressing DA activity at reward time against outcome", fontsize=20)
        plt.scatter(np.arange(trials_back), regresion_results.params[1:None])
        plt.axhline(y=0, linestyle="dashed", color="k")
        plt.xlabel("trials back", fontsize=15)
        plt.ylabel("regression coefficients", fontsize=15)
        print(regresion_results.params[1:None])

        if save:
            plt.savefig(save, bbox_inches="tight")

    def block_switch(self, trial_back=10, save=None):

        switch_high = np.where(np.diff(self.high_prob_blocks) != 0)[0] + 1
        early_trials = switch_high[0 : len(switch_high) - 1]
        block_iden = self.high_prob_blocks[early_trials]

        # finds times of left to right
        block_switch = early_trials[block_iden == -1] - 1

        time_window = np.arange(-trial_back, trial_back + 1)
        r_choice_mat = np.zeros([len(block_switch), len(time_window)])
        for i in np.arange(1, len(block_switch)):
            r_choice_mat[i, :] = self.choices[time_window + block_switch[i]]
            r_choice_mat[i, :] = (r_choice_mat[i, :] + 1) / 2

        # same except right to left
        block_switch = early_trials[block_iden == 1] - 1

        time_window = np.arange(-trial_back, trial_back + 1)
        l_choice_mat = np.zeros([len(block_switch), len(time_window)])
        for i in np.arange(1, len(block_switch)):
            l_choice_mat[i, :] = self.choices[time_window + block_switch[i]] * -1
            l_choice_mat[i, :] = (l_choice_mat[i, :] + 1) / 2

        final_choice_mat = np.concatenate([l_choice_mat, r_choice_mat], axis=0)

        plot_trace = np.mean(final_choice_mat, axis=0)
        sem_trace = stats.sem(final_choice_mat, axis=0)

        plot_trace2 = np.mean(-1 * (final_choice_mat - 1), axis=0)
        sem_trace2 = stats.sem(-1 * (final_choice_mat - 1), axis=0)

        ax = plt.figure(figsize=(5, 6)).gca()
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

    def rpe_plot(self, save=None):

        rpe_rewarded = np.mean(self.RPEs[self.rewarded_trials == 1, :], axis=0)
        rpe_unrewarded = np.mean(self.RPEs[self.rewarded_trials == 0, :], axis=0)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.title("rewarded trials", fontsize=15)
        plt.plot(self.times, rpe_rewarded, color="green")
        plt.ylim(-0.4, 1)
        plt.xlabel("time", fontsize=20)
        plt.ylabel("RPE", fontsize=20)
        plt.subplot(1, 2, 2)
        plt.title("unrewarded trials", fontsize=15)
        plt.plot(self.times, rpe_unrewarded, color="grey")
        plt.ylim(-0.4, 1)
        plt.xlabel("time", fontsize=20)

        if save:
            plt.savefig(save, bbox_inches="tight")

    def plot_NAc_activity(self, max_heatmap_val=0.005, save=None):

        block_switches = np.where(np.diff(self.high_prob_blocks) != 0)[0] + 1
        early_trials = block_switches[0 : len(block_switches) - 1]
        middle_trials = early_trials + 4
        late_trials = early_trials + 14

        # block identity of each trial
        block_iden = self.high_prob_blocks[early_trials]

        # indeces for block and actual choice
        left_left_early = early_trials[
            (block_iden == 1) & (self.choices[early_trials] == 1)
        ]
        left_right_early = early_trials[
            (block_iden == 1) & (self.choices[early_trials] == -1)
        ]
        right_right_early = early_trials[
            (block_iden == -1) & (self.choices[early_trials] == -1)
        ]
        right_left_early = early_trials[
            (block_iden == -1) & (self.choices[early_trials] == 1)
        ]

        left_left_middle = middle_trials[
            (block_iden == 1) & (self.choices[middle_trials] == 1)
        ]
        left_right_middle = middle_trials[
            (block_iden == 1) & (self.choices[middle_trials] == -1)
        ]
        right_right_middle = middle_trials[
            (block_iden == -1) & (self.choices[middle_trials] == -1)
        ]
        right_left_middle = middle_trials[
            (block_iden == -1) & (self.choices[middle_trials] == 1)
        ]

        left_left_late = late_trials[(block_iden == 1) & (self.choices[late_trials] == 1)]
        left_right_late = late_trials[
            (block_iden == 1) & (self.choices[late_trials] == -1)
        ]
        right_right_late = late_trials[
            (block_iden == -1) & (self.choices[late_trials] == -1)
        ]
        right_left_late = late_trials[
            (block_iden == -1) & (self.choices[late_trials] == 1)
        ]

        NAc_heatmap = np.zeros(
            (12, self.NAc_activity.shape[1], self.NAc_activity.shape[2])
        )

        NAc_heatmap[0, :, :] = np.mean(self.NAc_activity[left_left_early, :, :], axis=0)
        NAc_heatmap[1, :, :] = np.mean(self.NAc_activity[left_right_early, :, :], axis=0)
        NAc_heatmap[2, :, :] = np.mean(self.NAc_activity[right_left_early, :, :], axis=0)
        NAc_heatmap[3, :, :] = np.mean(self.NAc_activity[right_right_early, :, :], axis=0)

        NAc_heatmap[4, :, :] = np.mean(self.NAc_activity[left_left_middle, :, :], axis=0)
        NAc_heatmap[5, :, :] = np.mean(self.NAc_activity[left_right_middle, :, :], axis=0)
        NAc_heatmap[6, :, :] = np.mean(self.NAc_activity[right_left_middle, :, :], axis=0)
        NAc_heatmap[7, :, :] = np.mean(
            self.NAc_activity[right_right_middle, :, :], axis=0
        )

        NAc_heatmap[8, :, :] = np.mean(self.NAc_activity[left_left_late, :, :], axis=0)
        NAc_heatmap[9, :, :] = np.mean(self.NAc_activity[left_right_late, :, :], axis=0)
        NAc_heatmap[10, :, :] = np.mean(self.NAc_activity[right_left_late, :, :], axis=0)
        NAc_heatmap[11, :, :] = np.mean(self.NAc_activity[right_right_late, :, :], axis=0)

        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
        for i, ax in enumerate(axes.flat):
            im = ax.imshow(
                NAc_heatmap[i, :, :],
                extent=[self.times[0], self.times[-1], self.NAc_activity.shape[1], 0],
                aspect="auto",
                cmap=cm.cividis,
                vmin=0.0,
                vmax=max_heatmap_val,
            )
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        if save:
            plt.savefig(save, bbox_inches="tight")
