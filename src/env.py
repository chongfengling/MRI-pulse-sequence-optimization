import numpy as np
import matplotlib.pyplot as plt
from utilities import *
import argparse

class Env:
    # Environment, generates state and reward based on action
    def __init__(self, args, plot=False):
        # create the universal environment
        # x space (spatial space)
        self.FOV_x = args.FOV_x  # field of view in x space (mm)
        self.N = (
            args.N  # sampling points in x space (and k space, time space during ADC)
        )
        self.delta_x = self.FOV_x / self.N  # sampling interval in x space, (mm)
        self.x_axis = np.linspace(
            -self.FOV_x / 2, self.FOV_x / 2 - self.delta_x, self.N
        )  # symmetric x space
        # k space (frequency space)
        self.delta_k = 1 / self.FOV_x  # sampling interval in k space (1/mm)
        self.FOV_k = self.delta_k * self.N  # field of view in k space (1/mm)
        self.k_axis = np.linspace(
            -self.FOV_k / 2, self.FOV_k / 2 - self.delta_k, self.N
        )  # symmetric k space
        self.gamma = .68e5 # rad/ms/T
        self.gamma_bar = 0.5 * self.gamma / np.pi  # ms^-1T^-1
        # t space (time space) based on G1 and G2
        # create object over x space. one env, one object
        self.density = np.zeros(len(self.x_axis))
        # self.density[int(len(self.x_axis) * 0.4) : int(len(self.x_axis) * 0.6)] = 2
        self.density[int(len(self.x_axis) / 4 + len(self.x_axis) / 8): int(len(self.x_axis) / 4 * 3 - len(self.x_axis) / 8)] = 2
        self.density_complex = self.density.astype(complex)
        # prepare for simulation
        # create spins after the rf pulse (lying on the y-axis)
        # assume the spins are lying on each sampling point over y-axis
        self.m0 = 1.0
        self.w_0 = 0
        self.vec_spins = np.zeros((3, self.N))
        self.vec_spins[1, :] = 1
        self.env_name = args.env
        self.T2 = args.T2

        self.t_axis = None
        self.G_values_array = None
        self.k_traj = None

    def make(self, args):
        # specify the action space
        if self.env_name == "Two-Constant-Gradient":
            # 6 variables: t1:_, t3:_, d1:_, d2:_, G1symbol:_, G2symbol:_, Gvalues:_
            self.action_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            self.action_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            self.action_space = np.random.uniform(
                low=self.action_low, high=self.action_high, size=7
            )
        elif self.env_name == "Two-Constant-Gradient-With_Slope":
            self.action_low = np.array([0.0, 0.0, 0.0, 0.0])
            self.action_high = np.array([1.0, 1.0, 1.0, 1.0])
            self.action_space = np.random.uniform(
                low=self.action_low, high=self.action_high, size=4
            )
            self.max_slew_rate = args.max_slew_rate
            self.slope_penalty_factor = args.slope_penalty_factor
        else:
            raise ValueError("Invalid environment name.")

    def reset(self):
        # reset and return a new state (reconstructed density with two components: real and imaginary)
        return np.random.rand(len(self.x_axis) * 2)

    def step(self, action, plot=False, info=0):
        # info: used to measure how mse changes before and after the step. Here is the previous mse

        # (d1, d2, d3, d4, d5, d6, d7, d8, GValue_01) = action
        (alpha, beta, GValue_01) = action
        # max gradient is 40 mT/m = 4e-5 T/mm
        GValue = GValue_01 * 1e-5 * 40
        G1 = GValue * 1
        G2 = GValue * (-1)
        # calculate the delta_t and total time based on max gradient value
        gamma_bar_G = self.gamma_bar * GValue
        delta_t = self.delta_k / gamma_bar_G  # Fixed ms

        Ts = self.FOV_k / gamma_bar_G  # ADC duration FIXED
        t_max = (
            1.5 * Ts - delta_t
        )  # maximum time (ms) (rephasing process is 2 times longer than dephasing process)

        self.t_axis = np.linspace(0, t_max, int(self.N * 1.5))
        # define gradient values over time
        G_values_array = np.zeros(len(self.t_axis))

        N_G1 = nearest_even(alpha * self.N * 0.5)
        N_G2 = nearest_even(beta * self.N)
        N_G1_up = int(0.5 * (self.N * 0.5 - N_G1))
        N_G2_up = int(0.5 * (self.N - N_G2))
        assert N_G1 + N_G2 + N_G1_up * 2 + N_G2_up * 2 == self.N * 1.5

        G_values_array[:N_G1_up] = np.linspace(0, G1, N_G1_up)
        G_values_array[N_G1_up : N_G1_up + N_G1] = G1
        G_values_array[N_G1_up + N_G1 : N_G1_up + N_G1 + N_G1_up] = np.linspace(
            G1, 0, N_G1_up
        )

        G_values_array[
            N_G1_up + N_G1 + N_G1_up : N_G1_up + N_G1 + N_G1_up + N_G2_up
        ] = np.linspace(0, G2, N_G2_up)
        G_values_array[
            N_G1_up
            + N_G1
            + N_G1_up
            + N_G2_up : N_G1_up
            + N_G1
            + N_G1_up
            + N_G2_up
            + N_G2
        ] = G2
        G_values_array[
            N_G1_up
            + N_G1
            + N_G1_up
            + N_G2_up
            + N_G2 : N_G1_up
            + N_G1
            + N_G1_up
            + N_G2_up
            + N_G2
            + N_G2_up
        ] = np.linspace(G2, 0, N_G2_up)

        # store the G_values_array
        self.G_values_array = G_values_array
        # calculate k space trajectory
        self.k_traj = np.cumsum(G_values_array)

        # do relaxation
        # define larmor frequency w_G of spins during relaxation
        # shape = (number of time steps, number of sampling points)
        w_G = np.outer(self.G_values_array, self.x_axis) * self.gamma + self.w_0
        # T1 relaxation is not considered as it does not affect the signal on the xy plane
        res = multiple_Relaxation(
            self.vec_spins,
            m0=self.m0,
            w=0,
            w0=w_G,
            t1=1e10,
            t2=self.T2,
            t_axis=self.t_axis,
            steps=int(self.N * 1.5),
            axis='z',
        )
        store = []
        for i in range(2):
            tmp = res[
                i, :, :
            ].squeeze()  # shape: (number of steps, number of sampling points)
            store.append(tmp @ self.density)  # multiply by true density
        # signal during ADC
        Mx_2, My_2 = store[0][int(self.N / 2) :], store[1][int(self.N / 2) :]
        adc_signal = Mx_2 * (-1j) + My_2
        re_density = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(adc_signal)))
        abs_re_density = np.abs(re_density)
        r_i_re_density = np.concatenate(
            (np.real(re_density), np.imag(re_density)), axis=0
        )

        # reward has two components: the first is the MSE of two complex arrays, the second is the slew rate
        # reward of mse < 0
        mse = mse_of_two_complex_nparrays(re_density, self.density_complex)

        # positive reward if mse decreases
        reward = info - mse
        # record the current mse
        info = mse

        # calculate the slew rate
        sr_1, sr_2 = (GValue / (delta_t * N_G1_up), GValue / (delta_t * N_G2_up))
        done = 0.0

        if (
            max(sr_1, sr_2) > self.max_slew_rate * 100
        ):  # unacceptable slew rate, fail for this episode
            # print(f'max slew rate exceeded: {max(sr_1, sr_2)}, GValue: {GValue}, delta_t: {delta_t}')
            # reward += -1
            done = 1.0

        if plot:
            plt.plot(self.x_axis, abs_re_density, label='reconstruction')
            plt.plot(self.x_axis, self.density, label='original')
            plt.legend()
            plt.show()
        # at this time use abs_re_density as the state
        return r_i_re_density, reward, done, info

    def render(self):
        # display the state (reconstructed density) and the object (true density) and gradients and k space trajectory

        pass

    def example(self):
        _, _, _, _ = self.step(action=[1,1,1], plot=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description='DPPG_Two-Constant-Gradient')
    parser.add_argument(
        '--env', default='Two-Constant-Gradient-With_Slope', type=str, help='env'
    )
    parser.add_argument('--FOV_x', default=32, type=int, help='FOV_x')
    parser.add_argument(
        '--N',
        default=32,
        type=int,
        help='sampling points in x space (and k space, time space during ADC)',
    )
    parser.add_argument(
        '--T2', default=3e1, type=float, help='T2 relaxation time in ms'
    )
    parser.add_argument(
        '--max_slew_rate',
        default=2e-4,
        type=float,
        help='max slew rate of gradient in (T / (m * ms)), defined as peak amplitude of gradient divided by rise time',
    )
    parser.add_argument(
        '--slope_penalty_factor', default=0, type=float, help='slope penalty factor'
    )
    parser.add_argument('--seed', default=215, type=int, help='seed')
    parser.add_argument(
        '--state_space',
        default=32 * 2,
        type=int,
        help='state_space including density of objects',
    )
    parser.add_argument(
        '--action_space',
        default=3,
        type=int,
        help='action_space including t1, t3, d1, d2, G1symbol, G2symbol, Gvalue',
    )
    parser.add_argument(
        '--num_episode',
        default=300,
        type=int,
        help='number of episodes. Each episode initializes new random process and state',
    )
    parser.add_argument(
        '--num_steps_per_ep', default=1024, type=int, help='number of steps per episode'
    )
    parser.add_argument(
        '--num_episode_testing',
        default=64,
        type=int,
        help='number of episodes for testing. Each episode initializes new random process and state',
    )
    parser.add_argument(
        '--num_steps_per_ep_testing', default=1024, type=int, help='number of steps per episode for testing'
    )
    parser.add_argument(
        '--a_hidden1', default=512, type=int, help='hidden layer 1 in actor network'
    )
    parser.add_argument(
        '--a_hidden2', default=128, type=int, help='hidden layer 2 in actor network'
    )
    parser.add_argument(
        '--a_hidden3', default=32, type=int, help='hidden layer 3 in actor network'
    )

    parser.add_argument(
        '--c_s_hidden1',
        default=128,
        type=int,
        help='hidden layer 1 in critic network for state input stream',
    )
    parser.add_argument(
        '--c_s_hidden2',
        default=32,
        type=int,
        help='hidden layer 2 in critic network for state input stream',
    )
    parser.add_argument(
        '--c_a_hidden1',
        default=16,
        type=int,
        help='hidden layer 1 in critic network for action input stream',
    )
    parser.add_argument(
        '--c_a_hidden2',
        default=32,
        type=int,
        help='hidden layer 2 in critic network for action input stream',
    )

    parser.add_argument(
        '--c_combined_hidden1',
        default=128,
        type=int,
        help='hidden layer 1 in critic network for combined network',
    )
    parser.add_argument(
        '--c_combined_hidden2',
        default=256,
        type=int,
        help='hidden layer 2 in critic network for combined network',
    )

    parser.add_argument(
        '--lr_a', default=0.0001, type=float, help='learning rate of actor network'
    )
    parser.add_argument(
        '--lr_c', default=0.001, type=float, help='learning rate of critic network'
    )
    parser.add_argument('--warmup', default=256, type=int, help='warmup, no training')
    parser.add_argument(
        '--memory_capacity', default=10000, type=int, help='memory capacity'
    )
    parser.add_argument(
        '--batch_size', default=128, type=int, help='minibatch size from memory'
    )

    parser.add_argument(
        '--exploration_var',
        default=0.8,
        type=float,
        help='exploration variance in random process',
    )
    parser.add_argument('--left_clip', default=0.01, type=float, help='left clip')
    parser.add_argument('--right_clip', default=0.99, type=float, help='right clip')

    parser.add_argument(
        '--gamma', default=0.9, type=float, help='reward discount factor'
    )  # 0.8
    parser.add_argument(
        '--tau', default=0.01, type=float, help='soft update factor'
    )  # 0.1

    args = parser.parse_args()

    return args



args = parse_arguments()
a = Env(args)
a.example()


# hard constrain
# reward: change of mse
