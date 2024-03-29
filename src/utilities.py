import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def nearest_even(n):
    return int(round(n)) + (1 if round(n) % 2 else 0) if n - round(n) >= 0 else int(round(n)) - (1 if round(n) % 2 else 0)


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def mse_of_two_complex_nparrays(c1, c2):
    assert c1.shape == c2.shape, 'The shapes of two complex arrays are not equal.'
    mse = (np.linalg.norm(c1 - c2) ** 2) / len(c1)
    return mse

def show_state(env, path, state, i, j, info, reward):
    _, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=False, figsize=(10, 6))
    ax1.plot(env.t_axis, env.G_values_array)
    ax1.set_ylabel('Gx (T/mm)')
    ax1.set_ylim([-1e-5 * 4, 1e-5 * 4])
    ax2.plot(env.t_axis, env.k_traj)
    ax2.set_xticks(
        [
            0,
            env.t_axis[int(env.N / 2) - 1],
            env.t_axis[int(env.N * 1.5) - 1],
        ]
    )
    ax2.set_ylabel('k(t) (1/mm)')
    ax2.set_xticklabels(['$t_1$', r'$t_2 (t_3)$', r'$t_4$=' + str(env.t_axis[int(env.N * 1.5) - 1])[:5]])
    ax1.set_xticks(
        [
            0,
            env.t_axis[int(env.N / 2) - 1],
            env.t_axis[int(env.N * 1.5) - 1],
        ]
    )
    ax1.set_ylabel('k(t) (1/mm)')
    ax1.set_xticklabels(['$t_1$', r'$t_2 (t_3)$', r'$t_4$=' + str(env.t_axis[int(env.N * 1.5) - 1])[:5]])
    ax3.plot(env.x_axis, state[:int(len(state)/2)], '-o', label='real')
    ax3.plot(env.x_axis, state[int(len(state)/2):], '-*', label='imag')
    ax3.plot(env.x_axis, env.density, 'b-', label='object_real')
    ax3.plot(env.x_axis, np.zeros(len(env.x_axis)), 'k-', label='object_imag')
    ax3.legend()
    ax4.plot(env.x_axis, env.density, '-', label='object')
    ax4.plot(env.x_axis, np.abs(state[:int(len(state)/2)] + 1j * state[int(len(state)/2):]),
    '-o', label='reconstruction')
    ax1.set_title(f'mse = {info}, reward = {reward}')
    ax4.legend()
    ax4.set_xlabel(r'$x$')
    ax3.set_title(r'real and image part of $\rho(x)$ ')
    ax4.set_title(r'absolute $\rho(x)$')
    ax3.set_ylabel(r'$\rho(x)$')
    ax4.set_ylabel(r'$\rho(x)$')
    plt.tight_layout()
    plt.savefig(path + f'i_{i}_j_{j}.png', dpi=300) 
    plt.close()

def rotation(vectors, angle, axis):
    """Rotates m isochronism vectors (located at the origin) about axis by some angle in 3-D space. When looking down from the above aixs and angle > 0, the precession of vectors is clockwise.

    Parameters
    ----------
    vectors : np.array
        m isochronism vectors located at the origin in 3-D space, shape = (3, m)
    angle : float
        rotation angle
    axis : string
        axis = 'x' or 'y' or 'z'

    Returns
    -------
    np.array
        rotated_vectors, same shape to vectors

    Examples
    --------
    >>> np.isclose(rotation(np.array([1,0,0], float).reshape(3,1), angle=np.pi / 2, axis='z'), [[0], [-1], [0]])
    True
    """
    # check the validation of inputs
    assert vectors.shape[0] == 3, 'vectors is not in 3-D space.'
    # determine the rotation matrix with angle and axis
    if axis == 'x':
        rotation_matrix = np.asarray(
            [
                [1, 0, 0],
                [0, np.cos(angle), np.sin(angle)],
                [0, -np.sin(angle), np.cos(angle)],
            ]
        )
    elif axis == 'y':
        rotation_matrix = np.asarray(
            [
                [np.cos(angle), 0, 0],
                [0, 1, 0],
                [np.sin(angle), np.cos(angle), 0],
            ]
        )
    elif axis == 'z':
        rotation_matrix = np.asarray(
            [
                [np.cos(angle), np.sin(angle), 0],
                [-np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise ValueError('Please input an valid axis.')
    # operate
    rotated_vectors = rotation_matrix @ vectors

    return rotated_vectors


def single_Relaxation(vector, m0, w, w0, t1, t2, t, axis):
    """relaxation process of a single vector in a 3-D environment. See more: Bloch Equation

    Parameters
    ----------
    vector : np.array, shape=(3,1)
        a single vector or some isochronism vectors at time t=0.
    m0 : float
        the equilibrium value (m_0 = C * B_0 / T). see p55
    w : float
        rotating frame frequency.
    w0 : float
        the Larmor frequency
    t1 : float
        T1 relaxation
    t2 : float
        T2 relaxation
    t : float
        time of FID
    axis : string
        axis = 'x' or 'y' or 'z'
    """
    # frequency difference between rotating frame and the Larmor frequency of the vector
    delta_frequency = np.round(w0 - w, 4)
    vector_t = np.zeros((3, 1)) * 1e-10
    #
    if axis == 'x':
        raise NotImplementedError
    elif axis == 'y':
        raise NotImplementedError
    elif axis == 'z':
        vector_t[0] = np.exp(-t / t2) * (
            vector[0] * np.cos(delta_frequency * t)
            + vector[1] * np.sin(delta_frequency * t)
        )
        vector_t[1] = np.exp(-t / t2) * (
            vector[1] * np.cos(delta_frequency * t)
            - vector[0] * np.sin(delta_frequency * t)
        )
        vector_t[2] = vector[2] * np.exp(-t / t1) + m0 * (1 - np.exp(-t / t1))
    else:
        raise ValueError('Please input an valid axis.')

    return vector_t


def multiple_Relaxation(vectors, m0, w, w0, t1, t2, t_axis, steps, axis):
    """relaxation process of a set of vectors in a 3-D environment. See more: Bloch Equation

    Parameters
    ----------
    vectors : np.array, shape=(3,m)
        a set of m vectors
    m0 : float or np.array, shape=(m,)
        magnetisation of m vectors. If the density is not uniform, m0 is a vector.
    w : float or np.array
        rotating frame frequency.
    w0 : np.array, shape=(steps,m)
        Larmor frequency of m vectors at each step.
    t1 : float
        T1 relaxation
    t2 : float
        T2 relaxation
    t : float
        time of FID
    steps : int
        number of steps
    axis : string
        axis = 'x' or 'y' or 'z'

    Returns
    -------
    np.array, shape=(3,steps,m)
        vectors after FID.

    """
    (_, num_vectors) = vectors.shape
    # delta_time = t / steps
    assert w0.shape[0] == steps
    assert w0.shape[1] == int(num_vectors)

    res = np.ones((3, steps, num_vectors)) * 1e6
    res[:, 0, :] = vectors

    m0 = np.full(num_vectors, m0) if isinstance(m0, float) else m0

    for i in range(steps - 1):
        for j in range(num_vectors):
            res[:, i + 1, j] = np.squeeze(
                single_Relaxation(
                    res[:, i, j],
                    m0=m0[j],
                    w=w,
                    w0=w0[i, j],
                    # w0=w0[j],
                    t1=t1,
                    t2=t2,
                    t=t_axis[i+1] - t_axis[i],
                    axis=axis,
                )
            )
    return res


def w_grad(w_0, G_value, gamma, pos, dim=2):
    """Calculate the gradient of w with respect to gradient and position of spins. See more: ...

    Parameters
    ----------
    w_0 : float
        rotating frame frequency
    G_value : np.array
        (linear) gradient
    gamma : float
        gyromagentic ratio
    pos : np.array
        position of m spins
    dim : int
        dimension of space
    """
    if dim == 1:
        # pos.shape = (1, m)
        # G_value.shape = (1,)
        w_G = w_0 + gamma * G_value * pos
    elif dim == 2:
        raise NotImplementedError
    else:
        raise NotImplementedError

    return w_G


def density_pencil(z, delta_z, rho_0, plot=False):
    """physical image profile of a 1-D pencil. see more: Fig. 9.7

    Parameters
    ----------
    z : _type_
        _description_
    delta_z : _type_
        _description_
    rho_0 : _type_
        _description_
    plot : bool, optional
        _description_, by default False
    """
    num_spins = int((z / delta_z) * 2)
    eraser_points = int(num_spins / 10000)
    eraser_density = 0.5 * rho_0 * np.ones(eraser_points)
    nib_points = int(num_spins / 2)
    nib_density = rho_0 - rho_0 * ((np.arange(nib_points) + 1) / nib_points)
    stick_points = int(num_spins - eraser_points - nib_points)
    stick_density = np.ones(stick_points) * rho_0
    pencil_density = np.concatenate((eraser_density, stick_density, nib_density))
    if plot:
        x_axis = np.linspace(-z, z, num_spins)
        plt.plot(x_axis, pencil_density)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$\rho(z)$')
        plt.title('1D Pencil Density')
        plt.grid(True)
        plt.show()
    return pencil_density


def density_profile_umbrella(
    A_length, delta_x, max_rho, breakpoints=[0.05, 0.15, 1], plot=False
):
    """generate a 1-D umbrella density profile.
    ---    |\
       |   | \
        \  |  \ 
         - |   \ 
    Parameters
    ----------
    A_length : float
        size of the umbrella
    delta_x : float
        length of each step
    max_rho : float
        maximum density
    breakpoints : list, optional
        where the shape changed, by default [0.05, 0.15, 1]
    plot : bool, optional
        plot the figure or not, by default False

    Returns
    -------
    np.array
        density profile over x-axis
    """

    assert len(breakpoints) == 3, 'breakpoints should be a list of length 2.'

    x_axis = np.arange(0, A_length, delta_x)
    y = np.array([])
    # first part
    constant_part = lambda x: 0.5 * max_rho * np.ones(len(x))
    sub_x_axis = x_axis[x_axis <= breakpoints[0] * A_length]
    y = np.concatenate((y, constant_part(sub_x_axis)))
    # second part
    quadratic_part = (
        lambda x: 0.2 * max_rho
        + (x - A_length * breakpoints[1]) ** 2
        * (0.3 * max_rho)
        / ((breakpoints[1] - breakpoints[0]) * A_length) ** 2
    )
    sub_x_axis = x_axis[
        (x_axis > breakpoints[0] * A_length) & (x_axis <= breakpoints[1] * A_length)
    ]
    y = np.concatenate((y, quadratic_part(sub_x_axis)))
    # third part
    linear_part = lambda x: (-x + A_length) * (
        max_rho / (A_length * (1 - breakpoints[1]))
    )
    functions = [constant_part, quadratic_part, linear_part]
    sub_x_axis = x_axis[x_axis > breakpoints[1] * A_length]
    y = np.concatenate((y, linear_part(sub_x_axis)))

    if plot:
        plt.plot(x_axis, y)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$\rho(x)$')
        plt.title('1D Umbrella Density')
        plt.grid(True)
        plt.show()
    return y


def k_space(G_values, gamma):
    """change from time domain to k-space domain

    Parameters
    ----------
    G_values : np.array, shape=(steps,)
        1d gradient first
    gamma : float
        not gamma bar

    Returns
    -------
    np.array, shape=(steps,)
        _description_
    """
    # gamma-bar = gamma / 2pi
    gamma_bar = gamma / (2 * np.pi)
    return np.cumsum(G_values) * gamma_bar
