import numpy as np
import matplotlib.pyplot as plt


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


def single_FID(vector, m0, w, w0, t1, t2, t, axis):
    """Free Induction Decay of a single vector in a 3-D environment. See more: Bloch Equation

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
    delta_frequency = w0 - w
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


def multiple_FID(vectors, m0, w, w0, t1, t2, t, steps, axis):
    """Free Induction Decay of a set of vectors in a 3-D environment. See more: Bloch Equation

    Parameters
    ----------
    vectors : np.array, shape=(3,m)
        a set of m vectors
    m0 : float or np.array, shape=(m,)
        magnetisation of m vectors. If the density is not uniform, m0 is a vector.
    w : float or np.array
        rotating frame frequency.
    w0 : np.array, shape=(m,)
        Larmor frequency of m vectors
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
    delta_time = t / steps

    res = np.ones((3, steps, num_vectors)) * 1e6
    res[:, 0, :] = vectors

    m0 = np.full(num_vectors, m0) if isinstance(m0, float) else m0

    for i in range(steps - 1):
        for j in range(num_vectors):
            res[:, i + 1, j] = np.squeeze(
                single_FID(
                    res[:, 0, j],
                    m0=m0[j],
                    w=w,
                    w0=w0[j],
                    t1=t1,
                    t2=t2,
                    t=delta_time * (i + 1),
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
    eraser_points = int(num_spins / 10)
    eraser_density = 0.5 * rho_0 * np.ones(eraser_points)
    nib_points = int(num_spins / 4)
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


