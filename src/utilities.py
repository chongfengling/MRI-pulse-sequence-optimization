import numpy as np

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
    if axis == 'x': rotation_matrix = np.asarray([[1, 0, 0], [0, np.cos(angle), np.sin(angle)], [0, -np.sin(angle), np.cos(angle)]])
    elif axis == 'y': rotation_matrix = np.asarray([[np.cos(angle), 0, 0], [0, 1, 0], [np.sin(angle), np.cos(angle), 0]])
    elif axis == 'z': rotation_matrix = np.asarray([[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    else: raise ValueError('Please input an valid axis.')
    # operate
    rotated_vectors = rotation_matrix @ vectors

    return rotated_vectors

def single_FID(vector, m0, w, w0, t1, t2, t, axis):
    """Free Induction Decay of a single vector in a 3-D environment. See more: Bloch Equation

    Parameters
    ----------
    vector : np.array
        a single vector or some isochronism vectors at time t=0.
    m0 : float
        magnetisation of the vector
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
    vector_t = np.zeros((3,1)) * 1e-10
    # 
    if axis == 'x': raise NotImplementedError
    elif axis =='y': raise NotImplementedError
    elif axis == 'z': 
        vector_t[0] = np.exp(- t / t2) * (vector[0] * np.cos(delta_frequency * t) + vector[1] * np.sin(delta_frequency * t))
        vector_t[1] = np.exp(- t / t2) * (vector[1] * np.cos(delta_frequency * t) - vector[0] * np.sin(delta_frequency * t))
        vector_t[2] = vector[2] * np.exp(-t/t1) + m0 * (1 - np.exp(-t / t1))
    else: raise ValueError('Please input an valid axis.')

    return vector_t
