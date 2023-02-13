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
