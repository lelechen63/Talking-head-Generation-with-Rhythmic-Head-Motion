import numpy as np
from scipy.spatial.transform import Rotation as R

# import scipy.io as 
def frontalize(vertices):
    canonical_vertices = np.load('Data/uv-data/canonical_vertices.npy')

    vertices_homo = np.hstack((vertices, np.ones([vertices.shape[0],1]))) #n x 4
    P = np.linalg.lstsq(vertices_homo, canonical_vertices)[0].T # Affine matrix. 3 x 4
    front_vertices = vertices_homo.dot(P.T)

    return front_vertices, P
def recover(rt):
    new_rt = []
    for tt in range(rt.shape[0]):
        ret = rt[tt,:3]
        r = R.from_rotvec(ret)
        ret_R = r.as_dcm()
        ret_t = rt[tt, 3:]
        ret_t = ret_t.reshape(3,1)
        new_rt.append([ret_R, ret_t])
    return new_rt