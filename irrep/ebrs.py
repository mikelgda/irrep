import json
import numpy as np
from irreptables import load_ebr_data

def get_ebr_matrix(ebr_data):
    ebr_matrix = np.array([x["vector"] for x in ebr_data["ebrs"]], dtype=int).T

    return ebr_matrix

def get_smith_form(ebr_data, all=True):
    """Returns smith form matrices U,R,V such that the original matrix is given
    by U^{-1}RV^{-1}
    """
    u = np.array(ebr_data["smith_form"]["u"], dtype=int)
    v = np.array(ebr_data["smith_form"]["v"], dtype=int)
    r = np.array(ebr_data["smith_form"]["r"], dtype=int)

    if all:
        return u,r,v
    else:
        return r

def check_valid_multiplicity(multip):
    if not np.isclose(multip[0], np.round(multip[0]), atol=1e-3, rtol=0):
        return False
    elif not np.isclose(multip[1], 0.0, atol=1e-3, rtol=0):
        return False
    elif multip[0] < 0.:
        return False
    else:
        return True

def load_irrep_list_from_irrep_output(irrep_output):
    irrep_list = []
    data = irrep_output["characters_and_irreps"][0]["subspace"]["k-points"]
    for kpoint in data:
        for irrep_dict in kpoint["irreps"]:
            for name, multip in irrep_dict.items():
                if check_valid_multiplicity(multip):
                    multip = np.round(multip[0]).astype(int)
                    irrep_list.extend([name] * multip)

    return irrep_list
def create_symmetry_vector(irrep_list, basis):
    basis_index = {name : i for i, name in enumerate(basis["irrep_labels"])}

    vec = np.zeros(len(basis_index), dtype=int)

    for name in irrep_list:
        vec[basis_index[name]] += 1

    return vec

def load_vector_from_irrep_output(irrep_output, basis):
    irrep_list = load_irrep_list_from_irrep_output(irrep_output)
    vec = create_symmetry_vector(irrep_list, basis)

    return irrep_list, vec


def compute_ebr_decomposition(irrep_output, spinor):
    sg = irrep_output["spacegroup"]

    ebr_data = load_ebr_data(sg, "double" if spinor else "single")

    labels, vec = load_vector_from_irrep_output(irrep_output, ebr_data["basis"])

    u, r, _ = get_smith_form(ebr_data)
    d = r.diagonal()
    d = d[d > 0]

    vec_prime = u @ vec

    return vec_prime[:len(d)] % d
