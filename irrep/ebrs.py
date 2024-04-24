"""Module to compute EBR decompositions.
"""
import numpy as np
from irreptables import load_ebr_data
from .utility import vector_pprint

# Actual EBR decomposition requires OR-Tool's SAT problem solver.
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ModuleNotFoundError:
    ORTOOLS_AVAILABLE = False


def get_ebr_matrix(ebr_data):
    """
    Gets the EBR matrix from a dictionary of EBR data.

    Parameters
    ----------
    ebr_data : dict
        Dictionary containing the EBR data as saved in the package files

    Returns
    -------
    array
        EBR matrix with dimensions Nirreps x Nebrs
    """
    ebr_matrix = np.array([x["vector"] for x in ebr_data["ebrs"]], dtype=int).T

    return ebr_matrix

def get_smith_form(ebr_data, return_all=True):
    """
    Returns the Smith normal form from EBR data

    Parameters
    ----------
    ebr_data : dict
        Dictionary with EBR data as saved in the package files.
    return_all : bool, optional
        Whether to return all the matrices or only the diagonal, by default True

    Returns
    -------
    array or tuple of arrays
        Smith normal form
    """
    #U^{-1}RV^{-1}
    u = np.array(ebr_data["smith_form"]["u"], dtype=int)
    v = np.array(ebr_data["smith_form"]["v"], dtype=int)
    r = np.array(ebr_data["smith_form"]["r"], dtype=int)

    if return_all:
        return u,r,v
    else:
        return r

def get_ebr_names_and_positions(ebr_data):
    """
    Get the EBR labels and Wyckoff position from the EBR data

    Parameters
    ----------
    ebr_data : dict
        Dictionary with EBR data as save in the package files.

    Returns
    -------
    list
        list of tuples with EBR label and WP
    """
    return [(x["ebr_name"], x["wyckoff_position"]) for x in ebr_data["ebrs"]]

def load_irrep_list_from_irrep_output(irrep_output):
    """
    Reads the ouput json_data after irrep identification and makes a list with
    all irreps at all high-symmetry points.

    Parameters
    ----------
    irrep_output : dict
        json_data output from irrep

    Returns
    -------
    list
        list of irrep labels
    """
    def check_valid_multiplicity(multip):
        """
        Filter to include irreps with correct multiplicity: real, integer...

        Parameters
        ----------
        multip : tuple
            tuple with real and imaginary part of the multiplicity

        Returns
        -------
        bool
            whether the multiplicity is valid or not
        """
        # check integrality
        if not np.isclose(multip[0], np.round(multip[0]), atol=1e-2,):
            return False
        # check real
        elif not np.isclose(multip[1], 0.0, atol=1e-2):
            return False
        # check positive
        elif multip[0] < 0.:
            return False
        else:
            return True

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
    """
    Create a symmetry vector from a list of irrep labels in a given basis ordering.

    Parameters
    ----------
    irrep_list : list
        List of irrep labels
    basis : list
        List of irrep labels to use as basis

    Returns
    -------
    array
        symmetry vector: multiplicity of basis[i] at position i
    """
    basis_index = {name : i for i, name in enumerate(basis["irrep_labels"])}

    vec = np.zeros(len(basis_index), dtype=int)

    for name in irrep_list:
        vec[basis_index[name]] += 1

    return vec

def load_vector_from_irrep_output(irrep_output, basis):
    """
    Creates the irrep label list and symmetry vector from the json_data output.

    Parameters
    ----------
    irrep_output : dict
        Irrep output after irrep identification
    basis : list
        List with irrep labels to serve as basis

    Returns
    -------
    list, array
        list of irrep labels and symmetry vector
    """
    irrep_list = load_irrep_list_from_irrep_output(irrep_output)
    vec = create_symmetry_vector(irrep_list, basis)

    return irrep_list, vec

def compute_topological_classification_vector(irrep_output, ebr_data):
    """
    Checks wether the bands are topological according to the Smith singular
    values being divisors of the symmetry vector

    Parameters
    ----------
    irrep_output : dict
        json_data from irrep output after irrep identification
    ebr_data : dict
        Dictionary with EBR data as saved in the package files.

    Returns
    -------
    list, array, array, array, bool
        irrep labels, symmetry vector, transformed symmtry vector by Smith matrix,
        Smith singular values, nontriviality of the bands
    """

    labels, vec = load_vector_from_irrep_output(irrep_output, ebr_data["basis"])

    u, r, _ = get_smith_form(ebr_data)
    d = r.diagonal()
    d_pos = d[d > 0]

    vec_prime = u @ vec
    nontrivial = ((vec_prime[:len(d_pos)] % d_pos != 0)).any()

    return labels, vec, vec_prime, d, nontrivial

class varArraySolutionObtainer(cp_model.CpSolverSolutionCallback):
    """A class representing a solution callback for the solver.

    Args:
        cp_model (list): list of variables defined in your model.
    """

    def __init__(self, variables: cp_model.IntVar) -> None:
        """Initiate a solution initial state.

        Args:
            variables (ortools.sat.python.cp_model.IntVar): values of the variables.
        """
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solutionCount = 0
        self.__x = []

    def on_solution_callback(self) -> None:
        """Save the current solution in a list and add `1` to the solution count."""
        self.__solutionCount += 1
        self.__x.append([self.Value(v) for v in self.__variables])  # type:ignore

    def solutionCount(self) -> int:
        """Returns the total number of solutions.

        Returns:
            int: Total number of solutions
        """
        return self.__solutionCount

    def solutions(self) -> np.ndarray:
        """Returns all solutions of the model in a matrix form, where each row is a particular solution.

        Returns:
            numpy.ndarray: Numpy array which rows represent a solution to the model.
        """
        return np.array(self.__x)
    
    def n_smallest_solutions(self, n=5):
        solutions = [(x, np.linalg.norm(x)) for x in self.__x]
        solutions = sorted(solutions, key=lambda x: x[1])

        return [x[0] for x in solutions[:n]]


def compute_ebr_decomposition(ebr_data, vec):
    """
    Compute the decomposition of the symmetry vector into EBRs

    Parameters
    ----------
    ebr_data : dict
        Dictionary with EBR data as save in the package files.
    vec : array
        symmetry vector
    """

    def get_solutions(bounds=(0,15), n_smallest=5):
        """
        Solve the decomposition problem with some coefficient bounds and return
        some solutions, starting by the combinations with smallest coefficients

        Parameters
        ----------
        bounds : tuple, optional
            bounds for the coefficients, by default (0,15)
        n_smallest : int, optional
            how many solutions to, by default 5

        Returns
        -------
        list    
            list of solutions in form of lists of integers
        """
        lb, ub = bounds
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()
        x = [model.NewIntVar(lb, ub, f"x{i}") for i in range(n_ebr)]

        solution_obtainer = varArraySolutionObtainer(x)

        for i in range(n_ir):
            model.Add(A[i] @ x == vec[i])

        solver.SearchForAllSolutions(model, solution_obtainer)

        return solution_obtainer.n_smallest_solutions(n_smallest), solver.status_name()

    A = get_ebr_matrix(ebr_data)

    n_ir, n_ebr = A.shape

    is_positive = True
    # check positive coefficients only
    solutions, status = get_solutions()

    # if positive solutions were not found
    if status not in ["OPTIMAL", "FEASIBLE"]:
        is_positive = False
        
        # try with negative solutions
        solutions, status = get_solutions(bounds=(-15,15))

        # if negative solutions are not found, something's wrong
        if status not in ["OPTIMAL", "FEASIBLE"]:
            return None, is_positive
        # else return negative + positive solutions
        else:
            return solutions, is_positive
    else:
        return solutions, is_positive


def compose_irrep_string(labels):
    """
    Creates a string with the direct sum of irreps from a list of (repeated)
    irrep labels

    Parameters
    ----------
    labels : list
        list of irrep labels

    Returns
    -------
    str
        string with direct sum of irreps with multiplicities.
    """
    irrep_dict = {}

    for name in labels:
        value = irrep_dict.setdefault(name, 0)
        irrep_dict[name] = value + 1

    s = ""
    for i, (name, multip) in enumerate(irrep_dict.items()):
        s += f"{multip} {name} + "
        # carriage return when the line is too long
        if(i + 1) % 7 == 0:
            s += "\n"
    return s[:-2]

def compose_ebr_string(vec, ebrs):
    """
    Create a string with the direct sum of EBRS from a decomposition vector

    Parameters
    ----------
    vec : array
        Vector with coefficients for each irrep
    ebrs : list
        List of tuples with EBR label and Wyckoff position

    Returns
    -------
    str
        string representing the EBR decomposition in readable form
    """
    s = ""
    for ebr, multip in zip(ebrs, vec):
        label, wp = ebr
        s += f"{multip} [ {label} @ {wp} ] + "

    return s[:-2]

def show_ebr_decomposition(irrep_output, spinor):
    """
    Main function that performs all the EBR decomposition output

    Parameters
    ----------
    irrep_output : dict
        json_data from Irrep after irrep identification
    spinor : bool
        whether the bands are spinorial or not
    """
    def print_symmetry_info():
        """
        Print common EBR analysis information
        """
        basis = ebr_data["basis"]["irrep_labels"]
        print(
            f"Irrep decomposition at high-symmetry points:\n\n{compose_irrep_string(labels)}"
            f"\n\nIrrep basis:\n{vector_pprint(basis, fmt='s')}"
            f"\n\nSymmetry vector:\n{vector_pprint(vec, fmt='d')}"
            f"\n\nSmith singular values:\n{vector_pprint(d, fmt='d')}"
        )


    sg = irrep_output["spacegroup"]

    ebr_data = load_ebr_data(sg, "double" if spinor else "single")

    labels, vec, _, d, nontrivial = compute_topological_classification_vector(irrep_output, ebr_data)

    print("\n\n############### EBR decomposition ###############\n\n")

    if nontrivial:
        print("The set of bands is classified as TOPOLOGICAL\n")
        print_symmetry_info() 
    elif not ORTOOLS_AVAILABLE:
        print(
            "There exists integer-valued solutions to the EBR decomposition "
            "problem. Install OR-Tools to compute them."
            )
        print_symmetry_info()
    else:
        solutions, is_positive = compute_ebr_decomposition(ebr_data, vec)

        if is_positive:
            print("The set of bands is toplogically TRIVIAL")
            print_symmetry_info()
            print(
                "The following are positive, integer-valued linear combinations "
                "of EBRs that reproduce the set of bands:"
                )
        else:
            print("The set of bands displays FRAGILE TOPOLOGY.")
            print_symmetry_info()
            print(
                "There are no all-positive, integer-valued linear combinations "
                "of EBRs that reproduce the bands. Some possibilities are:"
                )

        ebrs = get_ebr_names_and_positions(ebr_data)
        for sol in solutions:
            print(compose_ebr_string(sol, ebrs))



        