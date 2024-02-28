import numpy as np
import os

def is_inversion(sym):
    return (
        np.isclose(sym.translation, 0.0).all() and
        np.isclose(sym.rotation + np.eye(3), 0.0).all()
    )

def is_trim(kp):
    return (-kp.K % 1 == kp.K).all()

def is_C2(sym, axis, t=(0,0,0)):
    c2 = -np.eye(3)
    c2[axis, axis] = 1

    return (
        np.isclose(sym.rotation, c2).all() and
        np.isclose(sym.translation, t).all()
        )

def is_mirror(sym, axis, t=(0,0,0)):
    m = np.eye(3)
    m[axis, axis] = -1
    
    return (
        np.isclose(sym.rotation, m).all() and
        np.isclose(sym.translation, t).all()
    )

def is_C4z(sym, t=(0,0,0)):
    c4z = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])

    return (
        np.isclose(sym.rotation, c4z).all() and
        np.isclose(sym.translation, t).all()
    )

def is_S4z(sym, t=(0,0,0)):
    s4z = -np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ])

    return (
        np.isclose(sym.rotation, s4z).all() and
        np.isclose(sym.translation, t).all()
    )

def is_C3z(sym, t=(0,0,0)):
    return (
        np.isclose(sym.axis, (0,0,1)).all() and
        np.isclose(sym.angle, 2 * np.pi / 3) and
        np.isclose(sym.translation, t).all()
    )

def is_C6z(sym, t=(0,0,0)):
    return (
        np.isclose(sym.axis, (0,0,1)).all() and
        np.isclose(sym.angle, 2 * np.pi / 6) and
        np.isclose(sym.translation, t).all()
    )

TRIM_POINTS = np.array([
        [0. , 0. , 0. ],
        [0.5, 0. , 0. ],
        [0. , 0.5, 0. ],
        [0. , 0. , 0.5],
        [0.5, 0.5, 0. ],
        [0.5, 0. , 0.5],
        [0. , 0.5, 0.5],
        [0.5, 0.5, 0.5]]
)

def count_states(kpoints, index_points, op_filter, eigs, occ, calc_points=None, **op_args):
    if calc_points is None:
        calc_points = np.array([kp.K for kp in kpoints])

    totals = np.zeros(len(eigs), dtype=int)
    for q in index_points:
            loc = np.where(np.isclose(calc_points, q).all(1))[0]
            if len(loc) == 0:
                raise Exception(f"{q=} was not found in the calculation.")
            kp = kpoints[loc[0]]
            for sym in kp.symmetries.keys():
                if op_filter(sym, **op_args):
                    op = sym
            op_vals = kp.symmetries[op]

            for i, eig in enumerate(eigs):
                totals[i] += np.isclose(op_vals[:occ], eig).sum()

    return totals

def filter_count_states(
        kpoints,
        index_points,
        op_filter,
        eigs_filter,
        op_count,
        eigs_count,
        occ,
        calc_points=None,
        args_filter={},
        args_count={}
):
    if calc_points is None:
        calc_points = np.array([kp.K for kp in kpoints])

    totals = np.zeros(len(eigs_count), dtype=int)
    for q in index_points:
            loc = np.where(np.isclose(calc_points, q).all(1))[0]
            if len(loc) == 0:
                raise Exception(f"{q=} was not found in the calculation.")
            kp = kpoints[loc[0]]
            for sym in kp.symmetries.keys():
                if op_filter(sym, **args_filter):
                    sym_filter = sym
                if op_count(sym, **args_count):
                    sym_count = sym
            
            filter_vals = kp.symmetries[sym_filter][:occ]
            count_vals = kp.symmetries[sym_count][:occ]

            for i, (fval, cval) in enumerate(zip(eigs_filter, eigs_count)):
                filter_mask = np.isclose(filter_vals, fval)
                totals[i] += np.isclose(count_vals, cval)[filter_mask].sum()

    return totals

################ INDICATORS #################

#############################################

def eta4I_2_4(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    trims = list(filter(is_trim, kpoints))
    assert len(trims) == 8, "The number of TRIMs is not 8"
    
    total = count_states(
        trims,
        TRIM_POINTS,
        is_inversion,
        [-1],
        occ
    )[0]
    return total % 4

#############################################

def z2Ii_2_4(kpoints, occ, i):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    trims = list(filter(is_trim, kpoints))
    assert len(trims) == 8, "The number of TRIMs is not 8"

    index_points = np.isclose(TRIM_POINTS[:,i], 0.5)
    index_points = TRIM_POINTS[index_points]

    total = count_states(
        trims,
        TRIM_POINTS,
        is_inversion,
        [-1],
        occ
    )[0]

    return total % 2

def z2Itriplet_2_4(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    indices = []
    for i in range(3):
        indices.append(z2Ii_2_4(kpoints, occ, i))
    return tuple(indices)

#############################################

def etaprime2I_2_4(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    return int(eta4I_2_4(kpoints, occ) / 2) % 2
        
#############################################

def z2R_3_1(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points = np.array([
        [0,   0.5, 0    ], # Z
        [0,   0.5, 0.5  ], # D
        [0.5, 0.5, 0    ], # C
        [0.5, 0.5, 0.5  ] # E
    ])

    total = count_states(
        kpoints,
        index_points,
        is_C2,
        [-1j],
        occ,
        axis=1
    )[0]
    
    return total % 2
        
                  
            
def z2R_41_215(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    total = count_states(
        kpoints,
        np.array([[0,0,0]]),
        is_C2,
        [-1j],
        occ,
        axis=2
    )[0]

    return total % 2

#############################################

def delta2m_10_42(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points1 = np.array([
        [0,   0.5, 0    ], # Z
        [0,   0.5, 0.5  ], # D
        [0.5, 0.5, 0    ], # C
        [0.5, 0.5, 0.5  ] # E
    ])
    index_points2 = np.array([
        [0,   0, 0    ], # GM
        [0.5, 0, 0.5  ], # A
        [0,   0, 0.5  ], # B
        [0.5, 0, 0    ] # Y
    ])
    calc_points = np.array([kp.K for kp in kpoints])
    
    counts1 = filter_count_states(
        kpoints,
        index_points1,
        is_C2,
        [np.exp(-1j * np.pi * 0.5)],
        is_mirror,
        [1j],
        occ,
        calc_points=calc_points,
        args_filter={"axis": 1},
        args_count={"axis": 1}
    )[0]

    counts2 = filter_count_states(
        kpoints,
        index_points2,
        is_C2,
        [np.exp(-1j * np.pi * 0.5)],
        is_mirror,
        [-1j],
        occ,
        calc_points=calc_points,
        args_filter={"axis": 1},
        args_count={"axis": 1}
    )[0]

    return (counts1 - counts2) % 2
        
#############################################

def z2mpiplus_10_42(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points = np.array([
        [0,   0.5, 0    ], # Z
        [0,   0.5, 0.5  ], # D
        [0.5, 0.5, 0    ], # C
        [0.5, 0.5, 0.5  ] # E
    ])
    total = filter_count_states(
        kpoints,
        index_points,
        is_C2,
        [np.exp(-1j * np.pi * 0.5)],
        is_mirror,
        [1j],
        occ,
        args_filter={"axis": 1},
        args_count={"axis": 1}
    )[0]
    return total % 2

#############################################

def z2mpiminus_10_42(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points = np.array([
        [0,   0.5, 0    ], # Z
        [0,   0.5, 0.5  ], # D
        [0.5, 0.5, 0    ], # C
        [0.5, 0.5, 0.5  ] # E
    ])
    total = filter_count_states(
        kpoints,
        index_points,
        is_C2,
        [np.exp(-1j * np.pi * 0.5)],
        is_mirror,
        [-1j],
        occ,
        args_filter={"axis": 1},
        args_count={"axis": 1}
    )[0]
    return total % 2

#############################################

def z4_2_5_47_249_83_45(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    trims = list(filter(is_trim, kpoints))
    assert len(trims) == 8, "The number of TRIMs is not 8"

    totals = count_states(
        trims,
        TRIM_POINTS,
        is_inversion,
        [-1, 1],
        occ
    )

    return int(0.25 * (totals[0] - totals[1])) % 4

#############################################

def z22i_2_5_47_249_83_45(kpoints, occ, i):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    trims = list(filter(is_trim, kpoints))
    assert len(trims) == 8, "The number of TRIMs is not 8"

    index_points = np.isclose(TRIM_POINTS[:,i], 0.5)
    index_points = TRIM_POINTS[index_points]

    total = count_states(
        trims,
        index_points,
        is_inversion,
        [-1],
        occ
    )[0]

    return int(0.5 * total) % 2

def z2Itriplet_2_5_47_249_83_45(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    indices = []
    for i in range(3):
        indices.append(z22i_2_5_47_249_83_45(kpoints, occ, i))
    return tuple(indices)

#############################################

def z4R_75_1(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points = np.array([
        [0,   0,   0.5], # Z
        [0.5, 0.5, 0.5] # A
    ])
    calc_points = np.array([kp.K for kp in kpoints])

    counts_ZA = count_states(
        kpoints,
        index_points,
        is_C4z,
        np.exp(-0.5j * np.pi * np.array([0.5, -0.5, 1.5, -1.5])),
        occ,
        calc_points=calc_points
    )

    counts_R = count_states(
        kpoints,
        np.array([[0.5, 0.5, 0.5]]),
        is_C2,
        np.exp(-1j * np.pi * np.array([0.5, -0.5])),
        occ,
        calc_points=calc_points,
        axis=2
    )

    return int(
        (np.array([-0.5, 0.5, -1.5, 1.5]) * counts_ZA).sum() +
            counts_R[0] - counts_R[1]
        ) % 4

#############################################

def zprime2R_77_13(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points = np.array([
        [0,   0,   0], # GM
        [0.5, 0.5, 0] # M
    ])
    calc_points = np.array([kp.K for kp in kpoints])

    counts_GMM = count_states(
        kpoints,
        index_points,
        is_C4z,
        np.exp(-0.5j * np.pi * np.array([0.5, -0.5, 1.5, -1.5])),
        occ,
        calc_points=calc_points,
        t=(0,0,0.5)
    )

    counts_X = count_states(
        kpoints,
        np.array([[0, 0.5, 0]]),
        is_C2,
        np.exp(-1j * np.pi * np.array([0.5, -0.5])),
        occ,
        calc_points=calc_points,
        axis=2
    )

    return int(
        (np.array([-0.25, 0.25, -0.75, 0.75]) * counts_GMM).sum() -
            0.5 * counts_X[0] + 0.5 * counts_X[1]
    ) % 2

def zprime2R_27_81_54_342_56_369(kpoints, occ, irreps=None): #### CHECK DOUBLETS
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points = np.array([
        [0,   0,   -0.5], # Z
        [0,   0.5, -0.5], # T
        [0.5, 0,   -0.5], # U
        [0.5, 0.5, -0.5]  # R
    ])
    
    total = count_states(
        kpoints,
        index_points,
        is_C2,
        [-1j],
        occ,
        axis=2
    )
    return total % 2

def zprime2R_60_424(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    etaprime =  etaprime2I_2_4(kpoints, occ)

    for kp in irreps:
        if kp["kpname"] == "GM":
            ir_occ = np.cumsum(kp["dimensions"]["data"])
            total = 0
            for n, ir in zip(ir_occ, kp["irreps"]):
                if n > occ:
                    break
                if "-GM3" in ir.keys():
                    total += 1

            return (etaprime + total) % 2

    raise Exception("Your calculation does not include the Gamma point")
                

def zprimeprime2R_110_249(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    for kp in irreps:
        if kp["kpname"] == "GM":
            ir_occ = np.cumsum(kp["dimensions"]["data"])
            total = 0
            for n, ir in zip(ir_occ, kp["irreps"]):
                if n > occ:
                    break
                if "-GM6" in ir.keys():
                    total += 1

            return total % 2

    raise Exception("Your calculation does not include the Gamma point")

#############################################

def z4S_81_33(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points = np.array([
        [0,   0,   0.5], # Z
        [0.5, 0.5, 0.5] # A
    ])
    calc_points = np.array([kp.K for kp in kpoints])

    counts_ZA = count_states(
        kpoints,
        index_points,
        is_S4z,
        np.exp(-0.5j * np.pi * np.array([0.5, -0.5, 1.5, -1.5])),
        occ,
        calc_points=calc_points
    )
    counts_R = count_states(
        kpoints,
        np.array([[0, 0.5, 0.5]]),
        is_C2,
        np.exp(-1j * np.pi * np.array([0.5, -0.5])),
        occ,
        calc_points=calc_points,
        axis=2
    )

    return int(
        (np.array([-0.5, 0.5, -1.5, 1.5]) * counts_ZA).sum() + 
            counts_R[0] - counts_R[1]
    ) % 4

#############################################

def delta2S_81_33(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points = np.array([
        [0,   0,   0.5], # Z
        [0.5, 0.5, 0.5], # A
        [0,   0 ,  0  ], # GM
        [0.5, 0.5, 0  ]  # M
    ])
    calc_points = np.array([kp.K for kp in kpoints])

    counts_ZA = count_states(
        kpoints,
        index_points[:2, :],
        is_S4z,
        np.exp(-0.5j * np.pi * np.array([1.5, -1.5])),
        occ,
        calc_points=calc_points
    )

    counts_GMM = count_states(
        kpoints,
        index_points[2:, :],
        is_S4z,
        np.exp(-0.5j * np.pi * np.array([1.5, -1.5])),
        occ,
        calc_points=calc_points
    )

    factors = np.array([-1, 1, -1, 1])

    return (
        factors * counts_ZA - factors * counts_GMM
    ).sum() % 2

#############################################

def z2_81_33(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points = np.array([
        [0,   0,   0.5], # Z
        [0.5, 0.5, 0.5], # A
        [0,   0 ,  0  ], # GM
        [0.5, 0.5, 0  ]  # M
    ])
    calc_points = np.array([kp.K for kp in kpoints])

    counts_C2 = count_states(
        kpoints,
        index_points,
        is_C2,
        [np.exp(-1j * np.pi * 0.5)],
        occ,
        calc_points=calc_points,
        axis=2

    )[0]

    counts_S4 = count_states(
        kpoints,
        index_points,
        is_S4z,
        [np.exp(-0.5j * np.pi * (-1.5))],
        occ,
        calc_points=calc_points,
    )[0]

    return int(
        (counts_C2 - counts_S4) * 0.5
    ) % 2

#############################################

def delta4m_83_43(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points1 = np.array([
        [0,   0,   0.5], # Z
        [0.5, 0.5, 0.5]  # A
    ])
    index_points2 = np.array([
        [0,   0,   0], # GM
        [0.5, 0.5, 0]  # M
    ])
    R = np.array([[0, 0.5, 0.5]])
    X = np.array([[0, 0.5, 0  ]])

    calc_points = np.array([kp.K for kp in kpoints])

    counts_ZA_C2 = filter_count_states(
        kpoints,
        index_points1,
        is_C2,
        np.exp(-1j * np.pi * np.array([0.5, -0.5])),
        is_mirror,
        np.array([1j, 1j]),
        occ,
        calc_points=calc_points,
        args_filter={"axis": 2},
        args_count={"axis": 2}
    )
    counts_ZA_C4 = filter_count_states(
        kpoints,
        index_points1,
        is_C4z,
        np.exp(-0.5j * np.pi * np.array([1.5, -1.5])),
        is_mirror,
        np.array([1j, 1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )
    counts_GMM_C2 = filter_count_states(
        kpoints,
        index_points2,
        is_C2,
        np.exp(-1j * np.pi * np.array([0.5, -0.5])),
        is_mirror,
        np.array([-1j, -1j]),
        occ,
        calc_points=calc_points,
        args_filter={"axis": 2},
        args_count={"axis": 2}
    )
    counts_GMM_C4 = filter_count_states(
        kpoints,
        index_points2,
        is_C4z,
        np.exp(-0.5j * np.pi * np.array([1.5, -1.5])),
        is_mirror,
        np.array([-1j, -1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )
    counts_R = filter_count_states(
        kpoints,
        R,
        is_C2,
        np.exp(-1j * np.pi * np.array([0.5, -0.5])),
        is_mirror,
        np.array([1j, 1j]),
        occ,
        calc_points=calc_points,
        args_filter={"axis": 2},
        args_count={"axis": 2}
    )
    counts_X = filter_count_states(
        kpoints,
        X,
        is_C2,
        np.exp(-1j * np.pi * np.array([0.5, -0.5])),
        is_mirror,
        np.array([-1j, -1j]),
        occ,
        calc_points=calc_points,
        args_filter={"axis": 2},
        args_count={"axis": 2}
    )

    counts_ZA = -(
        -0.5 * counts_ZA_C2[0] + 0.5 * counts_ZA_C2[1] - 1.5 * counts_ZA_C4[0] +
         1.5 * counts_ZA_C4[1]
    )
    counts_GMM = (
        -0.5 * counts_GMM_C2[0] + 0.5 * counts_GMM_C2[1] - 1.5 * counts_GMM_C4[0] +
         1.5 * counts_GMM_C4[1]
    )

    return int(
        counts_ZA + counts_GMM + counts_R[0] - counts_R[1] - counts_X[0] + counts_X[1]
    ) % 4

#############################################

def z4mpiplus_83_43(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points = np.array([
        [0,   0,   0.5], # Z
        [0.5, 0.5, 0.5]  # A
    ])
    R = np.array([[0, 0.5, 0.5]])

    calc_points = np.array([kp.K for kp in kpoints])

    counts_ZA_C2 = filter_count_states(
        kpoints,
        index_points,
        is_C2,
        np.exp(-1j * np.pi * np.array([0.5, -0.5])),
        is_mirror,
        np.array([1j, 1j]),
        occ,
        calc_points=calc_points,
        args_filter={"axis": 2},
        args_count={"axis": 2}
    )
    counts_ZA_C4 = filter_count_states(
        kpoints,
        index_points,
        is_C4z,
        np.exp(-0.5j * np.pi * np.array([1.5, -1.5])),
        is_mirror,
        np.array([1j, 1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )
    counts_R = filter_count_states(
        kpoints,
        R,
        is_C2,
        np.exp(-1j * np.pi * np.array([0.5, -0.5])),
        is_mirror,
        np.array([1j, 1j]),
        occ,
        calc_points=calc_points,
        args_filter={"axis": 2},
        args_count={"axis": 2}
    )

    counts_ZA = (
        -0.5 * counts_ZA_C2[0] + 0.5 * counts_ZA_C2[1] - 1.5 * counts_ZA_C4[0] +
         1.5 * counts_ZA_C4[1]
    )
    return int(
        counts_ZA + counts_R[0] - counts_R[1]
    )  % 4

#############################################

def z4mpiminus_83_43(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points = np.array([
        [0,   0,   0.5], # Z
        [0.5, 0.5, 0.5]  # A
    ])
    R = np.array([[0, 0.5, 0.5]])

    calc_points = np.array([kp.K for kp in kpoints])

    counts_ZA_C2 = filter_count_states(
        kpoints,
        index_points,
        is_C2,
        np.exp(-1j * np.pi * np.array([0.5, -0.5])),
        is_mirror,
        np.array([-1j,-1j]),
        occ,
        calc_points=calc_points,
        args_filter={"axis": 2},
        args_count={"axis": 2}
    )
    counts_ZA_C4 = filter_count_states(
        kpoints,
        index_points,
        is_C4z,
        np.exp(-0.5j * np.pi * np.array([1.5, -1.5])),
        is_mirror,
        np.array([-1j, -1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )
    counts_R = filter_count_states(
        kpoints,
        R,
        is_C2,
        np.exp(-1j * np.pi * np.array([0.5, -0.5])),
        is_mirror,
        np.array([-1j, -1j]),
        occ,
        calc_points=calc_points,
        args_filter={"axis": 2},
        args_count={"axis": 2}
    )

    counts_ZA = (
        -0.5 * counts_ZA_C2[0] + 0.5 * counts_ZA_C2[1] - 1.5 * counts_ZA_C4[0] +
         1.5 * counts_ZA_C4[1]
    )
    return int(
        counts_ZA + counts_R[0] - counts_R[1]
    )  % 4

#############################################

def z4m0plus_84_51(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points = np.array([
        [0,   0,   0    ], # GM
        [0.5, 0.5, 0    ], # M
    ])
    X = np.array([[0, 0.5, 0]])
    calc_points = np.array([kp.K for kp in kpoints])

    counts_GMM_C2 = filter_count_states(
        kpoints,
        index_points,
        is_C2,
        np.exp(-1j * np.pi * np.array([0.5, -0.5])),
        is_mirror,
        np.array([1j, 1j]),
        occ,
        calc_points=calc_points,
        args_filter={"axis": 2},
        args_count={"axis": 2}
    )
    counts_GMM_C4 = filter_count_states(
        kpoints,
        index_points,
        is_C4z,
        np.exp(-0.5j * np.pi * np.array([1.5, -1.5])),
        is_mirror,
        np.array([1j, 1j]),
        occ,
        calc_points=calc_points,
        args_filter={"t":(0,0,0.5)},
        args_count={"axis": 2}
    )
    counts_R = filter_count_states(
        kpoints,
        X,
        is_C2,
        np.exp(-1j * np.pi * np.array([0.5, -0.5])),
        is_mirror,
        np.array([1j, 1j]),
        occ,
        calc_points=calc_points,
        args_filter={"axis": 2},
        args_count={"axis": 2}
    )

    counts_GMM = (
        -0.5 * counts_GMM_C2[0] + 0.5 * counts_GMM_C2[1] - 1.5 * counts_GMM_C4[0] +
         1.5 * counts_GMM_C4[1]
    )

    return int(
        counts_GMM + counts_R[0] - counts_R[1]
    ) % 4

#############################################

def delta2m_84_51(kpoints, occ, irreps=None):
    return delta2m_10_42(kpoints, occ)

#############################################

def z8_83_44_123_339(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    def njpm(jval, ival):
        index_points1 = np.array([
            [0,   0,   0  ], # GM
            [0.5, 0.5, 0  ], # M
            [0,   0,   0.5], # Z
        ])
        index_points2 = np.array([
            [0, 0.5, 0  ], # X
            [0, 0.5, 0.5] # R
        ])

        counts1 = filter_count_states(
            kpoints,
            index_points1,
            is_C4z,
            [np.exp(-0.5j * np.pi * jval)],
            is_inversion,
            [ival],
            occ,
            calc_points=calc_points
        )[0]
        counts2 = filter_count_states(
            kpoints,
            index_points2,
            is_C2,
            [np.exp(-0.5j * np.pi * 0.5)], # correct?
            is_inversion,
            [ival],
            occ,
            calc_points=calc_points,
            args_filter={"axis": 2}
        )[0]

        return counts1 + counts2

    calc_points = np.array([kp.K for kp in kpoints])

    return int(
        1.5 * njpm(1.5, 1) - 1.5 * njpm(1.5, -1) - 0.5 * njpm(0.5, 1) + 0.5 * njpm(0.5, -1)
    ) % 8
    
#############################################

def z3R_147_13(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points = np.array([
        [ 0,     0,   0.5], # A
        [ 1/3,   1/3, 0.5], # H
        [-1/3, -1/3, 0.5 ] # HA
    ])

    counts = count_states(
        kpoints,
        index_points,
        is_C3z,
        np.exp(-2 * np.pi / 3 * np.array([-0.5, 1.5])),
        occ
    )

    return (
        counts[0] - counts[1]
    ) % 3
    
#############################################

def z6R_168_109(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    A = np.array([[0,   0,   0.5]])
    H = np.array([[1/3, 1/3, 0.5]])
    L = np.array([[0.5, 0 ,  0.5]])
    calc_points = np.array([kp.K for kp in kpoints])
    
    counts_A = count_states(
        kpoints,
        A,
        is_C6z,
        np.exp(-1j * np.pi / 3 * np.array([0.5, -0.5, 1.5, -1.5, 2.5, -2.5])),
        occ,
        calc_points=calc_points
    )
    counts_H = count_states(
        kpoints,
        H,
        is_C3z,
        np.exp(-2j * np.pi / 3 * np.array([0.5, -0.5, 1.5])),
        occ,
        calc_points=calc_points
    )
    counts_L = count_states(
        kpoints,
        L,
        is_C2,
        np.exp(-2j * np.pi / 3 * np.array([0.5, -0.5])),
        occ,
        calc_points=calc_points,
        axis=2
    )

    return int(
        (np.array(-0.5, 0.5, -1.5, 1.5, -2.5, 2.5) * counts_A +
         np.array(-1, 1) * counts_H + np.array(1.5, -1.5) * counts_L).sum()
    ) % 6

#############################################

def delta3m_174_133(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points1 = np.array([
        [ 0,    0,    0.5], # A
        [ 1/3,  1/3,  0.5], # H
        [-1/3, -1/3, -0.5]  # HA
    ])
    index_points2 = np.array([
        [ 0,    0,   0], # GM
        [ 1/3,  1/3, 0], # K
        [-1/3, -1/3, 0]  # KA
    ])
    calc_points = np.array([kp.K for kp in kpoints])

    counts1 = filter_count_states(
        kpoints,
        index_points1,
        is_C3z,
        np.exp(-2j * np.pi / 3 * np.array([-0.5, 1.5])),
        is_mirror,
        np.array([1j, 1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )
    counts2 = filter_count_states(
        kpoints,
        index_points2,
        is_C3z,
        np.exp(-2j * np.pi / 3 * np.array([-0.5, 1.5])),
        is_mirror,
        np.array([-1j, -1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )

    return (
        np.array([1, -1]) * counts1 - np.array([1, -1]) * counts2
    ).sum() % 3

#############################################

def z3mpiplus_174_133(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points = np.array([
        [ 0,    0,    0.5], # A
        [ 1/3,  1/3,  0.5], # H
        [-1/3, -1/3, -0.5]  # HA
    ])

    counts = filter_count_states(
        kpoints,
        index_points,
        is_C3z,
        np.exp(-2j * np.pi / 3 * np.array([-0.5, 1.5])),
        is_mirror,
        np.array([1j, 1j]),
        occ,
        args_count={"axis": 2}
    )

    return (
        counts[0] - counts[1]
    ) % 3

#############################################

def z3mpiminus_174_133(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    index_points = np.array([
        [ 0,    0,    0.5], # A
        [ 1/3,  1/3,  0.5], # H
        [-1/3, -1/3, -0.5]  # HA
    ])

    counts = filter_count_states(
        kpoints,
        index_points,
        is_C3z,
        np.exp(-2j * np.pi / 3 * np.array([-0.5, 1.5])),
        is_mirror,
        np.array([-1j, -1j]),
        occ,
        args_count={"axis": 2}
    )

    return (
        counts[0] - counts[1]
    ) % 3

#############################################

def delta6m_175_137(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    
    A =  np.array([[0, 0, 0.5]])
    H =  np.array([[1/3, 1/3, 0.5]])
    L =  np.array([[0.5, 0, 0.5]])
    GM = np.array([[0, 0, 0]])
    K =  np.array([[1/3, 1/3, 0]])
    M =  np.array([[0.5, 0, 0]])
    calc_points = np.array([kp.K for kp in kpoints])

    counts_A = filter_count_states(
        kpoints,
        A,
        is_C6z,
        np.exp(-1j * np.pi / 3 * np.array([0.5, -0.5, 1.5, -1.5, 2.5, -2.5])),
        is_mirror,
        np.array([1j, 1j, 1j, 1j, 1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )
    counts_H = filter_count_states(
        kpoints,
        H,
        is_C3z,
        np.exp(-2j * np.pi / 3 * np.array([0.5, -0.5, 1.5])),
        is_mirror,
        np.array([1j, 1j, 1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )
    counts_L = filter_count_states(
        kpoints,
        L,
        is_C2,
        np.exp(-0.5j * np.pi * np.array([0.5, -0.5])),
        is_mirror,
        np.array([1j, 1j]),
        occ,
        calc_points=calc_points,
        args_filter={"axis": 2},
        args_count={"axis": 2}
    )
    counts_GM = filter_count_states(
        kpoints,
        GM,
        is_C6z,
        np.exp(-1j * np.pi / 3 * np.array([0.5, -0.5, 1.5, -1.5, 2.5, -2.5])),
        is_mirror,
        np.array([-1j, -1j, -1j, -1j, -1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )
    counts_K = filter_count_states(
        kpoints,
        K,
        is_C3z,
        np.exp(-2j * np.pi / 3 * np.array([0.5, -0.5, 1.5])),
        is_mirror,
        np.array([-1j, -1j, -1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )
    counts_M = filter_count_states(
        kpoints,
        M,
        is_C2,
        np.exp(-0.5j * np.pi * np.array([0.5, -0.5])),
        is_mirror,
        np.array([-1j, -1j]),
        occ,
        calc_points=calc_points,
        args_filter={"axis": 2},
        args_count={"axis": 2}
    )

    return int(
        (np.array([-0.5, 0.5, -1.5, 1.5, -2.5, 2.5]) * counts_A).sum() +
        (np.array([-1, 1, 3]) * counts_H).sum() +
        (np.array([1.5, -1.5]) * counts_L).sum() +
        (np.array([0.5, -0.5, 1.5, -1.5]) * counts_GM).sum() +
        (np.array([1, -1, -3]) * counts_K).sum() +
        (np.array([-1.5, 1.5]) * counts_M).sum() 
    ) % 6
    
#############################################

def z6mpiplus_175_137(kpoints, occ, irreps=None):    
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    A =  np.array([[0, 0, 0.5]])
    H =  np.array([[1/3, 1/3, 0.5]])
    L =  np.array([[0.5, 0, 0.5]])
    calc_points = np.array([kp.K for kp in kpoints])

    counts_A = filter_count_states(
        kpoints,
        A,
        is_C6z,
        np.exp(-1j * np.pi / 3 * np.array([0.5, -0.5, 1.5, -1.5, 2.5, -2.5])),
        is_mirror,
        np.array([1j, 1j, 1j, 1j, 1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )
    counts_H = filter_count_states(
        kpoints,
        H,
        is_C3z,
        np.exp(-2j * np.pi / 3 * np.array([0.5, -0.5, 1.5])),
        is_mirror,
        np.array([1j, 1j, 1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )
    counts_L = filter_count_states(
        kpoints,
        L,
        is_C2,
        np.exp(-0.5j * np.pi * np.array([0.5, -0.5])),
        is_mirror,
        np.array([1j, 1j]),
        occ,
        calc_points=calc_points,
        args_filter={"axis": 2},
        args_count={"axis": 2}
    )

    return int(
        (np.array([-0.5, 0.5, -1.5, 1.5, -2.5, 2.5]) * counts_A).sum() +
        (np.array([-1, 1, 3]) * counts_H).sum() +
        (np.array([1.5, -1.5]) * counts_L).sum() 
    ) % 6

#############################################

def z6mpiminus_175_137(kpoints, occ, irreps=None):    
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    A =  np.array([[0, 0, 0.5]])
    H =  np.array([[1/3, 1/3, 0.5]])
    L =  np.array([[0.5, 0, 0.5]])
    calc_points = np.array([kp.K for kp in kpoints])

    counts_A = filter_count_states(
        kpoints,
        A,
        is_C6z,
        np.exp(-1j * np.pi / 3 * np.array([0.5, -0.5, 1.5, -1.5, 2.5, -2.5])),
        is_mirror,
        np.array([-1j, -1j, -1j, -1j, -1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )
    counts_H = filter_count_states(
        kpoints,
        H,
        is_C3z,
        np.exp(-2j * np.pi / 3 * np.array([0.5, -0.5, 1.5])),
        is_mirror,
        np.array([-1j, -1j, -1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )
    counts_L = filter_count_states(
        kpoints,
        L,
        is_C2,
        np.exp(-0.5j * np.pi * np.array([0.5, -0.5])),
        is_mirror,
        np.array([-1j, -1j]),
        occ,
        calc_points=calc_points,
        args_filter={"axis": 2},
        args_count={"axis": 2}
    )

    return int(
        (np.array([-0.5, 0.5, -1.5, 1.5, -2.5, 2.5]) * counts_A).sum() +
        (np.array([-1, 1, 3]) * counts_H).sum() +
        (np.array([1.5, -1.5]) * counts_L).sum() 
    ) % 6

#############################################

def z6m0plus_176_143(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    GM = np.array([[0, 0, 0]])
    K = np.array([[1/3, 1/3, 0]])
    M = np.array([[0.5, 0, 0]])
    calc_points = np.array([kp.K for kp in kpoints])

    counts_GM = filter_count_states(
        kpoints,
        GM,
        is_C6z,
        np.exp(-1j * np.pi / 3 * np.array([0.5, -0.5, 1.5, -1.5, 2.5, -2.5])),
        is_mirror,
        np.array([1j, 1j, 1j, 1j, 1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )
    counts_K = filter_count_states(
        kpoints,
        K,
        is_C3z,
        np.exp(-2j * np.pi / 3 * np.array([0.5, -0.5, 1.5])),
        is_mirror,
        np.array([1j, 1j, 1j]),
        occ,
        calc_points=calc_points,
        args_count={"axis": 2}
    )
    counts_M = filter_count_states(
        kpoints,
        M,
        is_C2,
        np.exp(-0.5j * np.pi * np.array([0.5, -0.5])),
        is_mirror,
        np.array([1j, 1j]),
        occ,
        calc_points=calc_points,
        args_filter={"axis": 2},
        args_count={"axis": 2}
    )

    return int(
        (np.array([-0.5, 0.5, -1.5, 1.5, -2.5, 2.5]) * counts_GM).sum() +
        (np.array([-1, 1, 3])* counts_K).sum() +
        (np.array([1.5, -1.5]) * counts_M).sum()
    ) % 6

#############################################

def z12_175_138_191_233(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"

    delta = delta6m_175_137(kpoints, occ)

    return delta + 3 * (delta - z4_2_5_47_249_83_45(kpoints, occ)) % 12

#############################################

def z12prime_176_144(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"

    z6 = z6m0plus_176_143(kpoints, occ)

    return(
        z6 + 3 * (z6 - z4_2_5_47_249_83_45(kpoints, occ)) % 12
    )

#############################################

def z4Rprime_103_199(kpoints, occ, irreps=None):
    index_points = np.array([
        [0,   0,   0.5], # Z
        [0.5, 0.5, 0.5]  # A
    ])
    R = np.array([[0, 0.5, 0.5]])
    calc_points = np.array([kp.K for kp in kpoints])

    counts_ZA = count_states(
        kpoints,
        index_points,
        is_C4z,
        np.exp(-0.5j * np.pi * np.array([0.5, -0.5, 1.5, -1.5])),
        occ,
        calc_points=calc_points
    )
    counts_R = count_states(
        kpoints,
        R,
        is_C2,
        np.exp(-1j * np.pi * np.array(0.5, -0.5)),
        occ,
        calc_points=calc_points,
        axis=2
    )

    return int(
        (np.array([-0.25, 0.25, -0.75, 0.75]) * counts_ZA).sum() + 
        0.5 * counts_R[0] - 0.5 * counts_R[1]
    ) % 4

#############################################

def z4prime_135_487(kpoints, occ, irreps=None):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    found_GM = False
    found_M = False
    found_X = False
    for kp in irreps:
        if kp["kpname"] == "GM":
            found_GM = True
            ir_occ = np.cumsum(kp["dimensions"]["data"])
            total_GM5 = 0
            total_GM6 = 0
            for n, ir in zip(ir_occ, kp["irreps"]):
                if n > occ:
                    break
                if "-GM5" in ir.keys():
                    total_GM5 += 1
                elif "-GM6" in ir.keys():
                    total_GM6 += 1
        if kp["kpname"] == "M":
            found_M = True
            ir_occ = np.cumsum(kp["dimensions"]["data"])
            total_M5 = 0
            for n, ir in zip(ir_occ, kp["irreps"]):
                if n > occ:
                    break
                if "-M5" in ir.keys():
                    total_M5 += 1
        if kp["kpname"] == "X":
            found_X = True
            ir_occ = np.cumsum(kp["dimensions"]["data"])
            total_X3 = 0
            for n, ir in zip(ir_occ, kp["irreps"]):
                if n > occ:
                    break
                if "-X3" in ir.keys():
                    total_X3 += 1

    if found_GM and found_M and found_X:
        return (
            2 * total_GM5 - total_GM6 - total_M5 + 2 * total_X3
        )
    else:    
        raise Exception("Your calculation does not include all the necessary "
                        "points: GM, M & X")
#############################################

def z4Rprime_184_195(kpoints, occ, irreps=None):    
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    A =  np.array([[0, 0, 0.5]])
    H =  np.array([[1/3, 1/3, 0.5]])
    L =  np.array([[0.5, 0, 0.5]])
    calc_points = np.array([kp.K for kp in kpoints])

    counts_A =count_states(
        kpoints,
        A,
        is_C6z,
        np.exp(-1j * np.pi / 3 * np.array([0.5, -0.5, 1.5, -1.5, 2.5, -2.5])),
        occ,
        calc_points=calc_points
    )
    counts_H = count_states(
        kpoints,
        H,
        is_C3z,
        np.exp(-2j * np.pi / 3 * np.array([0.5, -0.5, 1.5])),
        occ,
        calc_points=calc_points
    )
    counts_L = count_states(
        kpoints,
        L,
        is_C2,
        np.exp(-0.5j * np.pi * np.array([0.5, -0.5])),
        occ,
        calc_points=calc_points,
        axis=2
    )

    return int(
        (np.array([-0.5, 0.5, -1.5, 1.5, -2.5, 2.5]) * counts_A).sum() +
        (np.array([-1, 1, 3]) * counts_H).sum() +
        (np.array([1.5, -1.5]) * counts_L).sum() 
    ) % 6



#############################################



SSG_INDICATORS = {
    "2.4": [
        ("η₄ᵢ", eta4I_2_4),
        ("z₂ᵢ", z2Itriplet_2_4),
        ("η'₄ᵢ", etaprime2I_2_4)
        ],
    "2.5" : [
        ("z₄", z4_2_5_47_249_83_45),
        ("z₂ᵢ", z2Itriplet_2_5_47_249_83_45)
        ],
    "3.1": [
        ("z₂ᵣ", z2R_3_1)
        ],
    "41.215": [
        ("z₂ᵣ", z2R_41_215)
        ],
    "10.42": [
        ("δ₂ₘ", delta2m_10_42),
        ("z₂ₘπ+", z2mpiplus_10_42),
        ("z₂ₘπ-", z2mpiminus_10_42)
        ],
    "27.81": [
        ("z'₂ᵣ", zprime2R_27_81_54_342_56_369)
        ],
    "47.249": [
        ("z₄", z4_2_5_47_249_83_45),
        ("z₂ᵢ", z2Itriplet_2_5_47_249_83_45)
        ],
    "75.1": [
        ("z₄ᵣ", z4R_75_1)
        ],
    "54.342": [
        ("z'₂ᵣ", zprime2R_27_81_54_342_56_369)
        ],
    "56.369": [
        ("z'₂ᵣ", zprime2R_27_81_54_342_56_369)
        ],
    "60.424": [
        ("z'₂ᵣ", zprime2R_60_424)
        ],
    "77.13": [
        ("z'₂ᵣ", zprime2R_77_13)
        ],
    "81.33": [
        ("z₄ₛ", z4S_81_33),
        ("δ₂ₛ", delta2S_81_33, z2_81_33)
        ],
    "83.43": [
        ("δ₄ₘ", delta4m_83_43),
        ("z₄ₘπ+", z4mpiplus_83_43),
        ("z₄ₘπ-", z4mpiminus_83_43)
        ],
    "83.44": [
        ("z₈", z8_83_44_123_339)
        ],
    "83.45": [
        ("z₄", z4_2_5_47_249_83_45),
        ("z₂ᵢ", z2Itriplet_2_5_47_249_83_45)
        ],
    "84.51": [
        ("z₄ₘ₀+", z4m0plus_84_51),
        ("δ₂ₘ", delta2m_84_51)
        ],
    "103.199": [
        ("z'₄ᵣ", z4Rprime_103_199)
        ],
    "110.249": [
        ("z''", zprimeprime2R_110_249)
        ],
    "123.339": [
        ("z₈", z8_83_44_123_339)
        ],
    "135.487": [
        ("z'₄", z4prime_135_487)
        ],
    "147.13": [
        ("z₃ᵣ", z3R_147_13)
        ],
    "168.109": [
        ("z₆ᵣ", z6R_168_109)
        ],
    "174.133": [
        ("δ₃ₘ", delta3m_174_133),
        ("z₃ₘπ+", z3mpiplus_174_133),
        ("z₃ₘπ+", z3mpiminus_174_133)
        ],
    "175.137": [
        ("δ₆ₘ", delta6m_175_137),
        ("z₆ₘπ+", z6mpiplus_175_137),
        ("z₆ₘπ-", z6mpiminus_175_137)
        ],
    "175.138": [
        ("z₁₂", z12_175_138_191_233)
        ],
    "176.143": [
        ("z₆ₘ₀+", z6m0plus_176_143)
        ],
    "176.144": [
        ("z'₁₂", z12prime_176_144)
        ],
    "184.195": [
        ("z'₄ᵣ", z4Rprime_184_195)
        ],
    "191.233": [
        ("z₁₂", z12_175_138_191_233)
        ]
}

def get_si_from_ssg(sg_number):
    root = os.path.dirname(__file__)
    min_sg_number = 0
    with open(root + "/minimal_ssg.data") as f:
        for line in f:
            ssg, minimal = line.strip().split(",")
            if ssg == sg_number:
                min_sg_number = minimal
                break
    
    if min_sg_number == 0:
        raise KeyError(f"Could not find the minimal group corresponding to {sg_number}")
    else:
        return SSG_INDICATORS[min_sg_number]