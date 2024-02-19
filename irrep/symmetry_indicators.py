import numpy as np

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

def eta4I_2_4(kpoints, occ):
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

def z2Itriplet_2_4(kpoints, occ):
    indices = []
    for i in range(3):
        indices.append(z2Ii_2_4(kpoints, occ, i))
    return tuple(indices)

#############################################

def etaprime2I_2_4(kpoints, occ):
    return int(eta4I_2_4(kpoints, occ) / 2) % 2
        
#############################################

def z2R_3_1(kpoints, occ):
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
        
                  
            
def z2R_41_215(kpoints, occ):
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

def delta2m_10_42(kpoints, occ):
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

def z2mpiplus_10_42(kpoints, occ):
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

def z2mpiminus_10_42(kpoints, occ):
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

def z4_2_5_47_249_83_45(kpoints, occ):
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

def z22i_2_5_47_249_84_45(kpoints, occ, i):
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

def z2Itriplet_2_5_47_249_84_45(kpoints, occ):
    indices = []
    for i in range(3):
        indices.append(z22i_2_5_47_249_84_45(kpoints, occ, i))
    return tuple(indices)

#############################################

def z4R_75_1(kpoints, occ):
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

def zprime2R_77_13(kpoints, occ):
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

def zprime2R_27_81_54_342_56_369(kpoints, occ): #### CHECK DOUBLETS
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

def zprime2R_60_424(kpoints, occ):
     raise NotImplementedError("Indicator not implemented.")
     return etaprime2I_2_4(kpoints, occ) # TODO ADD m(GM3)

def zprimeprime2R_110_249(kpoints, occ):
    raise NotImplementedError("Indicator not implemented.")

#############################################

def z4S_81_33(kpoints, occ):
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

def delta2S_81_33(kpoints, occ):
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

def z2_81_33(kpoints, occ):
    index_points = np.array([
        [0,   0,   0.5], # Z
        [0.5, 0.5, 0.5], # A
        [0,   0 ,  0  ], # GM
        [0.5, 0.5, 0  ]  # M
    ])
    calc_points = np.array([kp.K for kp in kpoints])

    phases = np.array([-0.25, 0.75])
    j_vals = np.exp(phases * 1j* np.pi)

    total = 0
    for q in index_points:
        loc = np.where(np.isclose(calc_points, q).all(1))[0]
        if len(loc) == 0:
            raise Exception(f"{q=} was not found in the calculation.")
        kp = kpoints[loc[0]]
        for sym in kp.symmetries.keys():
            if is_S4z(sym):
                s4z = sym
        s4z_vals = kp.symmetries[s4z]
        for sign, j_val in zip([1, -1], j_vals):
            j_mask = np.isclose(s4z_vals, j_val)
            total += sign * (j_mask[:occ]).sum()
    

    return int(total / 2) % 2

#############################################

def z4m0plus_84_51(kpoints, occ):
    index_points = np.array([
        [0,   0,   0    ], # GM
        [0.5, 0.5, 0    ], # M
    ])
    calc_points = np.array([kp.K for kp in kpoints])

    phases = np.array([-0.25, 0.25, -0.75, 0.75])
    j_vals = np.exp(1j * np.pi * phases)
    total = 0
    for q in index_points:
        loc = np.where(np.isclose(calc_points, q).all(1))[0]
        if len(loc) == 0:
            raise Exception(f"{q=} was not found in the calculation.")
        kp = kpoints[loc[0]]
        for sym in kp.symmetries.keys():
            if is_C4z(sym, t=(0,0,0.5)):
                c4z = sym
            elif is_mirror(sym, 2):
                mz = sym
        c4z_vals = kp.symmetries[c4z]
        mz_vals = kp.symmetries[mz].imag.round()

        for phase, j_val in zip(phases, j_vals):
            j_mask = np.isclose(c4z_vals, j_val)
            total += phase * (mz_vals[:occ][j_mask] == 1).sum()

    loc = np.where(np.isclose(calc_points, [0, 0.5, 0]).all(1))[0]
    if len(loc) == 0:
        raise Exception("X=(0,1/2,0) was not found in the calculation.")
    kp = kpoints[loc[0]]
    for sym in kp.symmetries.keys():
        if is_C4z(sym, t=(0,0,0.5)):
            c4z = sym
        elif is_mirror(sym, 2):
            mz = sym
    c4z_vals = kp.symmetries[c4z]
    mz_vals = kp.symmetries[mz].imag.round()

    j_mask = np.isclose(c4z_vals, np.exp(-0.25j * np.pi))
    total += (mz_vals[:occ][j_mask] == 1).sum()

    j_mask = np.isclose(c4z_vals, np.exp(0.25j * np.pi))
    total -= (mz_vals[:occ][j_mask] == 1).sum()

    return total % 4

def delta2m_84_51(kpoints, occ):
    return delta2m_10_42(kpoints, occ)

#############################################

def z8_83_44_123_339(kpoints, occ):
    def njpm(jvals, ivals):

        index_points1 = np.array([
            [0,   0,   0  ], # GM
            [0.5, 0.5, 0  ], # M
            [0,   0,   0.5], # Z
        ])
        index_points2 = np.array([
            [0, 0.5, 0  ], # X
            [0, 0.5, 0.5] # R
        ])
        jvals = np.array(jvals)
        ivals = np.array(ivals)
        totals = np.zeros(len(jvals))
        jvals = np.zeros(-2j * np.pi * jvals * 0.25)
        for q in index_points1:
            loc = np.where(np.isclose(calc_points, q).all(1))[0]
            if len(loc) == 0:
                raise Exception(f"{q=} was not found in the calculation.")
            kp = kpoints[loc[0]]
            for sym in kp.symmetries.keys():
                if is_C4z(sym):
                    c4z = sym
                elif is_inversion(sym):
                    inv = sym
            c4z_vals = kp.symmetries[c4z]
            inv_vals = kp.symmetries[inv].real.round()

            for i, (jval, ival) in enumerate(zip(jvals, ivals)):
                j_mask = np.isclose(c4z_vals, jval)
                totals[i] += (inv_vals[j_mask][:occ] == ival).sum()
        
        for q in index_points2:
            loc = np.where(np.isclose(calc_points, q).all(1))[0]
            if len(loc) == 0:
                raise Exception(f"{q=} was not found in the calculation.")
            kp = kpoints[loc[0]]
            for sym in kp.symmetries.keys():
                if is_C2(sym, 2):
                    c2z = sym
                elif is_inversion(sym):
                    inv = sym
            c2z_vals = kp.symmetries[c2z]
            inv_vals = kp.symmetries[inv].real.round()

            jval = np.exp(-0.25j * np.pi)
            for i in enumerate(ivals):
                j_mask = np.isclose(c2z_vals, jval)
                totals[i] += (inv_vals[j_mask][:occ] == ival).sum()

        return totals

    calc_points = np.array([kp.K for kp in kpoints])

    counts = njpm([3/2, 3/2, 1/2, 1/2], [1, -1, 1, -1])

    return int(3 * counts[0] / 2 - 3 * counts[1] / 2 - 0.5 * counts[2] + 0.5 * counts[3]) % 8
    
#############################################

def z3R_147_13(kpoints, occ):
    index_points = np.array([
        [ 0,     0,   0.5], # A
        [ 1/3,   1/3, 0.5], # H
        [-1/3, -1/3, 0.5 ] # HA
    ])
    calc_points = np.array([kp.K for kp in kpoints])

    jvals = np.exp(
        -2j * np.pi * np.array([-0.5, 3/2]) / 3
    )
    total = 0
    for q in index_points:
            loc = np.where(np.isclose(calc_points, q).all(1))[0]
            if len(loc) == 0:
                raise Exception(f"{q=} was not found in the calculation.")
            kp = kpoints[loc[0]]
            for sym in kp.symmetries.keys():
                if is_C3z(sym):
                    c3z = sym
            c3z_vals = kp.symmetries[c3z]

            for sign, jval in zip([1, -1], jvals):
                j_mask = np.isclose(c3z_vals, jval)
                total += sign * (j_mask[:occ]).sum()

    return total % 3
    
#############################################

def z6R_168_109(kpoints, occ):
    index_points = np.array([
        [0,   0,   0.5], # A
        [1/3, 1/3, 0.5], # H
        [0.5, 0,   0.5] # L
    ])

