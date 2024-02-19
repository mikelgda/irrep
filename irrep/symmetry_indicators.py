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

################ INDICATORS #################

#############################################

def eta4I_2_4(kpoints, occ):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    trims = list(filter(is_trim, kpoints))
    assert len(trims) == 8, "The number of TRIMs is not 8"
    total = 0
    for kp in trims:
        for sym in kp.symmetries.keys():
            if is_inversion(sym):
                inv = sym
                break
        inv_vals = kp.symmetries[inv].real.round()
        total += (inv_vals[:occ] == -1).sum()
    return total % 4

#############################################

def z2Ii_2_4(kpoints, occ, i):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    trims = list(filter(is_trim, kpoints))
    assert len(trims) == 8, "The number of TRIMs is not 8"

    total = 0
    for kp in trims:
        if not np.isclose(kp.K[i], 0.5):
            continue
        for sym in kp.symmetries.keys():
            if is_inversion(sym):
                inv = sym
                break
        inv_vals = kp.symmetries[inv].real.round()
        total += (inv_vals[:occ] == -1).sum()

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
    calc_points = np.array([kp.K for kp in kpoints])
    total = 0
    for q in index_points:
        loc = np.where(np.isclose(calc_points, q).all(1))[0]
        if len(loc) == 0:
            raise Exception(f"{q=} was not found in the calculation but it is"
                             " needed for the z2R index of group 3.1.")
        kp = kpoints[loc[0]]
        for sym in kp.symmetries.keys():
            if is_C2(sym, 1):
                c2y = sym
                break
        c2y_vals = kp.symmetries[c2y].imag.round()
        total += (c2y_vals[:occ] == -1).sum()
    
    return total % 2
        
                  
            
def z2R_41_215(kpoints, occ):
    calc_points = np.array([kp.K for kp in kpoints])
    loc = np.where(np.isclose(calc_points, 0).all(1))[0]
    if len(loc) == 0:
        raise Exception("Gamma was not found in the calculation but it is"
                        " needed for the z2R index of group 41.215")
    kp = kpoints[loc[0]]
    for sym in kp.symmetries.keys():
        if is_C2(sym, 2):
            c2z = sym
            break
    c2z_vals = kp.symmetries[c2z].imag.round()
    return (c2z_vals[:occ] == -1).sum() % 2

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
    total = 0
    for val, qlist in zip([1,-1,],[index_points1, index_points2]):
        for q in qlist:
            loc = np.where(np.isclose(calc_points, q).all(1))[0]
            if len(loc) == 0:
                raise Exception(f"{q=} was not found in the calculation but it is"
                                " needed for the delta2m index of group 10.42.")
            kp = kpoints[loc[0]]
            for sym in kp.symmetries.keys():
                if is_C2(sym, 1):
                    c2y = sym
                elif is_mirror(sym, 1):
                    my = sym
            c2y_vals = kp.symmetries[c2y].imag.round()
            j_mask = c2y_vals == -1
            my_vals = kp.symmetries[my].imag.round()
            total += val * (my_vals[:occ][j_mask] == val).sum()

    return total % 2
        
#############################################

def z2mpiplus_10_42(kpoints, occ):
    index_points = np.array([
        [0,   0.5, 0    ], # Z
        [0,   0.5, 0.5  ], # D
        [0.5, 0.5, 0    ], # C
        [0.5, 0.5, 0.5  ] # E
    ])
    calc_points = np.array([kp.K for kp in kpoints])
    total = 0
    for q in index_points:
        loc = np.where(np.isclose(calc_points, q).all(1))[0]
        if len(loc) == 0:
            raise Exception(f"{q=} was not found in the calculation but it is"
                            " needed for the delta2m index of group 10.42.")
        kp = kpoints[loc[0]]
        for sym in kp.symmetries.keys():
            if is_C2(sym, 1):
                c2y = sym
            elif is_mirror(sym, 1):
                my = sym
        c2y_vals = kp.symmetries[c2y].imag.round()
        j_mask = c2y_vals == -1
        my_vals = kp.symmetries[my].imag.round()
        total += (my_vals[:occ][j_mask] == 1).sum()

    return total % 2

#############################################

def z2mpiminus_10_42(kpoints, occ):
    index_points = np.array([
        [0,   0.5, 0    ], # Z
        [0,   0.5, 0.5  ], # D
        [0.5, 0.5, 0    ], # C
        [0.5, 0.5, 0.5  ] # E
    ])
    calc_points = np.array([kp.K for kp in kpoints])
    total = 0
    for q in index_points:
        loc = np.where(np.isclose(calc_points, q).all(1))[0]
        if len(loc) == 0:
            raise Exception(f"{q=} was not found in the calculation but it is"
                            " needed for the delta2m index of group 10.42.")
        kp = kpoints[loc[0]]
        for sym in kp.symmetries.keys():
            if is_C2(sym, 1):
                c2y = sym
            elif is_mirror(sym, 1):
                my = sym
        c2y_vals = kp.symmetries[c2y].imag.round()
        j_mask = c2y_vals == -1
        my_vals = kp.symmetries[my].imag.round()
        total += (my_vals[:occ][j_mask] == -1).sum()

    return total % 2

#############################################

def z4_2_5_47_249_83_45(kpoints, occ):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    trims = list(filter(is_trim, kpoints))
    assert len(trims) == 8, "The number of TRIMs is not 8"
    total = 0
    for kp in trims:
        for sym in kp.symmetries.keys():
            if is_inversion(sym):
                inv = sym
                break
        inv_vals = kp.symmetries[inv].real.round()
        total += (inv_vals[:occ] == -1).sum() -(inv_vals[:occ] == 1).sum()
    return int(total / 4) % 4

#############################################

def z22i_2_5_47_249_84_45(kpoints, occ, i):
    assert kpoints[0].Energy.shape[0] >= occ, "Occupation is higher than computed bands"
    trims = list(filter(is_trim, kpoints))
    assert len(trims) == 8, "The number of TRIMs is not 8"

    total = 0
    for kp in trims:
        if not np.isclose(kp.K[i], 0.5):
            continue
        for sym in kp.symmetries.keys():
            if is_inversion(sym):
                inv = sym
                break
        inv_vals = kp.symmetries[inv].real.round()
        total += (inv_vals[:occ] == -1).sum()

    return int(total / 2) % 2

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

    phases = np.array([-0.25, 0.25, 0.75, -0.75])
    j_vals = np.exp(phases * 1j* np.pi)

    total = 0
    for q in index_points:
        loc = np.where(np.isclose(calc_points, q).all(1))[0]
        if len(loc) == 0:
            raise Exception(f"{q=} was not found in the calculation but it is"
                            " needed for the z4R index of group 75.1.")
        kp = kpoints[loc[0]]
        for sym in kp.symmetries.keys():
            if is_C4z(sym):
                c4z = sym
        c4z_vals = kp.symmetries[c4z]
        for phase, j_val in zip(phases, j_vals):
            j_mask = np.isclose(c4z_vals, j_val)
            total += phase * (j_mask[:occ]).sum()
    
    loc = np.where(np.isclose(calc_points, [0, 0.5, 0.5]).all(1))[0]
    if len(loc) == 0:
        raise Exception("R=(0, 1/2, 1/2) was not found in the calculation but" 
                        " it is needed for the z4R index of group 75.1.")
    kp = kpoints[loc[0]]
    for sym in kp.symmetries.keys():
        if is_C2(sym, 2):
            c2z = sym
            break
    c2z_vals = kp.symmetries[c2z].imag.round()
    total += (c2z_vals[:occ] == -1).sum() - (c2z_vals[:occ] == 1).sum()

    return total % 4

#############################################

def zprime2R_77_13(kpoints, occ):
    index_points = np.array([
        [0,   0,   0], # GM
        [0.5, 0.5, 0] # M
    ])
    calc_points = np.array([kp.K for kp in kpoints])

    phases = np.array([-0.25, 0.25, 0.75, -0.75])
    j_vals = np.exp(phases * 1j* np.pi)

    total = 0
    for q in index_points:
        loc = np.where(np.isclose(calc_points, q).all(1))[0]
        if len(loc) == 0:
            raise Exception(f"{q=} was not found in the calculation but it is"
                            " needed for the z4R index of group 75.1.")
        kp = kpoints[loc[0]]
        for sym in kp.symmetries.keys():
            if is_C4z(sym, t=(0,0,0.5)):
                c4z = sym
        c4z_vals = kp.symmetries[c4z]
        for phase, j_val in zip(phases, j_vals):
            j_mask = np.isclose(c4z_vals, j_val)
            total += phase * (j_mask[:occ]).sum()
    
    loc = np.where(np.isclose(calc_points, [0, 0.5, 0]).all(1))[0]
    if len(loc) == 0:
        raise Exception("X=(0, 1/2, 0) was not found in the calculation but" 
                        " it is needed for the z4R index of group 77.13.")
    kp = kpoints[loc[0]]
    for sym in kp.symmetries.keys():
        if is_C2(sym, 2):
            c2z = sym
    c2z_vals = kp.symmetries[c2z]
    total += (c2z_vals[:occ] == 1).sum() - (c2z_vals[:occ] == -1).sum()

    return (occ + total) % 2

def zprime2R_27_81_54_342_56_369(kpoints, occ): #### CHECK DOUBLETS
    index_points = np.array([
        [0,   0,   -0.5], # Z
        [0,   0.5, -0.5], # T
        [0.5, 0,   -0.5], # U
        [0.5, 0.5, -0.5]  # R
    ])
    calc_points = np.array([kp.K for kp in kpoints])
    
    total = 0
    for q in index_points:
        loc = np.where(np.isclose(calc_points, q).all(1))[0]
        if len(loc) == 0:
            raise Exception(f"{q=} was not found in the calculation but it is"
                            " needed for the z4R index of group 75.1.")
        kp = kpoints[loc[0]]
        for sym in kp.symmetries.keys():
            if is_C2(sym, 2):
                c2z = sym
                break
        c2z_vals = kp.symmetries[c2z]
        total += (c2z_vals[:occ] == -1).sum()

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

    phases = np.array([-0.25, 0.25, 0.75, -0.75])
    j_vals = np.exp(phases * 1j* np.pi)

    total = 0
    for q in index_points:
        loc = np.where(np.isclose(calc_points, q).all(1))[0]
        if len(loc) == 0:
            raise Exception(f"{q=} was not found in the calculation but it is"
                            " needed for the z4R index of group 75.1.")
        kp = kpoints[loc[0]]
        for sym in kp.symmetries.keys():
            if is_S4z(sym):
                s4z = sym
        s4z_vals = kp.symmetries[s4z]
        for phase, j_val in zip(phases, j_vals):
            j_mask = np.isclose(s4z_vals, j_val)
            total += phase * (j_mask[:occ]).sum()
    
    loc = np.where(np.isclose(calc_points, [0, 0.5, 0.5]).all(1))[0]
    if len(loc) == 0:
        raise Exception("R=(0, 1/2, 1/2) was not found in the calculation but" 
                        " it is needed for the z4R index of group 75.1.")
    kp = kpoints[loc[0]]
    for sym in kp.symmetries.keys():
        if is_C2(sym, 2):
            c2z = sym
            break
    c2z_vals = kp.symmetries[c2z].imag.round()
    total += (c2z_vals[:occ] == -1).sum() - (c2z_vals[:occ] == 1).sum()

    return total % 4

#############################################

def delta2S_81_33(kpoints, occ):
    index_points = np.array([
        [0,   0,   0.5], # Z
        [0.5, 0.5, 0.5], # A
        [0,   0 ,  0  ], # GM
        [0.5, 0.5, 0  ]  # M
    ])
    calc_points = np.array([kp.K for kp in kpoints])

    phases = np.array([-0.75, 0.75])
    signs = np.array([1, 1, -1, -1])
    j_vals = np.exp(phases * 1j* np.pi)

    total = 0
    for sign, q in zip(signs, index_points):
        loc = np.where(np.isclose(calc_points, q).all(1))[0]
        if len(loc) == 0:
            raise Exception(f"{q=} was not found in the calculation but it is"
                            " needed for the z4R index of group 75.1.")
        kp = kpoints[loc[0]]
        for sym in kp.symmetries.keys():
            if is_S4z(sym):
                s4z = sym
        c4z_vals = kp.symmetries[s4z]
        for j_val in j_vals:
            j_mask = np.isclose(c4z_vals, j_val)
            total += sign * (j_mask[:occ]).sum()

    return total % 2

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
            raise Exception(f"{q=} was not found in the calculation but it is"
                            " needed for the z4R index of group 75.1.")
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
            raise Exception(f"{q=} was not found in the calculation but it is"
                            " needed for the delta2m index of group 10.42.")
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
        raise Exception("X=(0,1/2,0) was not found in the calculation but it is"
                        " needed for the z4m0plus index of group 84.51.")
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






