import numpy as np

class FINDSYMData():
    """
    This class handles the acquisition of data from a FINDSYM output file. When supplied with 'auto' as intialization string
    it attempts to call FINDSYM locally. This requires that the program, along with the ISOTROPY suite, be downloaded on your computer
    and properly installed by setting your path environment variable.
    """

    def __init__(self,file='auto'):
        """
        Fields:
            -rotations: rotation matrices (3x3) set in the standard setting
            -translations: non-symmorphic translations associated to the rotations in the same setting
            -op_types: vector encoding the type of operation: unitary(+1) or antiunitary (-1)
            -basis_change: Change of coordinates from conventional to calculation Mca
            -shift: Origin of conventional given in calculation basis, tac.
            -origin: Change origin shift tac from calc basis coordinates to conventional basis coordinates
            -sg: Space group number
            -name: nomenclature in BNS setting
        """
        if file=='auto':
            import subprocess
            try:
                fsout=subprocess.run(['findsym','findsym.in'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                lines=(l for l in fsout.stdout.decode('utf-8').split('\n'))
                #Note that it captures the output into lines.
            except Exception as e:
                print("You do not have FINDSYM properly installed.")
                print(e)
                # exit()
            
        else:  
            try:
                lines=(l.strip() for l in open(file))
            except FileNotFoundError as error:
                print("Could not find the file {0} from FINDSYM. Make sure to provide it or run with \
-magnetic= auto to call FINDSYM automatically.".format(error.filename))
                pass

        rotations=[]
        translations=[]
        op_type=[]

        while '---' not in next(lines):
            continue
        for line in lines:
            if "Space Group" in line:
                line=line.split()
                self.sg=line[2]
                self.name=line[3]
            elif "Origin at" in line:
                #Origin of conventional given in calculation basis, tac.
                #Used to pass to spacegroup and define shiftUC, which must be in calculation basis (shiftuc=-self.shift)
                self.shift=np.array(line.split()[2:],dtype=float)
            elif "Vectors" in line:
                #Change of coordinates from conventional to calculation Mca
                self.basis_change=np.transpose(np.array([next(lines).split() for i in range(3)],dtype=float))
                #Change origin shift tac from calc basis coordinates to conventional basis coordinates
                #Used to change ops from conventional ouput of FINDSYM to actual calculation cell
                self.origin= np.dot(np.linalg.inv(self.basis_change),self.shift)
                while 'operation.id' not in next(lines):
                    continue
            elif "magn_operation.xyz" in line:
                print("operations")
                line=next(lines)
                while 'x' in line:
                    line=line[2:].split(',')
                    operation=self.__parse_matrix(line)
                    rotations.append(operation[0])
                    translations.append(operation[1])
                    op_type.append(operation[2])
            
                    line=next(lines)
            
                
            elif "magn_centering.xyz" in line:
                #Only for type IV Shubnikov space groups
                line=next(lines)
                while 'x' in line:
                    line=line[2:].split(',')
                    if line[-1]=='+1':
                        line=next(lines)
                        continue
                    else:
                        operation=self.__parse_matrix(line)
                        rotations.append(operation[0])
                        translations.append(operation[1])
                        op_type.append(operation[2])
                    line=next(lines)
                break
        lines.close()
        self.rotations=np.array(rotations)
        self.translations=np.array(translations)
        self.op_types=np.array(op_type,dtype=int)
        self.op_number=len(self.op_types)
        
    def __parse_matrix(self,string):
        """
        Auxiliary matrix to find the rotation and translations associated to an expression of the type
        'x+tx,y+ty,z+tz'.
        """
        def parse_term(s):
            row=[0,0,0]
            t,i,sign=0,0,1
            while i < len(s):
                if s[i]=='-':
                    sign=-1
                    i+=1
                elif s[i]=='x':
                    row[0]=sign
                    sign=1
                    i+=1
                elif s[i]=='y':
                    row[1]=sign
                    sign=1
                    i+=1
                elif s[i]=='z':
                    row[2]=sign
                    sign=1
                    i+=1
                elif s[i].isdigit():
                    t=sign*(int(s[i])/int(s[i+2]))
                    i+=3
                else:
                    i+=1
            
            return row,t
        rot=np.zeros((3,3),dtype=int)
        trans=np.zeros(3,dtype=float)
        for i,term in enumerate(string[:-1]):
            rot[i,:],trans[i]=parse_term(term)
        return (rot,trans,int(string[-1]))
        
    
    def unitary_operations(self,calcbasis=True):
        """
        Returns the set of unitary operations. Calcbasis determines whether the operations from FINDSYM are transformed
        from the output in the standard cell to the cell in the calculation (True by default).
        """
        if calcbasis:
            calcrot=np.array([self.__rotation_refUC(rot) for rot in self.rotations[self.op_types==1]],dtype=int)
            calctrans= np.array([self.__translation_refUC(rot,trans) for rot,trans in zip(self.rotations[self.op_types==1],self.translations[self.op_types==1])])
            return (calcrot,calctrans)
        else:
            print("rotations:\n",self.rotations[self.op_types==1])
            print("translations:\n",self.translations[self.op_types==1])
            return (self.rotations[self.op_types==1],self.translations[self.op_types==1])
    def antiunitary_operations(self,calcbasis=True):
        """
        Returns the set of antiunitary operations. Calcbasis determines wheter the operations from FINDSYM are transformed
        from the output in the standard cell to the cell in the calculation (True by default). Not used in actual irrep calculations.
        Currently only used to provide additional information of space group operations when non-unitary operations are present.
        """
        if calcbasis:
            calcrot=np.array([self.__rotation_refUC(rot) for rot in self.rotations[self.op_types==-1]],dtype=int)
            calctrans= np.array([self.__translation_refUC(rot,trans) for rot,trans in zip(self.rotations[self.op_types==-1],self.translations[self.op_types==-1])])
            return (calcrot,calctrans%1)
        else:
            return (self.rotations[self.op_types==-1],self.translations[self.op_types==-1])


    def __rotation_refUC(self,rotation):
        """
        Auxiliary function to transform from the standard setting to the calculation setting.
        """
        # self.basis_change is the change of coordinates from std to calculation -> Mca
        Mac=np.linalg.inv(self.basis_change) #Change of basis from calc to std -> Mac
        # Ra= Mca Rc Mac
        Ra=self.basis_change.dot(np.dot(rotation,Mac)) #Rotation in calculation basis
        
        return Ra.astype(int)

    def __translation_refUC(self,rotation,translation):
        """
        Auxiliary function to transform from the standard setting to the calculation setting.
        """
        #self.origin is the shift from calc to std in std coordinates -> tac
        #a shift induces v-> v+Rc tca - tca = v-Rc tac +tac
        #Then change to calc basis: Mca (v-Rc tac +tac)
        return self.basis_change.dot(translation+self.origin-rotation.dot(self.origin))%1

def make_fs_input(lattice,natoms,typeatoms,positions,magmoments=None,title="FINDSYM input",lattol=None,atpostol=None,occtol=None,magtol=None,centering="P"):
    """
    Function that takes the input of the lattice read from Spacegroup class (lattice, type of atoms, magnetic moments, etc.)
    and creates an input file for local use of FINDSYM.
    """
    with open("findsym.in",'w') as fsout:
        fsout.write("!title\n{0}\n".format(title))
        if lattol:
            fsout.write("!latticeTolerance\n{:f}\n".format(lattol)) # default:10e-5 ang
        if atpostol:
            fsout.write("!atomicPositionTolerance\n{:f}\n".format(atpostol)) # default:10e-3 ang
        if occtol:
            fsout.write("!occupationTolerance\n{:f}\n",format(occtol)) # default:10e-3
        if magtol:
            fsout.write("!magneticMomentTolerance\n{:f}\n".format(magtol)) # default:10e-3 bm
        # fsout.write("!latticeParameters\n") lengths & angles
        fsout.write("!latticeBasisVectors\n") #conventional unit cell
        np.savetxt(fsout,lattice,fmt="%.6f")
        
        fsout.write("!unitCellCentering\n{:s}\n".format(centering)) #P (primitive or unknown or centering included) IFABCR
        
        # fsout.write("!unitCellBasisVectors\n")
        # Enter the basis vectors of the lattice which defines the unit
        # cell, if different from the conventional unit cell defined by the lattice
        # parameters or the lattice basis vectors listed above.  This unit cell does not 
        # need to be primitive.  The vectors should be given in dimensionless units in
        # terms of the basis vectors of the conventional lattice.  Enter each vector on a
        # separate line.  These vector components are dimensionless and must be accurate 
        # to 3 decimal places.  For example, 1/2 would be entered as 0.5, and 1/3 would be
        # entered as 0.333.  
        
        fsout.write("!atomCount\n{0}\n".format(natoms))
        fsout.write("!atomType\n{0}\n".format(' '.join([str(x) for x in typeatoms])))
        fsout.write("!atomPosition\n")
        np.savetxt(fsout,positions,fmt="%.6f")
        # fsout.write("!atomOccupation\n")
        
        fsout.write("!atomMagneticMoment\n")
        if magmoments is not None:
            np.savetxt(fsout,magmoments,fmt="%.5f")
        else:
            np.savetxt(fsout,np.zeros((natoms,natoms),dtype=float),fmt="%.4f")
        print("Created input for FINDSYM from ab-initio data.")
        