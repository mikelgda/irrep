import numpy as np

class FINDSYMData():

    def __init__(self,file='auto'):
        self.rotations,self.translations,self.op_types=self.__readfile(file)
        self.op_number=len(self.op_types)
        
    def __readfile(self,file='auto'):
        if file=='auto':
            import subprocess
            fsout=subprocess.run(['findsym','findsym.in'],catpure_ouput=True)
            lines=(l for l in fsout.stdout.decode('utf-8').split('\n'))
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
            if "Origin at" in line:
                self.origin=np.array(line.split()[2:],dtype=float)
            if "Vectors" in line:
                self.basis=np.array([next(lines).split() for i in range(3)],dtype=float)
                while 'operation.id' not in next(lines):
                    continue
            if "magn_operation.xyz" in line:
                line=next(lines)
                while 'x' in line:
                    line=line[2:].split(',')
                    operation=self.__parse_matrix(line)
                    rotations.append(operation[0])
                    translations.append(operation[1])
                    op_type.append(operation[2])
            
                    line=next(lines)
            
                break
        lines.close()
        return np.array(rotations),np.array(translations),np.array(op_type,dtype=int)
    
    def __parse_matrix(self,string):
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
        
    
    def unitary_operations(self):
        return (self.rotations[self.op_types==1],self.translations[self.op_types==1])
    def antiunitary_operations(self):
        return (self.rotations[self.op_types==-1],self.translations[self.op_types==-1])

def make_fs_input(lattice,natoms,typeatoms,positions,magmoments=None,title="FINDSYM input",lattol=None,atpostol=None,occtol=None,magtol=None,centering="P"):
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
        fsout.write("!atomType\n{0}\n".format(' '.join(typeatoms)))
        fsout.write("!atomPosition\n")
        np.savetxt(fsout,positions,fmt="%.6f")
        # fsout.write("!atomOccupation\n")
        
        fsout.write("!atomMagneticMoment\n")
        if magmoments is not None:
            np.savetxt(fsout,magmoments)
        else:
            np.savetxt(fsout,np.zeros((natoms,natoms),dtype=float),fmt="%.4f")
        