import numpy as np
class FindsymData():

    def __init__(self,file='auto'):
        self.__readfile(file)
        
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
        self.rotations=np.array(rotations)
        self.translations=np.array(translations)
        self.op_types=np.array(op_type,dtype=int)
    
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
                