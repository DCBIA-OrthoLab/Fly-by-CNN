import sys, pdb, json, os
from threading import Thread


ARRAY_SIZE = 65536*3

vertices = []
indices = []
normals = []
scalars = []
colors = []

mode = "SOLID"

NOWHERE = 0    
POINTS = 1
LINES = 2
POLYGONS = 3
POINT_DATA = 4
NORMALS = 5
CELL_DATA = 6
TEXTURE_COORDINATES = 7
SCALARS = 8
LOOKUP_TABLE = 9
COLOR_SCALARS = 10

outputfile = ''
num_parts = 0

'''
-----------------------------------------------------------------------
Clears the console
-----------------------------------------------------------------------
'''
try:
    os.system('cls')
    os.system('clear')
except:
    pass

'''
-----------------------------------------------------------------------
Creates the line segments for multilines
-----------------------------------------------------------------------
'''
def createIndicesForLine(line):
    global indices
    elements = line.split()
    if len(elements) == 0:
        return
    N = int(elements[0])
    values = elements[1:len(elements)]
    
    
    for i,j in enumerate(values):
        if i+1<N:
            indices.append(values[i])
            indices.append(values[i+1])
    
 
''' 
-----------------------------------------------------------------------
 Parses the VTK file
-----------------------------------------------------------------------
'''
def parseVTK(filename):

    global mode, vertices, indices, normals, colors, scalars, hasLines
    
    location = NOWHERE
    
    linenumber = 0
    
    print('Hold on. Reading file...')
        
    for line in open(filename, 'r').readlines():
        
        linenumber = linenumber + 1
        try:
            if line.startswith('POINTS'):
                print(line)
                location = POINTS
                continue
            elif line.startswith('LINES'):
                print(line)
                location = LINES
                mode = "LINES"
                continue
            elif line.startswith('POLYGONS'):
                print(line)
                location = POLYGONS
                continue
            elif line.startswith('POINT_DATA'):
                location = POINT_DATA
                continue
            elif line.startswith('NORMALS'):
                print(line)
                location = NORMALS
                continue
            elif line.startswith('CELL_DATA'):
                print(line)
                location = CELL_DATA
                continue
            elif line.startswith('TEXTURE_COORDINATES'):
                print(line)
                location = TEXTURE_COORDINATES
                continue
            elif line.startswith('SCALARS'):
                print(line)
                location = SCALARS
                continue
            elif line.startswith('LOOKUP_TABLE'):
                print(line)
                location = LOOKUP_TABLE
                continue
            elif line.startswith('COLOR_SCALARS'):
                location = COLOR_SCALARS
                print(line)
                continue
            elif location == POINTS:
                for pp in line.split():
                    vertices.append(pp)
            elif location == LINES:
                createIndicesForLine(line)
            elif location == POLYGONS: #they have to be triangles
                tt = line.split()
                if len(tt)>0 and tt[0] != '3':
                    raise AssertionError('Not triangles here')
                for i in tt[1:len(tt)]:
                    indices.append(int(i))
            elif location == LOOKUP_TABLE:
                if line.startswith('LOOKUP_TABLE'):
                    continue
                else:
                    for pd in line.split():
                        scalars.append(float(pd))
            elif location == COLOR_SCALARS:
                for n in line.split():
                    colors.append(float(n))
            elif location == NORMALS:
                for n in line.split():
                    normals.append(float(n))
        except:
            print('Error while processing line '+str(linenumber))
            print(line)
            raise
    
    print('Veryfing numeric types...')
    
    vertices = [float(x) for x in vertices]
    indices =[int(x) for x in indices]
    
    print('Done')
    
    printInfo()

'''
-----------------------------------------------------------------------
Prints information about the geometry stored in the VTK file
-----------------------------------------------------------------------
'''    
def printInfo():
    v_count = len(vertices)/3
    n_count = len(normals)/3
    c_count = len(colors)/3
    pd_count = len(scalars)
    i_count = len(indices)
    
    print
    print('Info from file:')
    print('----------------------------')
    print('geometry:\t%s'%mode)
    print('vertices:\t'  + str(v_count))
    print('normals:\t'   + str(n_count))
    print('indices:\t'   + str(i_count))
    print('scalars:\t'   + str(pd_count))
    print('colors:\t'    +str(c_count))   
    print('max index id:\t[%d]'% max(indices)) 
    if mode == 'SOLID':
        print('triangles:\t' + str(i_count/3)) 
    else:
        print('lines:\t'+ str(i_count/2))
 
        
'''
-----------------------------------------------------------------------
 Obtains information for one vertex
-----------------------------------------------------------------------
'''
def getVertexData(index):
    
    vertex = {}
    
    vertex['coords']   = vertices[index*3: index*3+3]
    
    if (len(normals)>0):
        vertex['normal'] = normals[index*3: index*3+3]
        
    if (len(colors)>0):
        vertex['color'] = colors[index*3: index*3+3]
    
    if (len(scalars)>0):
        vertex['scalar'] = scalars[index]
    
    return vertex
   

def processBigData2():
    
    global num_parts
    
    print('')
    print('Processing Big Data 2')
    print('')
    
    global_index = indices
    size =  65535
    
    if mode=='LINES':
        size = size -1
    L = len(global_index)    
    has_normals = len(normals) > 0
    has_colors  = len(colors)  > 0
    has_scalars = len(scalars) > 0
    
    
    part = {'vertices':[], 'indices':[],'mode':mode}
    if (has_colors):  
        part['colors'] = []
    if (has_normals): 
        part['normals'] = []
    if (has_scalars):
        part['scalars'] = []
    
    index_map = {}
    part_number = 1
    new_index = 0
    
    for p, index in enumerate(global_index):
        
        progress = (p/float(L)) *100
        
        if index not in index_map:
             
            index_map[index] = new_index
            vertex = getVertexData(index)
            
            part['vertices'] += vertex['coords']
            
            if has_normals:
                part['normals'] += vertex['normal']
            if has_colors:
                part['colors'] += vertex['color']
            if has_scalars:
                part['scalars'].append(vertex['scalar'])
            
            new_index +=1
        
        part['indices'].append(index_map[index])
        
        if new_index == size + 1 or p == L-1:
            
            sys.stdout.write('Status: [Part %d]  [%s%.1f]\r' % (part_number,'%',progress))
            
            createPart(part_number, part)
            
            new_index    = 0
            part_number += 1
            part         = {'vertices':[], 'indices':[],'mode':mode}
            index_map    = {}
            
            if (has_colors):  
                part['colors'] = []
            if (has_normals): 
                part['normals'] = []
            if (has_scalars):
                part['scalars'] = []
     
       
    print ('DONE.')


def processBigData():
    
    global num_parts
    
    print 
    print('Processing Big Data')
    print
    
    global_index = indices
    size =  65535
    
    if mode=='LINES':
        size = size -1
    L = len(global_index)    
    N = len(global_index) // size
    R = len(global_index) % size
    
    has_normals = len(normals) > 0
    has_colors  = len(colors)  > 0
    has_scalars = len(scalars) > 0
    
    #parts = []
    num_parts = N if R==0 else N+1
    
    print( 'Number of Parts: %d'%(num_parts))
    print
    
    for i in range(N+1):
        
        part              = {}
        index_map         = {}
        global_index_part = []
        new_index         = 0
        
        if i<N:
            global_index_part = global_index[i*size:(i+1)*size];
        elif R>0:
            global_index_part = global_index[i*size:i*size+R];
        else:
            break;
        
        part['vertices'] = []
        part['indices']  = []
         
        if (has_colors):  
            part['colors'] = []
        if (has_normals): 
            part['normals'] = []
        if (has_scalars):
            part['scalars'] = []
            
        for k in range(len(global_index_part)):
            
            
            index = global_index_part[k]
            sys.stdout.write('Status: [Part %d of %d] [index %d out of %d] \r' % (i+1,num_parts, index,L))
            
            if index not in index_map:
                index_map[index] = new_index
                
                vertex = getVertexData(index)
                
                part['vertices'] += vertex['coords']
                
                if has_normals:
                    part['normals'] += vertex['normal']
                
 
                if has_colors:
                    part['colors'] += vertex['color']
                
 
                if has_scalars:
                    part['scalars'].append(vertex['scalar'])
                
                new_index +=1;
                
            part['indices'].append(index_map[index])
        part['mode'] = mode
        createPart(i+1, part)
        
    print ('DONE.')

        
# -----------------------------------------------------------------------    
# Selects a part to process
# -----------------------------------------------------------------------    
def createPart(part_id, part):
    
    file_id = str(part_id)
     
    if num_parts == 1:
        filename = outputfile + '.json'
    else:
        filename = outputfile+'_'+file_id+'.json'
    
    with open(filename,'w') as part_file:
        json.dump(part,part_file,indent=2)
    
   

''' 
-----------------------------------------------------------------------    
 Main function
-----------------------------------------------------------------------    
'''
def main():
    global outputfile
    if len(sys.argv) != 3:
        print
        print ('Usage:\n\n python vtk2json.py [vtk file (ASCII)] [output file (no extension)]')
        exit()
    
    outputfile = sys.argv[2]

    print('----------------------------------------------------------')
    print(' Processing: ' + sys.argv[1])
    print('----------------------------------------------------------')
    parseVTK(sys.argv[1])
    processBigData2()
    print('----------------------------------------------------------')
    print("                       DONE                               ")
    print('----------------------------------------------------------')

if __name__ == '__main__':
    main()
            
    