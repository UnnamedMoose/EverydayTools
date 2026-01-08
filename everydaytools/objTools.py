
def writePoint(pt, outfile=None):
    """ Format the point to an obj vector. Save when given a file pointer. """
    s = "v {:.6e} {:.6e} {:.6e}\n".format(pt[0], pt[1], pt[2])
    if outfile is not None:
        outfile.write(s)
    return s


def writeBox(vertices, filename=None, writeMode="w", di=0):
    """ Accepts (8,3) vertices making up the box and formats in object file standard.
    Writes to the file when given a filename.
    Assumes CGNS-type vertex ordering, i.e. around one face and then the next.
    Can offset the starting point index to allow multiple sets of vertices in the same file.
    """

    s = ""
    for i in range(vertices.shape[0]):
        s += writePoint(vertices[i,:])
    faces = [
        [1, 4, 3, 2],
        [5, 6, 7, 8],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 4, 8, 7],
        [1, 5, 8, 4],
    ]
    for i in faces:
        s += "f {:d} {:d} {:d} {:d}\n".format(i[0]+di, i[1]+di, i[2]+di, i[3]+di)
    
    if filename is not None:
        with open(filename, writeMode) as f:
            f.write(s)

    return s
