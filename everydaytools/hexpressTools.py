import numpy as np
import pandas

# TODO move cyl to stlTools and harmonise
# TODO add colours to obj file output of boxes to indicate ref lvl in Paraview

def make_ref_cyl(x0, iHat, jHat, dx, Rout):
    kHat = np.cross(iHat, jHat)
    
    t = np.linspace(0., 2.*np.pi, 101)
    xi = Rout*np.cos(t)
    yi = Rout*np.sin(t)
    # xih = Rin*np.cos(t)
    # yih = Rin*np.sin(t)
    xir = (dx+Rout)*np.cos(t)
    yir = (dx+Rout)*np.sin(t)
    
    x = x0[np.newaxis, :] + xi[:, np.newaxis]*jHat + yi[:, np.newaxis]*kHat
    # xh = x0[np.newaxis, :] + xih[:, np.newaxis]*jHat + yih[:, np.newaxis]*kHat
    xr = x0[np.newaxis, :] + xir[:, np.newaxis]*jHat + yir[:, np.newaxis]*kHat
    
    stlFaces = []
    for j in range(1, x.shape[0]):
        stlFaces.append(np.array([x0, xr[j-1, :], xr[j, :]]) + iHat*dx)
        stlFaces.append(np.array([x0, xr[j-1, :], xr[j, :]]) - iHat*dx)
        stlFaces.append(np.array([
            xr[j-1, :] - iHat*dx,
            xr[j-1, :] + iHat*dx,
            xr[j, :] + iHat*dx
        ]))
        stlFaces.append(np.array([
            xr[j-1, :] - iHat*dx,
            xr[j, :] + iHat*dx,
            xr[j, :] - iHat*dx
        ]))
    
    # pts = x.copy()
    # pts = np.concatenate([pts, xh])
    
    return stlFaces, pandas.DataFrame(data=np.array(x), columns=["x", "y", "z"])


def write_stl_list(stls, filename):
    with open(filename, "w") as sfile:
        for iStl, stl in enumerate(stls):
            sfile.write(f"solid obj_{iStl:f}\n")

            for face in stl:
                e0 = face[1, :] - face[0, :]
                e1 = face[2, :] - face[0, :]
                n = np.cross(e0, e1)
                sfile.write("facet normal {:.6e} {:.6e} {:.6e}\n".format(n[0], n[1], n[2]))
                sfile.write("  outer loop\n")
                sfile.write("    vertex {:.6e} {:.6e} {:.6e}\n".format(face[0, 0], face[0, 1], face[0, 2]))
                sfile.write("    vertex {:.6e} {:.6e} {:.6e}\n".format(face[1, 0], face[1, 1], face[1, 2]))
                sfile.write("    vertex {:.6e} {:.6e} {:.6e}\n".format(face[2, 0], face[2, 1], face[2, 2]))
                sfile.write("  endloop\n")
                sfile.write("endfacet\n")

            sfile.write("endsolid\n")


class RefBox(object):
    def __init__(self, vertices, level, directions, volumic):
        self.faces = [
            [1, 4, 3, 2],
            [5, 6, 7, 8],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 4, 8, 7],
            [1, 5, 8, 4],
        ]
        self.vertices = vertices
        self.level = level
        self.directions = directions
        self.volumic = volumic


class HexpressRefZones(object):
    def __init__(self):
        self.ref_boxes = []

    def scale(self, scale_factor):
        for box in self.ref_boxes:
            box.vertices *= scale_factor
    
    def add_box(self, basePoint, size, level, directions=[1, 1, 1], volumic=True, centreOnX=False):
        """ Make a box given a base point and dimensions along x,y,z axes. Centre of
        the first x-face is located at the base point and the second one is size[0]
        away along the x-axis. """
        vertices = np.zeros((8, 3))
        if centreOnX:
            x0 = -size[0]/2
            x1 = size[0]/2
        else:
            x0 = 0
            x1 = size[0]
        vertices[0,:] = np.array(basePoint) + [x0, size[1]/2, size[2]/2]
        vertices[1,:] = np.array(basePoint) + [x0, size[1]/2, -size[2]/2]
        vertices[2,:] = np.array(basePoint) + [x0, -size[1]/2, -size[2]/2]
        vertices[3,:] = np.array(basePoint) + [x0, -size[1]/2, size[2]/2]
        vertices[4,:] = np.array(basePoint) + [x1, size[1]/2, size[2]/2]
        vertices[5,:] = np.array(basePoint) + [x1, size[1]/2, -size[2]/2]
        vertices[6,:] = np.array(basePoint) + [x1, -size[1]/2, -size[2]/2]
        vertices[7,:] = np.array(basePoint) + [x1, -size[1]/2, size[2]/2]
        
        self.ref_boxes.append(
            RefBox(
                vertices,
                level,
                directions,
                volumic
            )
        )
        
        return vertices

    def add_box_extents(self, xMin, xMax, level, directions=[1, 1, 1], volumic=True):
        """ Make a box given bounding points. """
        basePoint = (np.array(xMin) + np.array(xMax)) / 2
        size = np.array(xMax) - np.array(xMin)
        return self.add_box(basePoint, size, level, directions=directions, volumic=volumic, centreOnX=True)

    def to_obj(self, filename):
        s = ""
        di = 0
        for iBox, box in enumerate(self.ref_boxes):
            for i, pt in enumerate(box.vertices):
                s += "v {:.6e} {:.6e} {:.6e}\n".format(pt[0], pt[1], pt[2])
            for i in box.faces:
                s += "f {:d} {:d} {:d} {:d}\n".format(i[0]+di, i[1]+di, i[2]+di, i[3]+di)
            di += len(box.vertices)
        
        with open(filename, "w") as outf:
            outf.write(s)
        
        return s

    def to_py(self, filename):
        # Autogenerate code for Hexpress.
        s = "HXP.delete_all_refinement_boxes()\n"
        for i, box in enumerate(self.ref_boxes):
            # Define using corners
            s += "HXP.create_refinement_cube({:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e})\n".format(
                np.min(box.vertices[:,0]), np.min(box.vertices[:,1]), np.min(box.vertices[:,2]),
                np.max(box.vertices[:,0]), np.max(box.vertices[:,1]), np.max(box.vertices[:,2]))
            # Volumetric refinement and active. Skip for surface-only refinement.
            if box.volumic:
                s += "HXP.refinement_box({:d}).set_adaptation_flags(1, 1)\n".format(i)
            else:
                s += "HXP.refinement_box({:d}).set_adaptation_flags(1, 0)\n".format(i)
            # Can override the zero size by using:
            s += "HXP.refinement_box({:d}).set_target_size(".format(i)
            for j in range(3):
                if box.directions[j]:
                    s += "0"
                else:
                    s += "1000"
                if j < 2:
                    s += ", "
                else:
                    s += ")\n"
            # Refinement level, obviously
            s += "HXP.refinement_box({:d}).set_refinement_level({:d})\n".format(i, box.level)

        with open(filename, "w") as outf:
            outf.write(s)
        
        return s


if __name__ == "__main__":

    hxp = HexpressRefZones()
    hxp.add_box([0, -0.153, -0.03], [0.41, 0.015, 0.17], centreOnX=True, level=10, volumic=False)
    print(hxp.to_obj("D:/2026_AUVconference/geometry/boxes.obj"))
    print(hxp.to_py("D:/2026_AUVconference/geometry/boxes.py"))