# -*- coding: utf-8 -*-

import numpy as np


def formatVrmlHeader():
    return "#VRML V2.0 utf8\n\n"


def formatVrmlWireframe(pts, connections, colour=(0, 0, 0)):
    s = (
        "Shape {\n"
        "appearance Appearance {\n"
        "    material Material {\n"
        "        ambientIntensity 0\n"
        "        diffuseColor     " + "{:.4f} {:.4f} {:.4f}\n".format(colour[0], colour[1], colour[2]) +
        "        specularColor    1 0 0\n"
        "        emissiveColor    1 0 0\n"
        "        shininess        1\n"
        "        transparency     0\n"
        "    }\n"
        "}\n"
        "geometry IndexedLineSet {\n"
        "    coord Coordinate {\n"
        "        point [\n"
    )
    for pt in pts:
        s += "            {:.6e} {:.6e} {:.6e}\n".format(pt[0], pt[1], pt[2])
    s += (
        "        ]\n"
        "    }\n"
        "    coordIndex [\n"
    )
    for edge in connections:
        s += "            "
        s += " ".join(["{:d}".format(i) for i in edge]) + " -1,\n"
    s += (
        "        ]\n"
        "    }\n"
        "}\n"
    )
    return s


def formatVrmlFaces(pts, faces, colour=(0, 0, 0), opacity=1):
    s = (
        "Shape {\n"
        "appearance Appearance {\n"
        "    material Material {\n"
        "        ambientIntensity 0\n"
        "        diffuseColor     " + "{:.4f} {:.4f} {:.4f}\n".format(colour[0], colour[1], colour[2]) +
        "        specularColor    1 0 0\n"
        "        emissiveColor    1 0 0\n"
        "        shininess        1\n"
        "        transparency     " + "{:.4f}\n".format(1.-opacity) +
        "    }\n"
        "}\n"
        "geometry IndexedFaceSet {\n"
        "    coord Coordinate {\n"
        "        point [\n"
    )
    for pt in pts:
        s += "            {:.6e} {:.6e} {:.6e}\n".format(pt[0], pt[1], pt[2])
    s += (
        "        ]\n"
        "    }\n"
        "    coordIndex [\n"
    )
    for face in faces:
        s += "            "
        s += " ".join(["{:d}".format(i) for i in face]) + " -1,\n"
    s += (
        "        ]\n"
        "    }\n"
        "}\n"
    )
    return s


def formatVrmlSphere(centre, radius, colour=(0, 0, 0)):
    # NOTE: this doesn't seem to work in Paraview, spheres stay at the origin.
    s = (
        "Transform {\n" +
        "translation {:.6e} {:.6e} {:.6e}\n".format(centre[0], centre[1], centre[2]) +
        "scale 1 1 1\n"
        "children [\n"
        "    Shape {\n"
        "        appearance Appearance {\n"
        "            material Material {diffuseColor " + "{:.4f} {:.4f} {:.4f}".format(colour[0], colour[1], colour[2]) + "}\n" +
        "        }\n"
        "       geometry Sphere {\n" +
        "           radius {:.6e}\n".format(radius) +
        "       }\n"
        "    }\n"
        "]\n"
        "}\n"
    )
    return s


def formatVrmlBox(centre, size, colour=(0, 0, 0)):
    pts = np.array([
        centre + [-size/2, -size/2, -size/2],
        centre + [size/2, -size/2, -size/2],
        centre + [size/2, size/2, -size/2],
        centre + [-size/2, size/2, -size/2],
        centre + [-size/2, -size/2, size/2],
        centre + [size/2, -size/2, size/2],
        centre + [size/2, size/2, size/2],
        centre + [-size/2, size/2, size/2],
    ])
    connections = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
        (1, 2, 6, 5),
        (0, 4, 7, 3),
    ]
    return formatVrmlFaces(pts, connections, colour=colour)
