#!/usr/bin/env python3

from subprocess import run
import sys
import vtk
import numpy as np
import trimesh


def gen_three_ply(
    vtk_filepath,
    output_base,
):
    ret = []
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_filepath)
    reader.Update()
    polydata = reader.GetOutput()

    # This portion is added to suppress the warning after the SHREC.
    # The result is identical.
    point_data = polydata.GetPointData()
    normals_array = point_data.GetNormals()
    if normals_array and normals_array.GetDataType() == vtk.VTK_DOUBLE:
        normals_float = vtk.vtkFloatArray()
        normals_float.SetName(normals_array.GetName())
        normals_float.SetNumberOfComponents(normals_array.GetNumberOfComponents())

        num_tuples = normals_array.GetNumberOfTuples()
        normals_float.SetNumberOfTuples(num_tuples)
        for i in range(num_tuples):
            normals_float.SetTuple(i, normals_array.GetTuple(i))
        point_data.SetNormals(normals_float)

    # shape
    shape_ply_path = output_base + "_shape.ply"
    writer_shape = vtk.vtkPLYWriter()
    writer_shape.SetFileName(shape_ply_path)
    writer_shape.SetInputData(polydata)
    writer_shape.SetFileTypeToASCII()
    writer_shape.Write()
    ret.append(shape_ply_path)

    # split surface
    points = polydata.GetPoints()
    polys = polydata.GetPolys()
    point_data = polydata.GetPointData()
    potential_array = point_data.GetArray("Potential")
    num_points = points.GetNumberOfPoints()
    potential_values = np.zeros(num_points)
    for i in range(num_points):
        potential_values[i] = potential_array.GetTuple1(i)

    pos_polys = vtk.vtkCellArray()
    neg_polys = vtk.vtkCellArray()
    polys.InitTraversal()
    id_list = vtk.vtkIdList()

    while polys.GetNextCell(id_list):
        num_poly_points = id_list.GetNumberOfIds()
        if num_poly_points < 3:
            continue
        poly_point_ids = [id_list.GetId(i) for i in range(num_poly_points)]
        vertex_potentials = potential_values[poly_point_ids]
        all_pos = np.all(vertex_potentials > 0)
        all_neg = np.all(vertex_potentials < 0)
        if all_pos:
            pos_polys.InsertNextCell(id_list)
        elif all_neg:
            neg_polys.InsertNextCell(id_list)

    # save positive surface
    pos_polydata = vtk.vtkPolyData()
    pos_polydata.SetPoints(points)
    pos_polydata.SetPolys(pos_polys)
    cleaner_pos = vtk.vtkCleanPolyData()
    cleaner_pos.PointMergingOff()
    cleaner_pos.SetInputData(pos_polydata)
    cleaner_pos.Update()
    pos_polydata_cleaned = cleaner_pos.GetOutput()

    pos_ply_path = output_base + "_pos.ply"
    if (
        pos_polydata_cleaned.GetNumberOfPoints() > 0
        and pos_polydata_cleaned.GetNumberOfPolys() > 0
    ):
        writer_pos = vtk.vtkPLYWriter()
        writer_pos.SetFileName(pos_ply_path)
        writer_pos.SetInputData(pos_polydata_cleaned)
        writer_pos.SetFileTypeToASCII()
        writer_pos.Write()
    ret.append(pos_ply_path)

    # save negative surface
    neg_polydata = vtk.vtkPolyData()
    neg_polydata.SetPoints(points)
    neg_polydata.SetPolys(neg_polys)
    cleaner_neg = vtk.vtkCleanPolyData()
    cleaner_neg.PointMergingOff()
    cleaner_neg.SetInputData(neg_polydata)
    cleaner_neg.Update()
    neg_polydata_cleaned = cleaner_neg.GetOutput()

    neg_ply_path = output_base + "_neg.ply"
    if (
        neg_polydata_cleaned.GetNumberOfPoints() > 0
        and neg_polydata_cleaned.GetNumberOfPolys() > 0
    ):
        writer_neg = vtk.vtkPLYWriter()
        writer_neg.SetFileName(neg_ply_path)
        writer_neg.SetInputData(neg_polydata_cleaned)
        writer_neg.SetFileTypeToASCII()
        writer_neg.Write()
    ret.append(neg_ply_path)

    return ret


def ply2obj(ply_filepath):
    mesh = trimesh.load(ply_filepath, process=False)
    obj_filepath = ply_filepath.replace(".ply", ".obj")
    mesh.export(obj_filepath, file_type="obj")
    return obj_filepath


def obj2grid(obj_filepath):
    _ = run(
        [
            "./bin/obj2grid",
            "-g",
            "64",
            f"{obj_filepath}",
        ],
        capture_output=True,
    )
    return f"{obj_filepath}.grid"


def map2zernike(grid_filepath):
    _ = run(
        [
            "./bin/map2zernike",
            "-c",
            "0.5",
            f"{grid_filepath}",
        ],
        capture_output=True,
    )
    return f"{grid_filepath}.inv"


def main():
    vtk_filepath = sys.argv[1]
    output_base = sys.argv[2]
    plys = gen_three_ply(vtk_filepath, output_base)
    objs = [ply2obj(ply) for ply in plys]
    grids = [obj2grid(obj) for obj in objs]
    invs = [map2zernike(grid) for grid in grids]
    return invs


if __name__ == "__main__":
    main()
