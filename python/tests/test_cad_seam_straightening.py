"""Leaning seam pcurves are straightened within edge tolerance and bounded (#361).

Open CASCADE's BRepMesh_IncrementalMesh can return IMeshData_Failure and leave
a valid conical face untriangulated when its seam pcurves lean from vertical by
even 1e-15 in UV (common in STEP imports of tapered holes). The adapter
straightens seam pcurves that are vertical or horizontal within their edge
tolerance on its private copy before meshing, charges the proven 3D move bound
into the face's error budget, and reports the count and bound.

Checked here without the adapter's own bookkeeping: facet edges matched on
bitwise coordinates, enclosed volume, three-direction deviation against the
unmodified input CAD solid, caller BRep immutability, reversed hole faces,
horizontal seams, BSpline seam pcurves, and refusal to straighten genuinely
slanted seams beyond their edge tolerance.
"""

import math
import unittest
from unittest import mock

try:
    import cadquery as cq
    import numpy as np
except ImportError:  # pragma: no cover - CI installs CadQuery
    cq = None

from camlib import _cad_surface as cad
from camlib.cadquery import export_surface_mesh

if cq is not None:
    from test_cad_watertight import (
        distance_to_mesh,
        distance_to_solid,
        edge_points,
        surface_points,
        unmatched_edges,
    )


def _cone_face(lean, *, tol=1e-7, match_top=False, bottom_lean=False, bspline_seam=False):
    """Build the 1-degree tapered cone face from issue #361 edge by edge."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepLib import BRepLib
    from OCP.Geom import Geom_Circle, Geom_ConicalSurface, Geom_Line
    from OCP.Geom2d import Geom2d_BSplineCurve, Geom2d_Line
    from OCP.gp import gp_Ax2, gp_Ax3, gp_Dir, gp_Dir2d, gp_Lin, gp_Pnt, gp_Pnt2d
    from OCP.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS_Edge, TopoDS_Face, TopoDS_Vertex, TopoDS_Wire

    try:
        from OCP.TColgp import TColgp_Array1OfPnt2d as Points2d
        from OCP.TColStd import TColStd_Array1OfInteger as Integers, TColStd_Array1OfReal as Reals
    except ImportError:  # OCP 8 moved collection aliases.
        from OCP.collections import Array1_double as Reals, Array1_gp_Pnt2d as Points2d, Array1_int as Integers

    radius, height, u0 = 1.1, 3.2, math.pi
    builder = BRep_Builder()
    surface = Geom_ConicalSurface(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, -1), gp_Dir(-1, 0, 0)), math.radians(1), radius)
    face = TopoDS_Face()
    builder.MakeFace(face, surface, tol)

    def make_vertex(u, v):
        out = TopoDS_Vertex()
        builder.MakeVertex(out, surface.Value(u, v), tol)
        return out

    u_top = u0 + (lean * height if match_top else 0.0)
    top, bottom = make_vertex(u_top, 0.0), make_vertex(u0, -height)

    def make_circle(u_start, v, end):
        circle = surface.VIso(v).Circ()
        edge = TopoDS_Edge()
        builder.MakeEdge(
            edge,
            Geom_Circle(gp_Ax2(circle.Location(), gp_Dir(0, 0, 1), gp_Dir(-1, 0, 0)), circle.Radius()),
            tol,
        )
        builder.Add(edge, end.Oriented(TopAbs_FORWARD))
        builder.Add(edge, end.Oriented(TopAbs_REVERSED))
        t_end = 2 * u0 + 2 * math.pi - u_start
        builder.Range(edge, t_end - 2 * math.pi, t_end)
        builder.UpdateEdge(edge, Geom2d_Line(gp_Pnt2d(2 * u0 + 2 * math.pi, v), gp_Dir2d(-1, 0)), face, tol)
        return edge

    c_top, c_bottom = make_circle(u_top, 0.0, top), make_circle(u0, -height, bottom)
    a, b = surface.Value(u0, -height), surface.Value(u_top, 0.0)
    seam = TopoDS_Edge()
    builder.MakeEdge(seam, Geom_Line(gp_Lin(a, gp_Dir(b.X() - a.X(), b.Y() - a.Y(), b.Z() - a.Z()))), tol)
    builder.Add(seam, bottom.Oriented(TopAbs_FORWARD))
    builder.Add(seam, top.Oriented(TopAbs_REVERSED))
    u_shift = -lean * height if bottom_lean else 0.0
    if bspline_seam:

        def make_bspline(u_base):
            pts = Points2d(1, 3)
            pts.SetValue(1, gp_Pnt2d(u_base + u_shift, -height))
            pts.SetValue(2, gp_Pnt2d(u_base + u_shift + 0.5 * lean * height, -0.5 * height))
            pts.SetValue(3, gp_Pnt2d(u_base + u_shift + lean * height, 0.0))
            knots, mults = Reals(1, 2), Integers(1, 2)
            knots.SetValue(1, 0.0)
            knots.SetValue(2, height)
            mults.SetValue(1, 3)
            mults.SetValue(2, 3)
            return Geom2d_BSplineCurve(pts, knots, mults, 2)

        c_fwd, c_rev = make_bspline(u0 + 2 * math.pi), make_bspline(u0)
    else:
        c_fwd = Geom2d_Line(gp_Pnt2d(u0 + 2 * math.pi + u_shift, -height), gp_Dir2d(lean, 1.0))
        c_rev = Geom2d_Line(gp_Pnt2d(u0 + u_shift, -height), gp_Dir2d(lean, 1.0))
    builder.UpdateEdge(seam, c_fwd, c_rev, face, tol)
    builder.Range(seam, face, 0.0, height)
    builder.Range(seam, 0.0, a.Distance(b), True)
    builder.SameParameter(seam, False)
    builder.SameRange(seam, False)
    wire = TopoDS_Wire()
    builder.MakeWire(wire)
    for edge, orientation in (
        (c_bottom, TopAbs_REVERSED),
        (seam, TopAbs_FORWARD),
        (c_top, TopAbs_FORWARD),
        (seam, TopAbs_REVERSED),
    ):
        builder.Add(wire, edge.Oriented(orientation))
    builder.Add(face, wire)
    BRepLib.SameParameter_s(face, tol, False)
    return face, c_top, c_bottom


def _frustum(lean, **kwargs):
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.TopoDS import TopoDS_Shell, TopoDS_Solid

    builder = BRep_Builder()
    face, c_top, c_bottom = _cone_face(lean, **kwargs)
    caps = [BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(e).Wire(), True).Face() for e in (c_top, c_bottom)]
    for flips in range(4):
        shell = TopoDS_Shell()
        builder.MakeShell(shell)
        builder.Add(shell, face)
        for k, cap in enumerate(caps):
            builder.Add(shell, cap.Reversed() if flips >> k & 1 else cap)
        solid = TopoDS_Solid()
        builder.MakeSolid(solid)
        builder.Add(solid, shell)
        result = cq.Solid(solid)
        if result.isValid() and result.Volume() > 0:
            return result
    raise RuntimeError("no valid frustum")


def _tapered_hole_bushing(lean):
    """A valid hollow cylinder whose inner wall is the REVERSED tapered cone."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.TopAbs import TopAbs_FORWARD
    from OCP.TopoDS import TopoDS, TopoDS_Shell, TopoDS_Solid

    builder = BRep_Builder()
    cone_face, c_top, c_bottom = _cone_face(lean)
    z_top = cq.Edge(c_bottom).Center().z
    outer = cq.Solid.makeCylinder(4.0, z_top)
    cyl_wall = next(f for f in outer.Faces() if f.geomType() == "CYLINDER")
    top_disk = next(f for f in outer.Faces() if f.geomType() == "PLANE" and abs(f.Center().z - z_top) < 1e-6)
    bot_disk = next(f for f in outer.Faces() if f.geomType() == "PLANE" and abs(f.Center().z) < 1e-6)

    def make_ring(disk, inner_edge):
        for flip in (False, True):
            hole = BRepBuilderAPI_MakeWire(TopoDS.Edge_s(inner_edge.Reversed() if flip else inner_edge)).Wire()
            ring = BRepBuilderAPI_MakeFace(TopoDS.Face_s(disk.wrapped.Oriented(TopAbs_FORWARD)), hole).Face()
            if cq.Face(ring).isValid() and cq.Face(ring).Area() < disk.Area():
                return ring
        raise RuntimeError("no valid ring face")

    f_top, f_bot = make_ring(top_disk, c_bottom), make_ring(bot_disk, c_top)
    for flips in range(8):
        shell = TopoDS_Shell()
        builder.MakeShell(shell)
        builder.Add(shell, cyl_wall.wrapped)
        for k, face in enumerate((f_top, f_bot, cone_face)):
            builder.Add(shell, face.Reversed() if flips >> k & 1 else face)
        solid = TopoDS_Solid()
        builder.MakeSolid(solid)
        builder.Add(solid, shell)
        result = cq.Solid(solid)
        if result.isValid() and result.Volume() > 100:
            return result
    raise RuntimeError("no valid bushing")


def _helical_seam_cylinder():
    """A valid cylinder whose seam genuinely twists by 0.25 rad (0.5 mm >> tol)."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire
    from OCP.BRepLib import BRepLib
    from OCP.Geom import Geom_Circle, Geom_CylindricalSurface
    from OCP.Geom2d import Geom2d_Line
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.gp import gp_Ax3, gp_Dir, gp_Dir2d, gp_Pnt, gp_Pnt2d
    from OCP.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS_Edge, TopoDS_Face, TopoDS_Shell, TopoDS_Solid, TopoDS_Vertex, TopoDS_Wire

    try:
        from OCP.TColgp import TColgp_HArray1OfPnt as HPoints
    except ImportError:  # OCP 8 moved collection aliases.
        from OCP.collections import HArray1_gp_Pnt as HPoints

    tol, radius, height, twist = 1e-7, 2.0, 4.0, 0.25
    builder = BRep_Builder()
    surface = Geom_CylindricalSurface(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)), radius)
    face = TopoDS_Face()
    builder.MakeFace(face, surface, tol)

    def make_vertex(u, v):
        out = TopoDS_Vertex()
        builder.MakeVertex(out, surface.Value(u, v), tol)
        return out

    v0, v1 = make_vertex(0.0, 0.0), make_vertex(twist, height)

    def make_circle(u_start, v, end):
        circle = surface.VIso(v).Circ()
        edge = TopoDS_Edge()
        builder.MakeEdge(edge, Geom_Circle(circle.Position(), circle.Radius()), tol)
        builder.Add(edge, end.Oriented(TopAbs_FORWARD))
        builder.Add(edge, end.Oriented(TopAbs_REVERSED))
        builder.Range(edge, u_start, u_start + 2 * math.pi)
        builder.UpdateEdge(edge, Geom2d_Line(gp_Pnt2d(0.0, v), gp_Dir2d(1, 0)), face, tol)
        return edge

    c_bot, c_top = make_circle(0.0, 0.0, v0), make_circle(twist, height, v1)
    samples = HPoints(1, 33)
    for i in range(1, 34):
        frac = (i - 1) / 32
        samples.SetValue(i, surface.Value(twist * frac, height * frac))
    interp = GeomAPI_Interpolate(samples, False, 1e-9)
    interp.Perform()
    seam = TopoDS_Edge()
    builder.MakeEdge(seam, interp.Curve(), tol)
    builder.Add(seam, v0.Oriented(TopAbs_FORWARD))
    builder.Add(seam, v1.Oriented(TopAbs_REVERSED))
    builder.UpdateEdge(
        seam,
        Geom2d_Line(gp_Pnt2d(2 * math.pi, 0.0), gp_Dir2d(twist, height)),
        Geom2d_Line(gp_Pnt2d(0.0, 0.0), gp_Dir2d(twist, height)),
        face,
        tol,
    )
    builder.Range(seam, face, 0.0, math.hypot(twist, height))
    builder.SameParameter(seam, False)
    builder.SameRange(seam, False)
    wire = TopoDS_Wire()
    builder.MakeWire(wire)
    for edge, orientation in (
        (c_bot, TopAbs_FORWARD),
        (seam, TopAbs_FORWARD),
        (c_top, TopAbs_REVERSED),
        (seam, TopAbs_REVERSED),
    ):
        builder.Add(wire, edge.Oriented(orientation))
    builder.Add(face, wire)
    BRepLib.SameParameter_s(face, 1e-5, False)
    caps = [BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(e).Wire(), True).Face() for e in (c_bot, c_top)]
    for flips in range(4):
        shell = TopoDS_Shell()
        builder.MakeShell(shell)
        builder.Add(shell, face)
        for k, cap in enumerate(caps):
            builder.Add(shell, cap.Reversed() if flips >> k & 1 else cap)
        solid = TopoDS_Solid()
        builder.MakeSolid(solid)
        builder.Add(solid, shell)
        result = cq.Solid(solid)
        if result.isValid() and result.Volume() > 0:
            return result
    raise RuntimeError("no valid helical cylinder")


def _lean_torus_seams(lean):
    """Copy a torus and lean both its vertical (U) and horizontal (V) seams."""
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve2d
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.Geom2d import Geom2d_Line
    from OCP.gp import gp_Dir2d, gp_Pnt2d
    from OCP.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.TopoDS import TopoDS

    copy = cq.Shape.cast(BRepBuilderAPI_Copy(cq.Solid.makeTorus(5, 1).wrapped, True, False).Shape())
    builder = BRep_Builder()
    for face in copy.Faces():
        face_fwd = TopoDS.Face_s(face.wrapped.Oriented(TopAbs_FORWARD))
        seen = []
        for edge in face.Edges():
            ew = edge.wrapped
            if not BRep_Tool.IsClosed_s(ew, face_fwd) or any(ew.IsSame(x) for x in seen):
                continue
            seen.append(ew)
            c1 = BRepAdaptor_Curve2d(TopoDS.Edge_s(ew.Oriented(TopAbs_FORWARD)), face_fwd)
            c2 = BRepAdaptor_Curve2d(TopoDS.Edge_s(ew.Oriented(TopAbs_REVERSED)), face_fwd)
            f1, l1 = c1.FirstParameter(), c1.LastParameter()
            f2, l2 = c2.FirstParameter(), c2.LastParameter()
            p1a, p1b = c1.Value(f1).Coord(), c1.Value(l1).Coord()
            p2a = c2.Value(f2).Coord()
            if abs(p1b[0] - p1a[0]) < abs(p1b[1] - p1a[1]):
                direction = gp_Dir2d(lean, 1.0 if p1b[1] > p1a[1] else -1.0)
            else:
                direction = gp_Dir2d(1.0 if p1b[0] > p1a[0] else -1.0, lean)
            o1 = gp_Pnt2d(p1a[0] - f1 * direction.X(), p1a[1] - f1 * direction.Y())
            o2 = gp_Pnt2d(p2a[0] - f2 * direction.X(), p2a[1] - f2 * direction.Y())
            builder.UpdateEdge(
                ew, Geom2d_Line(o1, direction), Geom2d_Line(o2, direction), face_fwd, BRep_Tool.Tolerance_s(ew)
            )
    return copy


@unittest.skipIf(cq is None, "needs CadQuery")
class SeamStraighteningTest(unittest.TestCase):
    def assert_watertight_and_bounded(self, solid, mesh, max_error_mm):
        self.assertEqual(unmatched_edges(mesh.triangles), [])
        self.assertLessEqual(mesh.surface_error_mm, max_error_mm)
        array = np.array(mesh.triangles)
        normals = np.cross(array[:, 1] - array[:, 0], array[:, 2] - array[:, 0])
        self.assertGreater(np.linalg.norm(normals, axis=1).min(), 2e-10)
        volume = np.einsum("ij,ij->", array[:, 0], normals) / 6
        area = np.linalg.norm(normals, axis=1).sum() / 2
        self.assertGreater(volume, 0)
        self.assertLessEqual(abs(volume - solid.Volume()), area * mesh.surface_error_mm)
        reach = 4 * max_error_mm
        measured = dict(
            surface_to_mesh=distance_to_mesh(surface_points(solid), mesh.triangles, reach),
            edges_to_mesh=distance_to_mesh(edge_points(solid), mesh.triangles, reach),
            mesh_to_surface=distance_to_solid(solid, mesh.triangles),
        )
        for direction, value in measured.items():
            self.assertLessEqual(value, mesh.surface_error_mm, (direction, measured))
        source = mesh.source
        for face in source["face_bounds"]:
            bound = (
                face["interpolation_error_mm"]
                + face["trim_error_mm"]
                + face["edge_weld_mm"]
                + source["cad_topology_error_mm"]
                + source["numerical_guard_mm"]
            )
            self.assertLessEqual(bound, mesh.surface_error_mm)
        self.assertEqual(sum(f["facets"] for f in source["face_bounds"]), len(mesh.triangles))

    def test_without_straightening_repro_fails_with_mesher_omitted_uv_nodes(self):
        solid = _frustum(-1e-15)
        with (
            mock.patch.object(cad, "_straighten_seams", lambda faces, *_: ([0.0] * len(faces), 0)),
            self.assertRaisesRegex(ValueError, "CAD mesher omitted face UV nodes"),
        ):
            export_surface_mesh(solid, max_error_mm=0.04)

    def test_repro_leaning_cone_seam_is_straightened_watertight_and_bounded(self):
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Curve2d
        from OCP.TopAbs import TopAbs_FORWARD
        from OCP.TopoDS import TopoDS

        for lean in (0.0, -1e-15, 1e-15, -9.69e-16, -1e-12, -1e-9):
            with self.subTest(lean=lean):
                solid = _frustum(lean)
                mesh = export_surface_mesh(solid, max_error_mm=0.04)
                self.assert_watertight_and_bounded(solid, mesh, 0.04)
                self.assertFalse(mesh.source["tolerances_tightened_for_meshing"])
                if lean == 0.0:
                    self.assertEqual(mesh.source["seams_straightened"], 0)
                    self.assertEqual(mesh.source["seam_straighten_error_mm"], 0.0)
                else:
                    self.assertEqual(mesh.source["seams_straightened"], 1)
                    self.assertGreater(mesh.source["seam_straighten_error_mm"], 0.0)
                    self.assertLessEqual(mesh.source["seam_straighten_error_mm"], 1e-7)
                    self.assertGreaterEqual(
                        mesh.source["face_bounds"][0]["trim_error_mm"], mesh.source["seam_straighten_error_mm"]
                    )
                    # Caller's BRep must not be mutated.
                    cone = solid.Faces()[0].wrapped
                    seam = next(e.wrapped for e in solid.Faces()[0].Edges() if BRep_Tool.IsClosed_s(e.wrapped, cone))
                    curve = BRepAdaptor_Curve2d(
                        TopoDS.Edge_s(seam.Oriented(TopAbs_FORWARD)), TopoDS.Face_s(cone.Oriented(TopAbs_FORWARD))
                    )
                    self.assertNotEqual(curve.Line().Direction().X(), 0.0)
        fine = _frustum(-1e-15)
        self.assert_watertight_and_bounded(fine, export_surface_mesh(fine, max_error_mm=0.002), 0.002)

    def test_reversed_tapered_hole_in_bushing_straightens_without_swapping_pcurves(self):
        from OCP.TopAbs import TopAbs_REVERSED

        bushing = _tapered_hole_bushing(-9.69e-16)
        cone = next(f for f in bushing.Faces() if f.geomType() == "CONE")
        self.assertEqual(cone.wrapped.Orientation(), TopAbs_REVERSED)
        with (
            mock.patch.object(cad, "_straighten_seams", lambda faces, *_: ([0.0] * len(faces), 0)),
            self.assertRaisesRegex(ValueError, "CAD mesher omitted face UV nodes"),
        ):
            export_surface_mesh(bushing, max_error_mm=0.04)
        mesh = export_surface_mesh(bushing, max_error_mm=0.04)
        self.assertEqual(mesh.source["seams_straightened"], 1)
        self.assertGreater(mesh.source["seam_straighten_error_mm"], 0.0)
        self.assert_watertight_and_bounded(bushing, mesh, 0.04)

    def test_moved_location_bottom_lean_and_loose_matched_slant_are_handled(self):
        from OCP.gp import gp_Trsf, gp_Vec
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS

        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec(50.0, -30.0, 12.0))
        moved = cq.Solid(TopoDS.Solid_s(_frustum(-1e-15).wrapped.Moved(TopLoc_Location(trsf))))
        self.assertFalse(moved.Faces()[0].wrapped.Location().IsIdentity())
        cases = (
            ("moved location", moved, 0.04, 1e-7),
            ("bottom lean 1e-15", _frustum(-1e-15, bottom_lean=True), 0.04, 1e-7),
            ("bottom lean 1e-7", _frustum(-1e-7, tol=1e-6, bottom_lean=True), 0.04, 1e-6),
            ("loose matched slant", _frustum(-2e-4, tol=2e-3, match_top=True), 0.04, 2e-3),
        )
        for label, solid, max_err, edge_tol in cases:
            with self.subTest(case=label):
                mesh = export_surface_mesh(solid, max_error_mm=max_err)
                self.assertEqual(mesh.source["seams_straightened"], 1)
                self.assertGreater(mesh.source["seam_straighten_error_mm"], 0.0)
                self.assertLessEqual(mesh.source["seam_straighten_error_mm"], edge_tol)
                self.assert_watertight_and_bounded(solid, mesh, max_err)

    def test_horizontal_and_vertical_seams_on_torus_and_bspline_seams(self):
        torus = _lean_torus_seams(-1e-12)
        torus_mesh = export_surface_mesh(torus, max_error_mm=0.02)
        self.assertEqual(torus_mesh.source["seams_straightened"], 2)
        self.assertGreater(torus_mesh.source["seam_straighten_error_mm"], 0.0)
        self.assert_watertight_and_bounded(torus, torus_mesh, 0.02)

        bspline_frustum = _frustum(-1e-12, bspline_seam=True)
        bspline_mesh = export_surface_mesh(bspline_frustum, max_error_mm=0.04)
        self.assertEqual(bspline_mesh.source["seams_straightened"], 1)
        self.assertGreater(bspline_mesh.source["seam_straighten_error_mm"], 0.0)
        self.assert_watertight_and_bounded(bspline_frustum, bspline_mesh, 0.04)

    def test_genuinely_slanted_helical_seam_is_not_straightened(self):
        cylinder = _helical_seam_cylinder()
        mesh = export_surface_mesh(cylinder, max_error_mm=0.04)
        self.assertEqual(mesh.source["seams_straightened"], 0)
        self.assertEqual(mesh.source["seam_straighten_error_mm"], 0.0)
        self.assertEqual(unmatched_edges(mesh.triangles), [])
        self.assertLessEqual(mesh.surface_error_mm, 0.04)
        array = np.array(mesh.triangles)
        normals = np.cross(array[:, 1] - array[:, 0], array[:, 2] - array[:, 0])
        self.assertGreater(np.linalg.norm(normals, axis=1).min(), 2e-10)
        volume = np.einsum("ij,ij->", array[:, 0], normals) / 6
        area = np.linalg.norm(normals, axis=1).sum() / 2
        self.assertLessEqual(abs(volume - cylinder.Volume()), area * mesh.surface_error_mm)
        # Exact analytic distance from facet points to the cylinder (radius 2, z in [0, 4]);
        # BRepExtrema_DistShapeShape does not wrap U across a helical seam.
        samples = np.concatenate([array[:, 0], array.mean(1), (array[:, 0] + array[:, 1]) / 2])
        r = np.hypot(samples[:, 0], samples[:, 1])
        z = samples[:, 2]
        to_wall = np.hypot(r - 2.0, np.maximum(0.0, np.maximum(-z, z - 4.0)))
        to_caps = np.hypot(np.maximum(0.0, r - 2.0), np.minimum(np.abs(z), np.abs(z - 4.0)))
        self.assertLessEqual(float(np.minimum(to_wall, to_caps).max()), mesh.surface_error_mm)
        self.assertLessEqual(distance_to_mesh(surface_points(cylinder), mesh.triangles, 0.16), mesh.surface_error_mm)
        self.assertLessEqual(distance_to_mesh(edge_points(cylinder), mesh.triangles, 0.16), mesh.surface_error_mm)

    def test_clean_analytic_solids_leave_exact_seams_untouched(self):
        solids = (
            ("cone frustum", cq.Solid.makeCone(4, 2, 5), 0.01),
            ("cone apex", cq.Solid.makeCone(4, 0, 5), 0.01),
            ("cylinder", cq.Solid.makeCylinder(3, 5), 0.01),
            ("sphere", cq.Workplane("XY").sphere(2).val(), 0.02),
            ("torus", cq.Solid.makeTorus(5, 1), 0.02),
        )
        for label, solid, max_err in solids:
            with self.subTest(shape=label):
                mesh = export_surface_mesh(solid, max_error_mm=max_err)
                self.assertEqual(mesh.source["seams_straightened"], 0)
                self.assertEqual(mesh.source["seam_straighten_error_mm"], 0.0)

    def test_pcurve_straighten_bounds_cover_sampled_deviation_on_lines_and_splines(self):
        from OCP.Geom2d import Geom2d_BezierCurve, Geom2d_BSplineCurve, Geom2d_Line
        from OCP.Geom2dAdaptor import Geom2dAdaptor_Curve
        from OCP.gp import gp_Dir2d, gp_Pnt2d

        try:
            from OCP.TColgp import TColgp_Array1OfPnt2d as Points2d
            from OCP.TColStd import TColStd_Array1OfInteger as Integers, TColStd_Array1OfReal as Reals
        except ImportError:  # OCP 8 moved collection aliases.
            from OCP.collections import Array1_double as Reals, Array1_gp_Pnt2d as Points2d, Array1_int as Integers

        # Vertical and horizontal lines with non-zero parameter offsets.
        vline = Geom2dAdaptor_Curve(Geom2d_Line(gp_Pnt2d(3.0, -1.0), gp_Dir2d(-1e-4, 1.0)), 2.0, 7.0)
        hline = Geom2dAdaptor_Curve(Geom2d_Line(gp_Pnt2d(1.0, 2.5), gp_Dir2d(-1.0, 2e-4)), -3.0, 4.0)
        # Rational BSpline with non-uniform weights leaning in U.
        pts, wts = Points2d(1, 3), Reals(1, 3)
        for i, (u, v, w) in enumerate(((1.0, 0.0, 1.0), (1.0 + 3e-4, 1.5, 0.7), (1.0 - 2e-4, 3.0, 1.0)), 1):
            pts.SetValue(i, gp_Pnt2d(u, v))
            wts.SetValue(i, w)
        knots, mults = Reals(1, 2), Integers(1, 2)
        knots.SetValue(1, 0.0)
        knots.SetValue(2, 3.0)
        mults.SetValue(1, 3)
        mults.SetValue(2, 3)
        bsp = Geom2dAdaptor_Curve(Geom2d_BSplineCurve(pts, wts, knots, mults, 2), 0.0, 3.0)
        bez = Geom2dAdaptor_Curve(Geom2d_BezierCurve(pts, wts), 0.0, 1.0)

        for label, curve, axis, target in (
            ("vertical line", vline, 0, 3.0),
            ("horizontal line", hline, 1, 2.5),
            ("rational bspline", bsp, 0, 1.0),
            ("rational bezier", bez, 0, 1.0),
        ):
            with self.subTest(curve=label):
                first, last = curve.FirstParameter(), curve.LastParameter()
                new_curve, du, dv = cad._pcurve_straighten(curve, first, last, axis, target)
                for i in range(101):
                    t = first + (last - first) * i / 100
                    orig = curve.Value(t).Coord()
                    fixed = new_curve.Value(t).Coord()
                    self.assertEqual(fixed[axis], target)
                    self.assertLessEqual(abs(fixed[0] - orig[0]), du + 1e-14)
                    self.assertLessEqual(abs(fixed[1] - orig[1]), dv + 1e-14)


if __name__ == "__main__":
    unittest.main()
