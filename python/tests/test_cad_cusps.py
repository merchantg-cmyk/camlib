"""Tangent-cusp UV trim self-intersection repair for CAD solid export (#360).

At a zero-angle tangent cusp, OpenCascade's BRepMesh discretises the two
meeting curves independently. Because the non-crossing condition
``k_more * d_more > k_less * d_less`` is scale-independent, finer deflection
alone never resolves a self-intersecting UV trim polygon.

When the kernel's seed triangulation fails trim matching, camlib checks each
face's UV trim polygon with exact 2D predicates, bisects offending curved
chords at their mid parameter on the real CAD edge in every incident face so
shared edges stay welded, and lets the kernel triangulate with those nodes.

Checked here independently without trusting the adapter: bitwise edge matching,
positive facet areas, enclosed volume, dense trim points (especially near the
cusp tip) within ``max_error_mm`` of the mesh boundary, no facet extending
outside its CAD face, refusal when the bisection cap is hit, and native engine
verification.
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


def _circle_y(radius, x):
    return radius - math.sqrt(radius * radius - x * x)


def _cusp_prism(r1, r2, span, ox=0.0, oy=0.0, height=3.0, mode="spline_circle"):
    """Prism whose planar end faces have a zero-angle tangent cusp at (ox, oy)."""
    wp = cq.Workplane("XY").moveTo(ox, oy)
    if mode in ("spline_circle", "spline_spline"):
        lower = [(ox + span * k / 8, oy + _circle_y(r2, span * k / 8)) for k in range(1, 9)]
        wp = wp.spline(
            lower,
            tangents=[(1, 0), (1, span / math.sqrt(r2 * r2 - span * span))],
            includeCurrent=True,
        )
    else:
        wp = wp.threePointArc((ox + span / 2, oy + _circle_y(r2, span / 2)), (ox + span, oy + _circle_y(r2, span)))
    wp = wp.lineTo(ox + span, oy + _circle_y(r1, span))
    if mode == "spline_spline":
        upper = [(ox + span * (8 - k) / 8, oy + _circle_y(r1, span * (8 - k) / 8)) for k in range(1, 9)]
        wp = wp.spline(
            upper,
            tangents=[(-1, -span / math.sqrt(r1 * r1 - span * span)), (-1, 0)],
            includeCurrent=True,
        )
    else:
        wp = wp.threePointArc((ox + span / 2, oy + _circle_y(r1, span / 2)), (ox, oy))
    return wp.close().extrude(height).val()


def _repro_solid():
    """Exact synthetic reproducer from issue #360."""
    return _cusp_prism(6.1986, 7.8300, 1.7439, ox=1.5484, oy=-2.2660, height=3.0, mode="spline_circle")


def _cusp_tip_points(r1, r2, span, ox, oy, height=3.0):
    """Dense interior points inside the narrow cusp tip on both planar end faces."""
    pts = []
    for frac in (0.005, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75):
        x = span * frac
        ymid = 0.5 * (_circle_y(r1, x) + _circle_y(r2, x))
        for z in (0.0, height):
            pts.append((ox + x, oy + ymid, z))
    return np.array(pts)


def _dense_trim_points(shape, n=64):
    """Dense true points on every CAD edge, concentrated near both ends."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve

    pts = []
    for edge in shape.Edges():
        if BRep_Tool.Degenerated_s(edge.wrapped):
            continue
        curve = BRepAdaptor_Curve(edge.wrapped)
        a, b = curve.FirstParameter(), curve.LastParameter()
        # Uniform samples plus geometric clustering near both endpoints (cusp tips).
        fracs = [i / n for i in range(n + 1)]
        for k in range(1, 10):
            f = 0.5**k
            fracs.extend((f, 1.0 - f))
        for f in fracs:
            pts.append(curve.Value(a + (b - a) * f).Coord())
    return np.array(pts)


def _all_facet_points_to_solid(shape, triangles):
    """Maximum distance from every facet's corners, edge midpoints and centroid
    to the CAD solid (checks that no facet extends outside any CAD face)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt

    shell = cq.Compound.makeCompound(shape.Faces())
    worst = 0.0
    for a, b, c in triangles:
        samples = (
            a,
            b,
            c,
            ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2),
            ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2, (b[2] + c[2]) / 2),
            ((c[0] + a[0]) / 2, (c[1] + a[1]) / 2, (c[2] + a[2]) / 2),
            ((a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3),
        )
        for p in samples:
            vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*p)).Vertex()
            dist = BRepExtrema_DistShapeShape(vertex, shell.wrapped)
            assert dist.IsDone()
            worst = max(worst, dist.Value())
    return worst


def _split_repro_solid(cut_distance):
    """Issue #360 repro split across the cusp at `cut_distance` mm from the tip."""
    from OCP.BOPAlgo import BOPAlgo_Splitter
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

    solid = _repro_solid()
    ox = 1.5484
    plane = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(ox + cut_distance, 0.0, 0.0), gp_Dir(1, 0, 0))).Face()
    splitter = BOPAlgo_Splitter()
    splitter.AddArgument(solid.wrapped)
    splitter.AddTool(plane)
    splitter.Perform()
    assert not splitter.HasErrors()
    solids = cq.Shape.cast(splitter.Shape()).Solids()
    assert len(solids) == 2
    return min(solids, key=lambda s: s.Volume())


def _two_edge_tangent_lens():
    """Prism whose planar end faces have two tangent cusps and zero straight edges."""
    span = 3.0
    lower = [(span * k / 8, 0.30 * (k / 8) ** 2 * (1 - k / 8) ** 2) for k in range(1, 9)]
    upper = [(span * (8 - k) / 8, 0.45 * ((8 - k) / 8) ** 2 * (1 - (8 - k) / 8) ** 2) for k in range(1, 9)]
    return (
        cq.Workplane("XY")
        .moveTo(0.0, 0.0)
        .spline(lower, tangents=[(1, 0), (1, 0)], includeCurrent=True)
        .spline(upper, tangents=[(-1, 0), (-1, 0)], includeCurrent=True)
        .close()
        .extrude(2.0)
        .val()
    )


def _cusp_with_hole():
    """Cusp prism with a circular through-hole in the wider part of the planar face."""
    r1, r2, span = 4.0, 8.0, 3.2
    solid = _cusp_prism(r1, r2, span, height=2.0, mode="spline_circle")
    cx = span * 0.86
    cy = 0.5 * (_circle_y(r1, cx) + _circle_y(r2, cx))
    hole_r = 0.28 * (_circle_y(r1, cx) - _circle_y(r2, cx))
    return solid.cut(cq.Solid.makeCylinder(hole_r, 4.0, cq.Vector(cx, cy, -1.0)))


class ExactSegmentPredicateTest(unittest.TestCase):
    def test_exact_orient2d_handles_near_collinear_large_coordinates(self):
        # At ~250 mm offset, a 1-ULP perturbation is lost in naive float arithmetic
        # when subtracting coordinates before taking the cross product.
        a = (250.0, -180.0)
        b = (251.0, -179.0)
        c_above = (250.5, math.nextafter(-179.5, 0.0))
        c_below = (250.5, math.nextafter(-179.5, -math.inf))
        c_on = (250.5, -179.5)
        self.assertEqual(cad._orient2d_sign(a, b, c_above), 1)
        self.assertEqual(cad._orient2d_sign(a, b, c_below), -1)
        self.assertEqual(cad._orient2d_sign(a, b, c_on), 0)

    def test_segments_improperly_intersect_crossings_and_foldbacks(self):
        # Proper crossing
        self.assertTrue(cad._segments_improperly_intersect((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0), False))
        # Disjoint
        self.assertFalse(cad._segments_improperly_intersect((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), False))
        # Adjacent normal corner vs 180-degree foldback
        self.assertFalse(cad._segments_improperly_intersect((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 1.0), True))
        self.assertTrue(cad._segments_improperly_intersect((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.5, 0.0), True))
        # Repro cusp crossing: spline chord (1->2) crosses arc chord (6->7)
        p1, p2 = (-0.889453, -0.079231), (-0.109132, -0.040252)
        p6, p7 = (0.281970, 0.032464), (-0.301085, -0.051244)
        self.assertTrue(cad._segments_improperly_intersect(p1, p2, p6, p7, False))


@unittest.skipIf(cq is None, "needs CadQuery")
class TangentCuspExportTest(unittest.TestCase):
    def assert_valid_watertight_surface(
        self, shape, surface, requested_error, extra_points=None, check_all_facets=False
    ):
        self.assertTrue(shape.isValid())
        declared = surface.surface_error_mm
        self.assertLessEqual(declared, requested_error)
        self.assertEqual(unmatched_edges(surface.triangles), [])
        array = np.array(surface.triangles)
        normals = np.cross(array[:, 1] - array[:, 0], array[:, 2] - array[:, 0])
        self.assertGreater(np.linalg.norm(normals, axis=1).min(), 2e-10)
        volume = np.einsum("ij,ij->", array[:, 0], normals) / 6
        area = np.linalg.norm(normals, axis=1).sum() / 2
        self.assertGreater(volume, 0.0)
        self.assertLessEqual(abs(volume - shape.Volume()), area * declared + 1e-9)

        source = surface.source
        self.assertEqual(sum(f["facets"] for f in source["face_bounds"]), len(surface.triangles))
        for face in source["face_bounds"]:
            bound = (
                face["interpolation_error_mm"]
                + face["trim_error_mm"]
                + face["edge_weld_mm"]
                + source["collinear_facet_error_mm"]
                + source["cad_topology_error_mm"]
                + source["numerical_guard_mm"]
            )
            self.assertLessEqual(bound, declared + 1e-12)

        reach = 4 * requested_error
        measured = dict(
            surface_to_mesh=distance_to_mesh(surface_points(shape), surface.triangles, reach),
            edges_to_mesh=distance_to_mesh(edge_points(shape), surface.triangles, reach),
            dense_trims_to_mesh=distance_to_mesh(_dense_trim_points(shape), surface.triangles, reach),
            mesh_to_surface=(
                _all_facet_points_to_solid(shape, surface.triangles)
                if check_all_facets
                else distance_to_solid(shape, surface.triangles)
            ),
        )
        if extra_points is not None and len(extra_points):
            measured["tip_to_mesh"] = distance_to_mesh(extra_points, surface.triangles, reach)
        for direction, value in measured.items():
            self.assertLessEqual(value, declared + 1e-9, (direction, declared, measured))

    def test_issue_360_repro_meshes_watertight_at_required_errors(self):
        solid = _repro_solid()
        tip_pts = _cusp_tip_points(6.1986, 7.8300, 1.7439, 1.5484, -2.2660, 3.0)
        for max_error in (0.04, 0.06, 0.1, 0.01, 0.005, 0.001):
            with self.subTest(max_error_mm=max_error):
                surface = export_surface_mesh(solid, max_error_mm=max_error)
                self.assert_valid_watertight_surface(
                    solid, surface, max_error, extra_points=tip_pts, check_all_facets=(max_error >= 0.04)
                )
        # At 0.06 mm (the failing case on main), repair bisects the offending chords
        # and produces deterministic output.
        first = export_surface_mesh(solid, max_error_mm=0.06)
        second = export_surface_mesh(solid, max_error_mm=0.06)
        self.assertGreater(first.source["cusp_trim_bisections"], 0)
        self.assertEqual(first.triangles, second.triangles)
        self.assertEqual(first.surface_error_mm, second.surface_error_mm)

    def test_tilted_production_scale_sharp_cusp_and_split_tip_faces(self):
        tilted = _repro_solid().rotate((0, 0, 0), (1, 2, 3), 37).translate((250.0, -180.0, 42.0))
        surface = export_surface_mesh(tilted, max_error_mm=0.06)
        self.assertGreater(surface.source["cusp_trim_bisections"], 0)
        self.assert_valid_watertight_surface(tilted, surface, 0.06, check_all_facets=True)

        for r2 in (10.05, 10.005):
            with self.subTest(r2=r2):
                sharp = _cusp_prism(10.0, r2, 2.0, mode="spline_circle")
                tip_pts = _cusp_tip_points(10.0, r2, 2.0, 0.0, 0.0, 3.0)
                sharp_mesh = export_surface_mesh(sharp, max_error_mm=0.05)
                self.assertGreater(sharp_mesh.source["cusp_trim_bisections"], 0)
                self.assert_valid_watertight_surface(
                    sharp, sharp_mesh, 0.05, extra_points=tip_pts, check_all_facets=True
                )

        for cut in (0.3, 0.6):
            with self.subTest(split_cut_mm=cut):
                tip_solid = _split_repro_solid(cut)
                tip_surface = export_surface_mesh(tip_solid, max_error_mm=0.06)
                self.assertGreater(tip_surface.source["cusp_trim_bisections"], 0)
                self.assert_valid_watertight_surface(tip_solid, tip_surface, 0.06, check_all_facets=True)

    def test_two_edge_tangent_lens_and_cusp_with_hole(self):
        lens = _two_edge_tangent_lens()
        self.assert_valid_watertight_surface(lens, export_surface_mesh(lens, max_error_mm=0.05), 0.05)

        holed = _cusp_with_hole()
        self.assert_valid_watertight_surface(holed, export_surface_mesh(holed, max_error_mm=0.05), 0.05)

    def test_hitting_bisection_cap_refuses_and_names_cusp(self):
        sharp = _cusp_prism(10.0, 10.005, 2.0, mode="spline_circle")
        with mock.patch.object(cad, "_MAX_CUSP_BISECTIONS_PER_FACE", 1):
            with self.assertRaisesRegex(ValueError, "cusp"):
                export_surface_mesh(sharp, max_error_mm=0.05)

    def test_exported_repro_mesh_passes_native_material_verifier_and_limits(self):
        from array import array
        from camlib.material import SurfaceTarget, Verifier
        from camlib.simulation import StockSpec

        solid = _repro_solid()
        surface = export_surface_mesh(solid, max_error_mm=0.06)
        box = solid.BoundingBox()
        stock = StockSpec((box.xmin - 0.5, box.ymin - 0.5, 0.0), (box.xmax + 0.5, box.ymax + 0.5, 4.0), 0.25)
        nx = math.ceil((stock.max_mm[0] - stock.min_mm[0]) / stock.cell_size_mm)
        ny = math.ceil((stock.max_mm[1] - stock.min_mm[1]) / stock.cell_size_mm)
        heights = array("d", [4.0, 4.0]) * (nx * ny)
        report = Verifier(SurfaceTarget("cusp", "1", surface, 0.0)).check_bounds(heights, stock)
        self.assertTrue(report["complete"])
        self.assertEqual(report["gouge"]["definite_cells"], 0)
        self.assertGreater(report["residual"]["volume_mm3"][1], 0.0)

        with self.assertRaisesRegex(ValueError, "max_facets"):
            export_surface_mesh(solid, max_error_mm=0.06, max_facets=10)
        with self.assertRaisesRegex(ValueError, "cancelled"):
            export_surface_mesh(solid, max_error_mm=0.06, cancel=lambda: True)


if __name__ == "__main__":
    unittest.main()
