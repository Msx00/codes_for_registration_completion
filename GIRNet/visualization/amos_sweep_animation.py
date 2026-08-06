import argparse
import os
import re
import yaml
import numpy as np
import torch
from tqdm import tqdm

import vtk
from vtk.util.numpy_support import vtk_to_numpy

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import vtk_utils, pc_utils, data_amos
from models.P_V2S_Net_V5_downsampled_intraop_v2 import PV2SNetV5DownsampledIntraopV2
from models.P_V2S_Net_V5_downsampled_intraop_v2_I2P import PV2SNetV5DownsampledIntraopV2I2P
from models.P_V2S_Net_V5_downsampled_intraop_v2_I2P_dgcnn import PV2SNetV5DownsampledIntraopV2I2PDGCNN


def _model_selector(model_name):
    models = {
        "PV2SNetV5DownsampledIntraopV2": PV2SNetV5DownsampledIntraopV2,
        "PV2SNetV5DownsampledIntraopV2I2P": PV2SNetV5DownsampledIntraopV2I2P,
        "PV2SNetV5DownsampledIntraopV2I2PDGCNN": PV2SNetV5DownsampledIntraopV2I2PDGCNN,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models)}")
    return models[model_name]


def load(checkpoint_dir, model_filename="best_model.pth", device="cuda"):
    """Load and warm up the PIVOTS model from a checkpoint directory."""
    if checkpoint_dir.endswith("/"):
        checkpoint_dir = checkpoint_dir[:-1]

    with open(os.path.join(checkpoint_dir, "params.yaml")) as f:
        params = yaml.safe_load(f)

    cmd_file = os.path.join(os.path.dirname(checkpoint_dir), "command_line_args.yaml")
    cmd = yaml.safe_load(open(cmd_file))

    P_V2S_Net = _model_selector(cmd["model"])
    n_intermediate_features = [200, 150, 110, 80, 60, 50]
    n_intermediate_points = [
        params["n_layer_0_pre_points"],
        params["n_layer_1_pre_points"],
        params["n_layer_2_pre_points"],
        params["n_layer_3_pre_points"],
        params["n_layer_4_pre_points"],
        params["n_layer_5_pre_points"],
    ]

    model = P_V2S_Net(
        n_input_features=5,
        n_preprocess_features=50,
        n_intermediate_features=n_intermediate_features,
        n_intermediate_points=n_intermediate_points,
        n_output_features=3,
        embedding_size=params["embedding_size"],
        points_per_region=params["points_per_region"],
        enc_freq=[2e-2, 2e-1, 2, 4, 8, 16, 32, 64],
        enc_freq_scale=1,
        append_df_self=True,
        append_df_cross=True,
        append_positional_encoding=True,
        compact_return=True,
    )

    state = torch.load(
        os.path.join(checkpoint_dir, model_filename),
        map_location=device,
        weights_only=False,
    )["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        _ = model(
            preop=torch.randn(1, 7, 2500).to(device),
            intraop=torch.randn(1, 6, 2500).to(device),
        )
    print("Model loaded:", cmd["model"])
    return model


_COLOR_PREOP   = np.array([0.5, 0.5, 0.5])
_COLOR_INTRAOP = np.array([0.9, 0.6, 0.1])
_COLOR_ESTIM   = np.array([0.2, 0.2, 1.0])
_COLOR_TUMOR            = np.array([0.9, 0.15, 0.15])   # GT tumors: red
_COLOR_TUMOR_ESTIMATED  = np.array([0.0, 0.82, 0.82])   # estimated tumors: turquoise


def _deform_mesh_with_displacement(mesh, coords_pre, displ):
    """Interpolate the predicted displacement field onto an arbitrary mesh and warp it.

    Mirrors what save_output_as_vtk_dry does for preop_meshes: build a VTK point
    cloud from coords_pre with the displacement array attached, then call
    apply_deformation so the field is scattered/interpolated to the mesh vertices.
    """
    from vtk.util.numpy_support import numpy_to_vtk as _n2v
    mask = np.all(np.abs(coords_pre) < 1e3, axis=1)
    cp = coords_pre[mask]
    d  = displ[mask]
    source = vtk_utils.to_pointcloud(cp)
    arr = _n2v(d)
    arr.SetName("displacement_predicted")
    source.GetPointData().AddArray(arr)
    source.GetPointData().SetActiveVectors("displacement_predicted")
    return vtk_utils.apply_deformation(mesh, source, "displacement_predicted")


def _camera_params(*coords_arrays, eye_direction, up_direction):
    """Compute center and eye from the combined bounding box of all coordinate arrays.

    Ignores dummy points (|x| >= 1e3). Accepts multiple arrays so the camera
    fits all rendered content and every panel shares an identical view.
    """
    valid = []
    for pts in coords_arrays:
        mask = np.all(np.abs(pts) < 1e3, axis=1)
        if mask.any():
            valid.append(pts[mask])
    if not valid:
        valid = [coords_arrays[0]]
    all_pts = np.concatenate(valid, axis=0)
    center = (all_pts.max(axis=0) + all_pts.min(axis=0)) * 0.5
    extent = all_pts.max(axis=0) - all_pts.min(axis=0)
    radius = np.linalg.norm(extent) * 0.9
    eye = center + eye_direction / np.linalg.norm(eye_direction) * radius
    return center, eye


def _vtk_to_o3d_mesh(vtk_mesh):
    """Convert a VTK PolyData or UnstructuredGrid to an Open3D TriangleMesh."""
    import open3d as o3d

    if vtk_mesh is None:
        return None

    if isinstance(vtk_mesh, vtk.vtkUnstructuredGrid):
        geo = vtk.vtkGeometryFilter()
        geo.SetInputData(vtk_mesh)
        geo.Update()
        poly = geo.GetOutput()
    else:
        poly = vtk_mesh

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()
    poly = tri.GetOutput()

    pts = vtk_to_numpy(poly.GetPoints().GetData())
    cells = poly.GetPolys()
    cells.InitTraversal()
    triangles = []
    id_list = vtk.vtkIdList()
    while cells.GetNextCell(id_list):
        if id_list.GetNumberOfIds() == 3:
            triangles.append([id_list.GetId(0), id_list.GetId(1), id_list.GetId(2)])

    if not triangles:
        return None

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pts)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.compute_vertex_normals()
    return mesh


def _setup_lighting(renderer):
    """Add a directional sun light to an OffscreenRenderer scene."""
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.set_sun_light(
        direction=[0.577, -0.577, -0.577],
        color=[1.0, 1.0, 1.0],
        intensity=75000,
    )
    try:
        renderer.scene.scene.enable_ibl(True)
        renderer.scene.scene.set_ibl_intensity(25000)
    except AttributeError:
        pass


def _setup_camera(renderer, center, eye, up, width, height):
    """Position the camera and set an explicit near clip to avoid mesh clipping."""
    renderer.setup_camera(60.0, center, eye, up)
    dist = float(np.linalg.norm(np.asarray(eye, dtype=float) - np.asarray(center, dtype=float)))
    near = max(dist * 0.005, 1e-4)
    far  = dist * 50.0
    aspect = width / height
    try:
        import open3d.visualization.rendering as rendering
        renderer.scene.camera.set_projection(
            60.0, aspect, near, far,
            rendering.Camera.FovType.Vertical,
        )
    except Exception:
        pass


# Filament crashes if more than one OffscreenRenderer exists at a time.
# We keep a single instance and clear its scene between renders.
_g_renderer: object = None
_g_renderer_size: tuple = (0, 0)


def _reset_renderer():
    """Destroy the current renderer so Filament state is fully flushed.

    Must be called only when no local variable holds a reference to the renderer
    (i.e., between frames, not inside a render loop). Recreating mid-loop would
    produce two simultaneous Filament instances and crash the process.
    """
    global _g_renderer, _g_renderer_size
    import gc
    if _g_renderer is not None:
        del _g_renderer
        _g_renderer = None
        gc.collect()
    _g_renderer_size = (0, 0)


def _get_renderer(width, height):
    """Return the shared OffscreenRenderer, creating it if necessary."""
    global _g_renderer, _g_renderer_size
    import open3d as o3d
    if _g_renderer is None or _g_renderer_size != (width, height):
        if _g_renderer is not None:
            del _g_renderer
            _g_renderer = None
        _g_renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        _g_renderer_size = (width, height)
        _setup_lighting(_g_renderer)
    return _g_renderer


def _panel(specs, center, eye, up, width, height):
    """Render a mixed list of point clouds and meshes; return a PIL Image.

    specs: list of dicts, each one of:
      {"type": "pcd",  "pts": ndarray (N,3), "color": ndarray (3,), "point_size": float}
      {"type": "mesh", "vtk_mesh": vtkObject, "rgba": [r, g, b, a]}

    Meshes are rendered individually as opaque (defaultLit) against a chromakey
    background and alpha-composited in software, which avoids z-fighting and
    gives correct shading regardless of IBL support.  Point clouds are rendered
    in a single pass and painted on top, fully opaque.
    """
    import open3d as o3d
    from PIL import Image as PILImage

    # Bright magenta chromakey — none of our mesh/pcd colors are magenta.
    _BG = [1.0, 0.0, 1.0, 1.0]

    def _make_renderer():
        r = _get_renderer(width, height)
        r.scene.clear_geometry()
        r.scene.set_background(_BG)
        return r

    def _fg_mask(arr):
        """True for pixels that are NOT the magenta background."""
        return ~((arr[:, :, 0] > 220) & (arr[:, :, 1] < 35) & (arr[:, :, 2] > 220))

    # Float canvas starting at white.
    canvas = np.ones((height, width, 3), dtype=np.float32) * 255.0

    # --- Meshes: one render per mesh, alpha-composited back-to-front ---
    for spec in specs:
        if spec["type"] != "mesh":
            continue
        vtk_mesh = spec.get("vtk_mesh")
        if vtk_mesh is None:
            continue
        o3d_mesh = _vtk_to_o3d_mesh(vtk_mesh)
        if o3d_mesh is None:
            continue

        r = _make_renderer()
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        rgba = spec["rgba"]
        mat.base_color = [rgba[0], rgba[1], rgba[2], 1.0]
        r.scene.add_geometry("mesh", o3d_mesh, mat)
        _setup_camera(r, center, eye, up, width, height)

        img = np.asarray(r.render_to_image()).astype(np.float32)
        mask = _fg_mask(img.astype(np.uint8))
        alpha = rgba[3]
        canvas[mask] = canvas[mask] * (1.0 - alpha) + img[mask] * alpha

    # --- Point clouds: single render pass, painted fully opaque on top ---
    pcd_renderer = _make_renderer()
    pcd_added = 0
    for i, spec in enumerate(specs):
        if spec["type"] != "pcd":
            continue
        pts = spec["pts"]
        valid = np.all(np.abs(pts) < 1e3, axis=1)
        pts = pts[valid]
        if len(pts) == 0:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = spec.get("point_size", 6.0)
        c = spec["color"]
        mat.base_color = [float(c[0]), float(c[1]), float(c[2]), 1.0]
        pcd_renderer.scene.add_geometry(f"pcd_{i}", pcd, mat)
        pcd_added += 1

    if pcd_added > 0:
        _setup_camera(pcd_renderer, center, eye, up, width, height)
        img = np.asarray(pcd_renderer.render_to_image()).astype(np.float32)
        mask = _fg_mask(img.astype(np.uint8))
        canvas[mask] = img[mask]

    return PILImage.fromarray(canvas.clip(0, 255).astype(np.uint8))


def _add_legend(img, panel_width, height):
    """Draw panel titles, vertical separators, and a color legend onto the composite image."""
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img)
    W = img.width

    # Vertical panel separators
    for x in [panel_width, 2 * panel_width]:
        draw.line([(x, 0), (x, height)], fill=(160, 160, 160), width=2)

    # Font — try common system paths, fall back to PIL built-in
    font_size = int(max(22, height // 38) * 1.5)
    font = None
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    def _tw(text):
        try:
            return int(draw.textlength(text, font=font))
        except Exception:
            try:
                return font.getsize(text)[0]
            except Exception:
                return len(text) * font_size // 2

    # Panel titles
    titles = ["Simulation Meshes", "Network Input Point Clouds", "Network Output (and Ground Truth)"]
    for i, title in enumerate(titles):
        x = i * panel_width + (panel_width - _tw(title)) // 2
        draw.text((x, 10), title, fill=(30, 30, 30), font=font)

    # Color legend at bottom, centered across full width
    legend = [
        ((128, 128, 128), "Preoperative"),
        ((230, 153,  26), "Intraoperative"),
        ( (51,  51, 255), "Estimated"),
    ]
    sw  = font_size   # swatch size
    gap = 8           # swatch → label gap
    sep = 40          # between items

    item_widths = [sw + gap + _tw(lbl) for _, lbl in legend]
    total_w = sum(item_widths) + sep * (len(legend) - 1)
    x = (W - total_w) // 2
    y = height - sw - 14

    for (color, lbl), iw in zip(legend, item_widths):
        draw.rectangle([x, y, x + sw, y + sw], fill=color, outline=(80, 80, 80), width=1)
        draw.text((x + sw + gap, y + (sw - font_size) // 2), lbl, fill=(30, 30, 30), font=font)
        x += iw + sep


def _add_zoom_inset(composite, img_right, vtk_tumor_estimated,
                    cam_center, eye, up_hint,
                    panel_width, height,
                    box_size=420, pad=20, mm_per_unit=1000.0, fov_deg=60.0):
    """Overlay a zoom-in inset at the bottom-right of the composite image.

    Crops `img_right` around the estimated tumor A centroid, scales it up, adds
    a perspective-correct mm scale bar, draws a source-region indicator on the
    right panel, and pastes the inset with a thin border onto the composite.

    mm_per_unit: world units → mm conversion (1000 when data is in metres).
    The scale bar is computed at the tumour's actual depth (perspective-correct).
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont as PILImageFont

    if vtk_tumor_estimated is None:
        return

    # --- Camera basis ---
    eye_v      = np.asarray(eye,        dtype=float)
    center_v   = np.asarray(cam_center, dtype=float)
    up_v       = np.asarray(up_hint,    dtype=float)
    f = center_v - eye_v;  f /= np.linalg.norm(f)
    r = np.cross(f, up_v); r /= np.linalg.norm(r)
    u = np.cross(r, f);    u /= np.linalg.norm(u)
    tan_half = np.tan(np.radians(fov_deg / 2))
    aspect   = panel_width / height

    def _project(pts_3d):
        v   = pts_3d - eye_v
        xc  = v @ r
        yc  = v @ u
        zc  = v @ f
        valid = zc > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = np.where(valid, xc / (zc * aspect * tan_half), 0.0)
            yn = np.where(valid, yc / (zc * tan_half),          0.0)
        px = (xn + 1) * panel_width  / 2
        py = (1 - yn) * height / 2
        return np.stack([px, py], axis=1), zc, valid

    # --- Project tumor vertices ---
    try:
        pts_3d = vtk_to_numpy(vtk_tumor_estimated.GetPoints().GetData())
    except Exception:
        return
    if len(pts_3d) == 0:
        return

    pix, depths, valid = _project(pts_3d)
    if not valid.any():
        return
    pix_v   = pix[valid]
    depth_v = depths[valid]

    # --- Determine crop region (square, centred on tumour projection) ---
    cx = float(pix_v[:, 0].mean())
    cy = float(pix_v[:, 1].mean())
    half = max(
        (pix_v[:, 0].max() - pix_v[:, 0].min()) / 2,
        (pix_v[:, 1].max() - pix_v[:, 1].min()) / 2,
    ) * 2.2 + 30
    half = max(half, 40.0)

    crop_x0 = int(max(0,           cx - half))
    crop_y0 = int(max(0,           cy - half))
    crop_x1 = int(min(panel_width, cx + half))
    crop_y1 = int(min(height,      cy + half))
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        return

    # Make square, re-centre
    crop_sz   = max(crop_x1 - crop_x0, crop_y1 - crop_y0)
    crop_x0   = max(0,           int(cx - crop_sz / 2))
    crop_y0   = max(0,           int(cy - crop_sz / 2))
    crop_x1   = min(panel_width, crop_x0 + crop_sz)
    crop_y1   = min(height,      crop_y0 + crop_sz)
    actual_w  = crop_x1 - crop_x0
    actual_h  = crop_y1 - crop_y0

    crop_img = img_right.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    zoom_img = crop_img.resize((box_size, box_size), PILImage.LANCZOS)

    # --- Perspective-correct scale bar ---
    mean_depth        = float(depth_v.mean())
    pixels_per_unit   = height / (2.0 * mean_depth * tan_half)
    pix_per_mm_orig   = pixels_per_unit / mm_per_unit
    zoom_factor       = box_size / max(actual_w, 1)
    pix_per_mm_zoom   = pix_per_mm_orig * zoom_factor

    bar_mm = 10  # fallback
    for candidate in [1, 2, 5, 10, 20, 50, 100]:
        bar_px_candidate = candidate * pix_per_mm_zoom
        if 0.20 * box_size <= bar_px_candidate <= 0.65 * box_size:
            bar_mm = candidate
            break
    bar_px = int(round(bar_mm * pix_per_mm_zoom))

    draw = ImageDraw.Draw(zoom_img)

    # White background strip at bottom
    strip_h = max(40, box_size // 10)
    draw.rectangle([0, box_size - strip_h, box_size, box_size], fill=(255, 255, 255))

    # Load font first so text width is known for centering bar + label together
    font_s = max(20, strip_h * 4 // 5)
    font   = None
    for fpath in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            font = PILImageFont.truetype(fpath, font_s)
            break
        except Exception:
            pass
    if font is None:
        font = PILImageFont.load_default()

    label = f"{bar_mm} mm"
    try:
        tw = int(draw.textlength(label, font=font))
    except Exception:
        tw = len(label) * font_s // 2

    # Layout: [bar] [gap] [label], centred as a unit in the strip
    gap          = 10
    total_w      = bar_px + gap + tw
    bar_x0       = max(8, (box_size - total_w) // 2)
    bar_x1       = bar_x0 + bar_px
    bar_h        = max(5, strip_h // 6)
    bar_center_y = box_size - strip_h // 2
    bar_y0       = bar_center_y - bar_h // 2
    bar_y1       = bar_center_y + bar_h // 2
    tick_h       = bar_h * 2

    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], fill=(40, 40, 40))
    draw.rectangle([bar_x0,     bar_y0 - tick_h, bar_x0 + 2, bar_y1], fill=(40, 40, 40))
    draw.rectangle([bar_x1 - 2, bar_y0 - tick_h, bar_x1,     bar_y1], fill=(40, 40, 40))

    # Label to the right of bar, vertically centred on bar
    draw.text(
        (bar_x1 + gap, bar_center_y - font_s // 2),
        label, fill=(40, 40, 40), font=font,
    )



    # --- Border + paste onto composite ---
    border = 3
    W_comp = composite.width
    H_comp = composite.height
    ix = W_comp - pad - box_size - border * 2
    iy = H_comp - pad - box_size - border * 2
    draw_c = ImageDraw.Draw(composite)

    # Draw source indicator rectangle on the right panel portion of the composite
    rx0 = 2 * panel_width + crop_x0
    ry0 = crop_y0
    rx1 = 2 * panel_width + crop_x1
    ry1 = crop_y1
    draw_c.rectangle([rx0, ry0, rx1, ry1], outline=(0, 0, 0), width=5)


    draw_c.rectangle([ix - 1, iy - 1,
                      ix + box_size + border * 2 + 1,
                      iy + box_size + border * 2 + 1],
                     fill=(100, 100, 100))
    composite.paste(zoom_img, (ix + border, iy + border))


def _render_composite(coords_pre, coords_estimated,
                       vtk_preop_mesh, vtk_intraop_full_mesh, vtk_intraop_partial_mesh,
                       output_path,
                       coords_intra=None,
                       vtk_tumor_gt_meshes=None,
                       vtk_tumor_estimated_meshes=None,
                       eye_direction=np.array([0.0, 0.0, 1.0]),
                       up_direction=np.array([0.0, 1.0, 0.0]),
                       fixed_center=None, fixed_eye=None,
                       panel_width=1280, height=960,
                       mm_per_unit=1000.0):
    """Render a 3-panel composite image (3*panel_width x height):
      Left:   preop mesh (gray) + deforming intraop full mesh (orange), both transparent
      Center: preop point cloud (gray, small) + partial intraop surface mesh (orange, transparent)
      Right:  intraop full mesh (orange, transparent) + estimated output point cloud (blue)
    """
    from PIL import Image as PILImage

    if fixed_center is not None and fixed_eye is not None:
        center = np.asarray(fixed_center, dtype=float)
        eye    = np.asarray(fixed_eye,    dtype=float)
    else:
        center, eye = _camera_params(coords_pre, coords_estimated,
                                     eye_direction=eye_direction, up_direction=up_direction)
    up = up_direction / np.linalg.norm(up_direction)
    c = _COLOR_INTRAOP

    tg = _COLOR_TUMOR
    te = _COLOR_TUMOR_ESTIMATED
    left_specs = [
        {"type": "mesh", "vtk_mesh": vtk_preop_mesh,         "rgba": [0.5, 0.5, 0.5, 0.3]},
        {"type": "mesh", "vtk_mesh": vtk_intraop_full_mesh,  "rgba": [c[0], c[1], c[2], 0.4]},
    ]
    for tm in (vtk_tumor_gt_meshes or []):
        left_specs.append({"type": "mesh", "vtk_mesh": tm, "rgba": [tg[0], tg[1], tg[2], 0.7]})
    img_left = _panel(left_specs, center, eye, up, panel_width, height)

    center_specs = [
        {"type": "pcd",  "pts": coords_pre,               "color": _COLOR_PREOP, "point_size": 5.0},
        {"type": "mesh", "vtk_mesh": vtk_intraop_partial_mesh, "rgba": [c[0], c[1], c[2], 0.6]},
    ]
    if coords_intra is not None:
        center_specs.append(
            {"type": "pcd", "pts": coords_intra, "color": _COLOR_INTRAOP, "point_size": 6.0}
        )
    img_center = _panel(center_specs, center, eye, up, panel_width, height)

    right_specs = [
        {"type": "mesh", "vtk_mesh": vtk_intraop_full_mesh,  "rgba": [c[0], c[1], c[2], 0.4]},
        {"type": "pcd",  "pts": coords_estimated,             "color": _COLOR_ESTIM, "point_size": 6.0},
    ]
    for tm in (vtk_tumor_gt_meshes or []):
        right_specs.append({"type": "mesh", "vtk_mesh": tm, "rgba": [tg[0], tg[1], tg[2], 0.7]})
    for tm in (vtk_tumor_estimated_meshes or []):
        right_specs.append({"type": "mesh", "vtk_mesh": tm, "rgba": [te[0], te[1], te[2], 0.85]})
    img_right = _panel(right_specs, center, eye, up, panel_width, height)

    composite = PILImage.new("RGB", (3 * panel_width, height))
    composite.paste(img_left,   (0, 0))
    composite.paste(img_center, (panel_width, 0))
    composite.paste(img_right,  (2 * panel_width, 0))
    _add_legend(composite, panel_width, height)
    if vtk_tumor_estimated_meshes:
        _add_zoom_inset(
            composite, img_right, vtk_tumor_estimated_meshes[0],
            center, eye, up,
            panel_width, height,
            mm_per_unit=mm_per_unit,
        )
    composite.save(output_path)
    print("Saved:", output_path)


def _show_on_screen(coords_pre, coords_intra, coords_estimated):
    """Display three point clouds interactively."""
    import open3d as o3d

    colors = [
        np.array([0.5, 0.5, 0.5]),
        np.array([0.9, 0.6, 0.1]),
        np.array([0.2, 0.2, 1.0]),
    ]
    pcds = []
    for pts, color in zip([coords_pre, coords_intra, coords_estimated], colors):
        mask = np.all(np.abs(pts) < 1e3, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pts), 1)))
        pcds.append(pcd)
    if pcds:
        o3d.visualization.draw_geometries(pcds)


def _frame_index(path):
    """Extract integer frame index from a filename like liver_camera_0_f42.vtp."""
    m = re.search(r'_f(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1


def _create_gifs(output_dir, frame_duration_ms=200):
    """Collect all per-frame PNGs in output_dir, group by base name (prefix before _fN),
    sort by frame index, and write one GIF per group into output_dir."""
    from PIL import Image

    groups = {}
    for subdir in os.listdir(output_dir):
        subdir_path = os.path.join(output_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for fname in os.listdir(subdir_path):
            if not fname.endswith(".png"):
                continue
            stem = os.path.splitext(fname)[0]
            base = re.sub(r'_f\d+', '', stem)
            groups.setdefault(base, []).append(os.path.join(subdir_path, fname))

    for base, paths in groups.items():
        paths.sort(key=_frame_index)
        frames = [Image.open(p) for p in paths]
        gif_path = os.path.join(output_dir, f"{base}.gif")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )
        print(f"  Saved GIF ({len(frames)} frames): {gif_path}")


def parse_directory(model, input_dir, output_dir, preop_filename, intraop_surface_regex,
                    intraop_full_template="liver_surface_f{frame}.stl",
                    scale=1.0, npoints=2500, device="cuda", visualize=False):
    """Apply PIVOTS to every intraop file in input_dir matching intraop_surface_regex.

    Saves per-frame results to output_dir/<intraop_stem>/.
    """
    preop_path = os.path.join(input_dir, preop_filename)
    if not os.path.exists(preop_path):
        print(f"  Skipping {input_dir}: {preop_filename} not found")
        return

    regex = re.compile(intraop_surface_regex)
    intraop_files = sorted(
        [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if regex.search(f) and os.path.isfile(os.path.join(input_dir, f))
        ],
        key=_frame_index,
    )
    intraop_files = [f for f in intraop_files if _frame_index(f) != 0]

    if not intraop_files:
        print(f"  Skipping {input_dir}: no intraop files match '{intraop_surface_regex}'")
        return

    print(f"  {os.path.basename(input_dir)}: preop + {len(intraop_files)} intraop frames")
    os.makedirs(output_dir, exist_ok=True)

    # Compute the centering offset from the preop mesh once (needed for save_output_as_vtk_dry).
    # Also extract vertices of the centered mesh to fix the camera across all frames.
    _preop_mesh = vtk_utils.load_mesh(preop_path)
    _preop_centered, transform_center = vtk_utils.center_mesh(_preop_mesh)
    preop_verts = vtk_to_numpy(_preop_centered.GetPoints().GetData())
    cam_top_center, cam_top_eye = _camera_params(
        preop_verts,
        eye_direction=np.array([0.0, 0.0, 1.0]),
        up_direction=np.array([0.0, 1.0, 0.0]),
    )
    cam_front_center, cam_front_eye = _camera_params(
        preop_verts,
        eye_direction=np.array([0.0, 1.0, 0.0]),
        up_direction=np.array([0.0, 0.0, 1.0]),
    )

    # Discover tumors in this scene from f0 files, load and center the preop meshes once.
    tumor_names = sorted({
        m.group(1)
        for f in os.listdir(input_dir)
        for m in [re.match(r"tumor_(\w+)_surface_f0\.stl", f)]
        if m
    })
    tumor_preop_meshes = {}
    for name in tumor_names:
        p = os.path.join(input_dir, f"tumor_{name}_surface_f0.stl")
        tumor_preop_meshes[name] = vtk_utils.transform_mesh(vtk_utils.load_mesh(p), transform_center)
    if tumor_names:
        print(f"  Found {len(tumor_names)} tumor(s): {tumor_names}")

    with torch.no_grad():
        for intraop_path in tqdm(intraop_files, desc=os.path.basename(input_dir), leave=False):
            fidx = _frame_index(intraop_path)
            intraop_basename = os.path.basename(intraop_path)
            intraop_full_filename = intraop_full_template.format(frame=fidx)
            intraop_full_path = os.path.join(input_dir, intraop_full_filename)

            if not os.path.exists(intraop_full_path):
                print(f"    Skipping frame {fidx}: {intraop_full_filename} not found")
                continue

            intraop_stem = os.path.splitext(intraop_basename)[0]
            frame_dir = os.path.join(output_dir, intraop_stem)
            predictions_path = os.path.join(frame_dir, "predictions.npz")

            try:
                _reset_renderer()
                if os.path.exists(predictions_path):
                    # Fast path: skip model inference, load cached predictions.
                    npz = np.load(predictions_path)
                    coords_pre = npz["coords_pre"]
                    coords_estimated = npz["coords_estimated"]
                    centering_offset = tuple(npz["centering_offset"].tolist())
                    coords_intra = npz["coords_intra"] if "coords_intra" in npz else None

                    vtk_preop_mesh = vtk_utils.load_mesh(preop_path)
                    vtk_preop_mesh, _ = vtk_utils.center_mesh(vtk_preop_mesh)
                    vtk_intraop_full_mesh = vtk_utils.transform_mesh(
                        vtk_utils.load_mesh(intraop_full_path), centering_offset)
                    vtk_intraop_partial_mesh = vtk_utils.transform_mesh(
                        vtk_utils.load_mesh(intraop_path), centering_offset)
                else:
                    # Slow path: run model inference and cache results.
                    sample = data_amos.LiverSampleAMOS(
                        path=input_dir,
                        int_id=0,
                        check_for_files=False,
                        scale=scale,
                        return_all_intraop=True,
                        preop_filename=preop_filename,
                        intraop_full_filename=intraop_full_filename,
                        intraop_filename=intraop_basename,
                    )
                    data = sample.load(npoints=npoints)

                    preop_tensor = torch.FloatTensor(data["preop"]).unsqueeze(0).to(device)
                    intraop_tensor = torch.FloatTensor(data["intraop"][0]).unsqueeze(0).to(device)

                    predictions = model(preop=preop_tensor, intraop=intraop_tensor)

                    coords_pre = preop_tensor.cpu().numpy()[0, :3, :].T
                    coords_intra = intraop_tensor.cpu().numpy()[0, :3, :].T
                    displ = predictions[-1]["result"].cpu().numpy()[0, :3, :].T
                    coords_estimated = coords_pre + displ

                    vtk_utils.save_output_as_vtk_dry(
                        coords_pre=coords_pre,
                        coords_intra=coords_intra,
                        displ=displ,
                        preop_meshes=[preop_path],
                        folder=frame_dir,
                        scale=scale,
                        transform_center=transform_center,
                    )

                    vtk_preop_mesh = sample.geometry.get("preop_volume") or sample.geometry.get("preop_surface")
                    vtk_intraop_full_mesh = sample.geometry.get("intraop_volume")
                    intraop_surface_list = sample.geometry.get("intraop_surface") or []
                    vtk_intraop_partial_mesh = intraop_surface_list[0] if intraop_surface_list else None

                    os.makedirs(frame_dir, exist_ok=True)
                    np.savez(
                        predictions_path,
                        coords_pre=coords_pre,
                        coords_intra=coords_intra,
                        coords_estimated=coords_estimated,
                        centering_offset=np.array(transform_center),
                    )

                # Load GT tumor meshes for this frame (apply same centering as the liver).
                vtk_tumor_gt_meshes = []
                for name in tumor_preop_meshes:
                    gt_path = os.path.join(input_dir, f"tumor_{name}_surface_f{fidx}.stl")
                    if os.path.exists(gt_path):
                        vtk_tumor_gt_meshes.append(
                            vtk_utils.transform_mesh(vtk_utils.load_mesh(gt_path), transform_center))

                # Estimate tumor positions by interpolating the predicted displacement field.
                displ_for_tumor = coords_estimated - coords_pre
                vtk_tumor_estimated_meshes = [
                    _deform_mesh_with_displacement(tm, coords_pre, displ_for_tumor)
                    for tm in tumor_preop_meshes.values()
                ]

                _render_composite(
                    coords_pre=coords_pre,
                    coords_estimated=coords_estimated,
                    vtk_preop_mesh=vtk_preop_mesh,
                    vtk_intraop_full_mesh=vtk_intraop_full_mesh,
                    vtk_intraop_partial_mesh=vtk_intraop_partial_mesh,
                    output_path=os.path.join(frame_dir, f"{intraop_stem}_top.png"),
                    coords_intra=coords_intra,
                    vtk_tumor_gt_meshes=vtk_tumor_gt_meshes,
                    vtk_tumor_estimated_meshes=vtk_tumor_estimated_meshes,
                    eye_direction=np.array([0.0, 0.0, 1.0]),
                    up_direction=np.array([0.0, 1.0, 0.0]),
                    fixed_center=cam_top_center,
                    fixed_eye=cam_top_eye,
                )
                _render_composite(
                    coords_pre=coords_pre,
                    coords_estimated=coords_estimated,
                    vtk_preop_mesh=vtk_preop_mesh,
                    vtk_intraop_full_mesh=vtk_intraop_full_mesh,
                    vtk_intraop_partial_mesh=vtk_intraop_partial_mesh,
                    output_path=os.path.join(frame_dir, f"{intraop_stem}_front.png"),
                    coords_intra=coords_intra,
                    vtk_tumor_gt_meshes=vtk_tumor_gt_meshes,
                    vtk_tumor_estimated_meshes=vtk_tumor_estimated_meshes,
                    eye_direction=np.array([0.0, 1.0, 0.0]),
                    up_direction=np.array([0.0, 0.0, 1.0]),
                    fixed_center=cam_front_center,
                    fixed_eye=cam_front_eye,
                )
                if visualize and coords_intra is not None:
                    _show_on_screen(coords_pre, coords_intra, coords_estimated)
            except Exception as e:
                print(f"    Skipping frame {fidx}: {type(e).__name__}: {e}")

    _create_gifs(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PIVOTS on AMOS sweep data and save per-frame deformed meshes"
    )
    parser.add_argument("input_dir", help="Root directory with one scene subfolder per sample")
    parser.add_argument("checkpoint", help="Path to checkpoint subfolder (e.g. checkpoints/run/0)")
    parser.add_argument("--output_dir", default="outputs/amos_sweep")
    parser.add_argument("--model_filename", default="best_model.pth")
    parser.add_argument("--preop_volume", default="liver_volume_f0.vtk")
    parser.add_argument("--intraop_surface_regex", default=r"liver_surface_partial.*.vtp")
    parser.add_argument("--intraop_full_template", default="liver_surface_f{frame}.stl",
                        help="Template for the full intraop surface filename. {frame} is replaced by the frame index.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor (use 1e-3 if data is in mm)")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "xpu"])
    parser.add_argument("--visualize", action="store_true", help="Show each frame interactively (blocks until window is closed)")
    args = parser.parse_args()

    model = load(args.checkpoint, model_filename=args.model_filename, device=args.device)

    subdirs = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    print(f"Found {len(subdirs)} subdirectories in {args.input_dir}")

    for subdir in tqdm(subdirs, desc="scenes"):
        if subdir >= "000018":# and subdir < "000015":
        #if subdir == "000005":
            parse_directory(
                model=model,
                input_dir=os.path.join(args.input_dir, subdir),
                output_dir=os.path.join(args.output_dir, subdir),
                preop_filename=args.preop_volume,
                intraop_surface_regex=args.intraop_surface_regex,
                intraop_full_template=args.intraop_full_template,
                scale=args.scale,
                device=args.device,
                visualize=args.visualize,
            )