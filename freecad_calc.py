"""
Анализ набора винтов на предмет собственных частот через FreeCAD (gmsh + ccx)

Запуск:
  QT_QPA_PLATFORM=offscreen freecadcmd /path/to/freecad_calc.py
"""

import os
import csv
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import FreeCAD as App
import Fem
import ObjectsFem
from femtools import ccxtools


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(_SCRIPT_DIR, "models/generated_enhanced_1000")
OUTPUT_CSV = os.path.join(_SCRIPT_DIR, "results_gen_enhanced.csv")

EIGENMODES_COUNT = 12

# Закрепление торца резьбы (минимальный Y, ось винта Y).
#   TraceParts (/models): цилиндр резьбы разбит на 2 симметричные половинки → Face4,Face7
#   generated_calibrated_* (как TraceParts): торец резьбы max Y → Face12
#
# Режимы FIXED_FACE:
#   "Face4,Face7" / "Face1" — явный список через запятую
#   "auto" — грани на верхнем торце резьбы (y = YMax, см. _find_thread_end_faces)
FIXED_FACE = "Face1"


def _find_geometry(doc):
    for obj in doc.Objects:
        if obj.isDerivedFrom("Part::Feature") and hasattr(obj, "Shape") and not obj.Shape.isNull():
            if obj.Shape.Solids:
                return obj
    active = doc.ActiveObject
    if active and active.isDerivedFrom("Part::Feature"):
        return active
    raise RuntimeError("После импорта STEP не найден твёрдотельный объект Part::Feature.")


def _find_thread_end_faces(geo_obj, y_tol=1e-4, planar_span=1e-3):

    if not geo_obj.Shape.Faces:
        return []
    y_end = geo_obj.Shape.BoundBox.YMax

    candidates: list[tuple[int, float]] = []  # (index, y_span)
    for index, face in enumerate(geo_obj.Shape.Faces, start=1):
        bb = face.BoundBox
        if bb.YMax >= y_end - y_tol:
            candidates.append((index, bb.YMax - bb.YMin))

    if not candidates:
        return []

    planar = [idx for idx, span in candidates if span <= planar_span]
    if planar:
        return [f"Face{idx}" for idx in planar]

    min_span = min(span for _, span in candidates)
    chosen = [idx for idx, span in candidates if span <= min_span + 1e-6]
    return [f"Face{idx}" for idx in chosen]


def _resolve_fixed_faces(geo_obj):
    text = (FIXED_FACE or "").strip()
    if text.lower() != "auto":
        parts = [p.strip() for p in text.split(",") if p.strip()]
        resolved = []
        for p in parts:
            try:
                face_index = int(p.replace("Face", ""))
            except ValueError:
                continue
            if 1 <= face_index <= len(geo_obj.Shape.Faces):
                resolved.append(f"Face{face_index}")
        if resolved:
            return resolved

    auto_faces = _find_thread_end_faces(geo_obj)
    if auto_faces:
        return auto_faces
    raise RuntimeError(
        f"Не удалось определить грань(и) закрепления (FIXED_FACE={FIXED_FACE!r})."
    )


def _add_thread_fixation(doc, analysis, geo_obj):
    face_names = _resolve_fixed_faces(geo_obj)

    con_fixed = ObjectsFem.makeConstraintFixed(doc, "Fix_Thread")
    con_fixed.References = [(geo_obj, name) for name in face_names]
    analysis.addObject(con_fixed)
    print(f"[INFO] Закрепление: {geo_obj.Name}." + ",".join(face_names))


def _configure_material(material):
    # Сталь: E=200 GPa, nu=0.3, rho=7900 kg/m^3
    mat = material.Material
    mat["Name"] = "Steel-Generic"
    mat["YoungsModulus"] = "200 GPa"
    mat["PoissonRatio"] = "0.30"
    mat["Density"] = "7900 kg/m^3"
    material.Material = mat


def _configure_gmsh_mesh(
    fem_mesh,
    h_max: float = 0.0,
    h_min: float = 0.0,
    curvature: float = 12.0,
):
    # Параметры сетки Gmsh
    fem_mesh.Algorithm2D = "Automatic"
    fem_mesh.Algorithm3D = "Automatic"
    fem_mesh.CharacteristicLengthMax = h_max
    fem_mesh.CharacteristicLengthMin = h_min
    fem_mesh.CoherenceMesh = True
    fem_mesh.ElementDimension = "3D"
    fem_mesh.ElementOrder = "2nd"
    fem_mesh.GeometryTolerance = 0.0
    fem_mesh.GroupsOfNodes = True
    fem_mesh.HighOrderOptimize = "None"
    fem_mesh.MeshSizeFromCurvature = int(curvature)
    fem_mesh.OptimizeNetgen = False
    fem_mesh.OptimizeStd = True
    fem_mesh.RecombinationAlgorithm = "Simple"
    fem_mesh.Recombine3DAll = False
    fem_mesh.RecombineAll = False
    fem_mesh.SecondOrderLinear = False
    fem_mesh.SubdivisionAlgorithm = "None"


def _make_calculix_solver(doc):
    makers = (
        "makeSolverCalculiXCcxTools",
        "makeSolverCalculixCcxTools",
        "makeSolverCalculix",
    )
    for name in makers:
        factory = getattr(ObjectsFem, name, None)
        if callable(factory):
            return factory(doc, "CalculiX_Solver")

    obj = doc.addObject("Fem::FemSolverObjectPython", "CalculiX_Solver")
    from femobjects import solver_ccxtools

    solver_ccxtools.SolverCcxTools(obj)
    return obj


def _configure_frequency_solver(solver):
    # Параметры солвера CalculiX
    solver.WorkingDir = ""
    solver.AnalysisType = "frequency"
    solver.ThermoMechSteadyState = True
    solver.ThermoMechType = "coupled"
    solver.BeamReducedIntegration = True
    solver.ExcludeBendingStiffness = False
    solver.ModelSpace = "3D"
    solver.BeamShellResultOutput3D = True
    solver.BucklingAccuracy = 0.01
    solver.BucklingFactors = 1
    solver.EigenmodeHighLimit = 1_000_000.0  # 1 MHz
    solver.EigenmodeLowLimit = 0.0
    solver.EigenmodesCount = EIGENMODES_COUNT
    solver.GeometricalNonlinearity = "linear"
    solver.IterationsControlParameterCutb = "0.25,0.5,0.75,0.85,,,1.5,"
    solver.IterationsControlParameterIter = "4,8,9,200,10,400,,200,,"
    solver.IterationsControlParameterTimeUse = False
    solver.MaterialNonlinearity = "linear"
    solver.MatrixSolverType = "default"
    solver.OutputFrequency = 1
    solver.PastixMixedPrecision = False
    solver.SplitInputWriter = False
    solver.AutomaticIncrementation = True
    solver.IncrementsMaximum = 2000
    solver.TimeInitialIncrement = 1.0
    solver.TimeMaximumIncrement = 1.0
    solver.TimeMinimumIncrement = 0.0
    solver.TimePeriod = 1.0


def _generate_gmsh_mesh(fem_mesh):
    try:
        from femexamples.meshes.generate_mesh import mesh_from_mesher

        if mesh_from_mesher(fem_mesh, "gmsh"):
            return None
        return "Gmsh mesh_from_mesher returned False"
    except ImportError:
        pass

    try:
        from femmesh.gmsh import mesh as gmsh_mesh_mod

        tool = gmsh_mesh_mod.Mesh(fem_mesh)
        err = tool.create_mesh()
        return err if err else None
    except ImportError:
        from femmesh.gmshtools import GmshTools

        tool = GmshTools(fem_mesh)
        if hasattr(tool, "run"):
            if not tool.run(blocking=True):
                return "GmshTools.run() failed"
            return None
        err = tool.create_mesh()
        return err if err else None


def _read_frequencies_from_dat(dat_path):
    from feminout.importCcxDatResults import readResult

    if not os.path.isfile(dat_path):
        return []

    modes = readResult(dat_path)
    freqs = [m["frequency"] for m in modes if "frequency" in m]
    return freqs[:EIGENMODES_COUNT]


def _run_calculix(fea):

    fea.update_objects()
    fea.setup_working_dir()

    message = fea.check_prerequisites()
    if message:
        raise RuntimeError(f"Проверка FEM: {message}")

    fea.write_inp_file()
    if not fea.inp_file_name:
        raise RuntimeError("Не удалось записать входной файл CalculiX (.inp).")

    ret_code = fea.ccx_run()
    if ret_code is None:
        raise RuntimeError("CalculiX (ccx) не найден. Установите пакет calculix или укажите путь в настройках FEM.")
    if ret_code != 0:
        raise RuntimeError(f"CalculiX завершился с кодом {ret_code}.")

    return fea.inp_file_name


def process_single_model(
    file_path,
    *,
    mesh_h_max: float = 0.0,
    mesh_h_min: float = 0.0,
    mesh_curvature: float = 12.0,
    fixed_face: str | None = None,
    quiet: bool = False,
):
    file_name = os.path.basename(file_path)
    if not quiet:
        print(f"\n[INFO] Обработка файла: {file_name}")

    t_start = time.perf_counter()
    doc = App.newDocument("AnalysisDoc")
    frequencies = None
    node_count = 0
    tet_count = 0

    try:
        import Import

        Import.insert(file_path, doc.Name)
        geo_obj = _find_geometry(doc)

        analysis = ObjectsFem.makeAnalysis(doc, "FEM_Analysis")

        material = ObjectsFem.makeMaterialSolid(doc, "MechanicalMaterial")
        _configure_material(material)
        analysis.addObject(material)

        saved_face = FIXED_FACE
        if fixed_face is not None:
            globals()["FIXED_FACE"] = fixed_face
        try:
            _add_thread_fixation(doc, analysis, geo_obj)
        finally:
            if fixed_face is not None:
                globals()["FIXED_FACE"] = saved_face

        fem_mesh = ObjectsFem.makeMeshGmsh(doc, "Gmsh_Mesh")
        fem_mesh.Shape = geo_obj
        _configure_gmsh_mesh(
            fem_mesh,
            h_max=mesh_h_max,
            h_min=mesh_h_min,
            curvature=mesh_curvature,
        )
        analysis.addObject(fem_mesh)
        doc.recompute()

        if not quiet:
            print("[INFO] Генерация сетки через Gmsh...")
        mesh_error = _generate_gmsh_mesh(fem_mesh)
        if mesh_error:
            raise RuntimeError(f"Gmsh: {mesh_error}")

        if not fem_mesh.FemMesh or fem_mesh.FemMesh.NodeCount == 0:
            raise RuntimeError("Сетка пуста после генерации Gmsh.")

        node_count = fem_mesh.FemMesh.NodeCount
        tet_count = fem_mesh.FemMesh.VolumeCount
        if not quiet:
            print(f"[INFO] Сетка: {node_count} узлов, {tet_count} тетраэдров")
        doc.recompute()

        solver = _make_calculix_solver(doc)
        _configure_frequency_solver(solver)
        analysis.addObject(solver)
        doc.recompute()

        if not quiet:
            print("[INFO] Запуск CalculiX...")
        fea = ccxtools.CcxTools(solver=solver)
        fea.purge_results()
        inp_file = _run_calculix(fea)

        dat_file = os.path.splitext(inp_file)[0] + ".dat"
        frequencies = _read_frequencies_from_dat(dat_file)

        if not frequencies:
            raise RuntimeError(f"Не удалось прочитать частоты из {dat_file}")

        if not quiet:
            print(f"[SUCCESS] Найдено частот (Hz): {frequencies}")
        stats = {
            "nodes": node_count,
            "tets": tet_count,
            "mesh_h_max": mesh_h_max,
            "mesh_h_min": mesh_h_min,
            "mesh_curvature": mesh_curvature,
        }
        return frequencies, time.perf_counter() - t_start, stats

    except Exception as e:
        if not quiet:
            print(f"[ERROR] Ошибка при обработке {file_name}: {e}")
        return None, time.perf_counter() - t_start, {}

    finally:
        App.closeDocument(doc.Name)


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"[ERROR] Папка с моделями не найдена: {INPUT_DIR}")
        sys.exit(1)

    step_files = sorted(
        f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".stp", ".step"))
    )
    if not step_files:
        print(f"[ERROR] В {INPUT_DIR} нет файлов .stp / .step")
        sys.exit(1)

    headers = (
        ["File Name"]
        + [f"Freq_{i + 1} (Hz)" for i in range(EIGENMODES_COUNT)]
        + ["Time (s)"]
    )

    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for file in step_files:
            full_path = os.path.join(INPUT_DIR, file)
            freqs, elapsed, _ = process_single_model(full_path)
            time_str = f"{elapsed:.2f}"
            print(f"[INFO] Время обработки {file}: {time_str} с")

            if freqs:
                row = [file] + freqs + [""] * (EIGENMODES_COUNT - len(freqs)) + [time_str]
                writer.writerow(row)
            else:
                writer.writerow([file] + ["Error/Failed"] * EIGENMODES_COUNT + [time_str])

    print(f"\n[INFO] Готово. Результаты: {OUTPUT_CSV}")



if __name__ == "__main__":
    main()
