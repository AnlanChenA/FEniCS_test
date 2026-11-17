"""
全分辨率(400x400)严谨版本: 用 microstructure-ve (plastic 分支, DOLFINx 后端) 复现
VF_2_ParVF_1_Agg_2 的 ABAQUS bilinear elastoplastic 结果。

跟 testfinx/run_msve_comparison.py 的区别:
  - COARSEN 固定为 1, 用完整 400x400 网格 (约32万自由度), 不做降采样冒烟测试。
  - 加载点不再是均匀 linspace, 而是直接取 testfinx/S-Eabaqus.txt 里 ABAQUS 记录的
    13 个应变点, 这样两条曲线在完全相同的应变上做逐点比较, 不需要插值, 更严谨。

材料/几何/边界条件的映射关系跟 run_msve_comparison.py 里写的完全一样 (400x400像素,
scale=0.0025um/pixel, interphase_image 0/1/2=matrix/interphase/particle, 双线性
塑性硬化, 周期性边界条件 + free-lateral 单轴 x 向拉伸 —— 注意: 不是 confined,
TOPLEFT 的 y 位移在原 .inp 里从未被约束或驱动, 详见此前的讨论)。

free-lateral 路径每个 Static step 内部要做 20 个子增量 x 外层根查找(最多25次) x
牛顿迭代(每次都要重新LU分解约32万自由度的复数矩阵), 13 个 step 顺序求解, 预计要
远超之前 50x50 粗网格版本的耗时, 可能几十分钟到数小时, 请耐心等待。

跑法 (同一个 conda 环境, 见仓库根目录 README 的 Setup 部分):
    conda activate microstructure-ve
    python run_msve_full.py
"""
import os

# 必须在 import numpy/petsc4py/dolfinx 之前设, 否则底层 BLAS 会自己开满所有核心。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time

import numpy as np
import scipy.io as sio
from scipy import interpolate

from microstructure_ve.core import GridNodes, GridElements, ElementSet
from microstructure_ve.materials import Material, PlasticMaterial
from microstructure_ve.boundary import (
    PeriodicBoundaryCondition, FixedBoundaryCondition, DisplacementBoundaryCondition,
)
from microstructure_ve.steps import Model, Simulation, Step, Static, Heading

# ------------------------------------------------------------------ 1. 输入/输出路径
# 原始 .mat/.txt 输入和参考曲线放在仓库根目录(跟 run_msve_comparison.py 共用,
# 不在 full_resolution/ 里重复存一份), 本脚本自己的输出(.inp / S-Emsve-full.txt)
# 仍然写回 full_resolution/ 这个文件夹
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE)
DATA_DIR = f"{ROOT_DIR}/data"
INTPH_IMAGE_MAT = f"{DATA_DIR}/b=0.03_e=0.929_z=0_UniIP_IP1_20pix_intph.mat"  # 400x400, 0/1/2
UNI_LEVEL_TXT = f"{DATA_DIR}/uni_level.txt"          # 界面层 [NormE, deltaTg]
YIELD_MAT = f"{DATA_DIR}/Temp_Yield_PMMA.mat"        # Temp_sigmaY: [T, sigma_y, ...]

OUT_DIR = BASE
ABAQUS_SE_TXT = f"{ROOT_DIR}/S-Eabaqus.txt"      # 前半=应力(MPa), 后半=应变

# ------------------------------------------------------------------ 2. 材料参数 (硬编码, 抄自
#    agglgradcomp_intph_CPE4R_scaled_response_v4.py 的 run_FEA / assignMatPara)
scale = 0.0025          # micron / pixel
Rou_mat = 1.18e-15
E0_mat = 3500.0
v0_mat = 0.3
Rou_Ptc = 2.65e-15
E_Ptc = 70000.0
v_Ptc = 0.3
Etan = 500.0
T_base = 293.0           # T_operate
STRAIN_CAP = 0.15        # *Plastic 表第二个点固定写死在 15% 总应变处


def bilinear_plastic_table(sigma_y0, E):
    """复刻 InputFile.write('*Plastic', ...) 里的两点表: (yield, 0), (sigma@15%, eps_pl@15%)."""
    sigma2 = sigma_y0 + Etan * (STRAIN_CAP - sigma_y0 / E)
    eps_pl2 = STRAIN_CAP - sigma2 / E
    return [sigma_y0, sigma2], [0.0, eps_pl2]


def load_target_strains():
    """testfinx/S-Eabaqus.txt: 前半是应力数组, 后半是应变数组, 等长; 只取应变。"""
    se = np.loadtxt(ABAQUS_SE_TXT)
    n = len(se) // 2
    strains = se[n:]
    assert np.all(np.diff(strains) > 0), "expected ascending strains in S-Eabaqus.txt"
    return strains


def build_simulation():
    # --- 微结构图像: 0=matrix, 1=interphase, 2=particle (全分辨率 400x400, 不降采样) ---
    img = sio.loadmat(INTPH_IMAGE_MAT)["interphase_image"].astype(int)
    Y, X = img.shape
    assert (X, Y) == (400, 400), f"expected 400x400, got {img.shape}"

    # --- 界面层 NormE (来自 uni_level.txt 第一行第一列) ---
    uni_bin = np.loadtxt(UNI_LEVEL_TXT)
    norm_e_int = float(uni_bin[0])
    delta_tg_int = float(uni_bin[1])
    E0_int = E0_mat * norm_e_int

    # --- 屈服应力表 T -> sigma_y (线性插值+外推, 与 scipy.interpolate.interp1d 行为一致) ---
    yield_data = sio.loadmat(YIELD_MAT)["Temp_sigmaY"]
    f_yield = interpolate.interp1d(
        yield_data[:, 0], yield_data[:, 1], bounds_error=False, fill_value="extrapolate"
    )
    Y0_mat = float(f_yield(T_base))
    Y0_int = float(f_yield(T_base - delta_tg_int))

    mat_ys, mat_eps = bilinear_plastic_table(Y0_mat, E0_mat)
    int_ys, int_eps = bilinear_plastic_table(Y0_int, E0_int)

    # --- 几何: 节点/单元网格 (2D 平面应变, CPE4 全积分 —— DOLFINx Q1 的紧配对;
    #     legacy ABAQUS 用的是 CPE4R 缩减积分, 二者预期有 ~3-4% 的系统性差异,
    #     这是 dolfinx_backend_design.md §5/§9 里记录过的已知 gap, 不是 bug) ---
    nodes = GridNodes.from_matl_img(img, scale=scale)
    elements = GridElements(nodes, type="CPE4")

    matrix_elset, interphase_elset, particle_elset = ElementSet.from_matl_img(img)

    materials = [
        Material(particle_elset, density=Rou_Ptc, poisson=v_Ptc, youngs=E_Ptc),
        PlasticMaterial(matrix_elset, density=Rou_mat, poisson=v0_mat, youngs=E0_mat,
                         yield_stress=mat_ys, plastic_strain=mat_eps),
        PlasticMaterial(interphase_elset, density=Rou_mat, poisson=v0_mat, youngs=E0_int,
                         yield_stress=int_ys, plastic_strain=int_eps),
    ]

    # --- 边界条件: 周期性 + free-lateral 单轴 x 向拉伸 (对应 legacy .inp 里
    #     BOTMLEFT=origin 全约束, BOTMRIGHT=x_drive 压 y, TOPLEFT=y_drive 只压 x,
    #     y 位移始终未被约束/驱动 -> E_yy 自由收缩) ---
    origin, x_drive, y_drive = (nodes.nsets[k] for k in ("X0Y0", "X1Y0", "X0Y1"))
    bcs = [
        PeriodicBoundaryCondition(nodes=nodes),
        FixedBoundaryCondition(origin, dofs=[1, 2]),
        FixedBoundaryCondition(x_drive, dofs=[2]),
        FixedBoundaryCondition(y_drive, dofs=[1]),
        DisplacementBoundaryCondition(x_drive, 1, 1, 0.0),  # baseline
    ]
    model = Model(nodes=nodes, elements=elements, materials=materials, bcs=bcs)

    # --- 加载步: 直接取 S-Eabaqus.txt 里 ABAQUS 记录的应变点, 逐点重建, 便于跟
    #     ABAQUS 曲线做完全同应变点的比较 (不需要插值) ---
    Lx = X * scale  # == 1.0 um
    target_strains = load_target_strains()
    disp_targets = target_strains * Lx

    steps = [
        Step(subsections=[Static(), DisplacementBoundaryCondition(x_drive, 1, 1, float(d))],
             perturbation=False)
        for d in disp_targets
    ]

    sim = Simulation(heading=Heading("VF_2_ParVF_1_Agg_2 full-res bilinear plastic vs ABAQUS"),
                      model=model, steps=steps)
    return sim, Lx, target_strains


def main():
    t0 = time.time()
    sim, Lx, target_strains = build_simulation()
    n_steps = len(list(sim.steps))
    print(f"built simulation: {n_steps} Static steps at ABAQUS's strain points, "
          f"{sim.model.nodes.shape} node grid ({time.time()-t0:.1f}s)")

    from microstructure_ve.backends.abaqus import write_inp
    inp_path = f"{OUT_DIR}/vf2_parvf1_agg2_full_check.inp"
    write_inp(sim, inp_path)
    print(f"wrote {inp_path} for a manual sanity diff against the legacy .inp")

    from microstructure_ve.backends.dolfinx import run
    t1 = time.time()
    rows = run(sim)  # (13, 7): [frame, RF_x, RF_y, 0, 0, U_x, U_y]
    print(f"dolfinx solve done in {time.time()-t1:.1f}s")

    RF_x = rows[:, 1]
    U_x = rows[:, 5]
    stress = RF_x / Lx        # cross_area = Ly = Lx = 1.0 micron here, 平面应变单位厚度
    strain = U_x / Lx

    # 输出成跟 S-Eabaqus.txt / S-Emsve.txt 一样的格式: 前半应力, 后半应变
    out_path = f"{OUT_DIR}/S-Emsve-full.txt"
    with open(out_path, "w") as f:
        for s in stress:
            f.write(f"{s}\n")
        for e in strain:
            f.write(f"{e}\n")
    print(f"wrote {out_path}")

    # 跟 ABAQUS 在完全相同的应变点上逐点比较 (无需插值)
    se = np.loadtxt(ABAQUS_SE_TXT)
    n = len(se) // 2
    abq_stress = se[:n]

    assert np.allclose(strain, target_strains, rtol=1e-6), \
        "solved strains drifted from the requested targets -- check the run"

    rel_err = np.abs(stress - abq_stress) / abq_stress
    print("\nstrain\t\tmsve(MPa)\tabaqus(MPa)\trel_err")
    for e, m, a, r in zip(strain, stress, abq_stress, rel_err):
        print(f"{e:.6f}\t{m:.4f}\t{a:.4f}\t{r*100:.2f}%")
    print(f"\nmax relative stress error: {rel_err.max()*100:.2f}%")
    print(f"mean relative stress error: {rel_err.mean()*100:.2f}%")
    print(f"total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
