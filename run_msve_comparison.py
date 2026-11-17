"""
用 microstructure-ve (plastic 分支, DOLFINx 后端) 复现已有的 ABAQUS bilinear
elastoplastic 结果，用于验证该代码库是否能得到同样的应力-应变曲线。

对应关系（从 /FEA/uniw/files/{run_MS.py, testcase_12.py,
agglgradcomp_intph_CPE4R_scaled_response_v4.py} 反推得到，细节见随附说明）：

  几何:    400x400 像素, scale = 0.0025 micron/pixel -> RVE 边长 Lx=Ly=1.0 micron
  材料编码: interphase_image 中 0=matrix, 1=interphase(单层, 20像素), 2=particle
            (ElementSet.from_matl_img 按数值升序排序，顺序正好对应)
  颗粒:     E=70000 MPa, nu=0.3, rho=2.65e-15 kg/micron^3 (纯弹性)
  基体:     E0=3500 MPa, nu=0.3, rho=1.18e-15, 双线性 (bilinear) 塑性硬化
  界面层:   E0=3500*NormE MPa (NormE 取自 uni_level.txt 第一列),
            nu=0.3, rho=1.18e-15 (代码里界面密度用的也是 Rou_mat, 不是 Rou_int),
            双线性塑性硬化, 屈服应力由 Temp_Yield_PMMA.mat 的 T-sigma_y 表插值得到
  边界条件: 周期性边界条件(PeriodicBoundaryCondition) + confined 单轴 x 向拉伸
            (原 .inp 里 BOTMLEFT 全约束, BOTMRIGHT 只压 y, TOPLEFT 压 x,
            BOTMRIGHT 施加 x 位移) —— 与 microstructure_ve README quickstart
            里的 "confined" x-drive 完全一致
  加载:     单调准静态, 最终工程应变 = 0.105*scale*X / (X*scale) = 0.105 (10.5%)
            *Plastic 表在 15% 总应变处截断(代码写死 0.15), 之后按 ABAQUS
            *Plastic 惯例做完美塑性外推 (PlasticMaterial/ dolfinx 后端一致)

在 /Users/anlan/anaconda3/envs/microstructure-ve 这个 conda 环境（已确认装好了
microstructure_ve + dolfinx + dolfinx_mpc + basix + petsc4py）里直接跑:

    /Users/anlan/anaconda3/envs/microstructure-ve/bin/python3.12 run_msve_comparison.py

400x400 像素 = 401x401 节点 = 约 32 万自由度。这比 docs/dolfinx_backend_design.md
里 2D benchmark 记录的最大档 (256^2, 13.2万dof, LU一次~20-30s) 还大不少，而这里
每个 Static step 内部要做多次牛顿迭代、每次迭代都要重新LU分解一次同样规模的复数
矩阵，41 个 step 顺序算完可能要几十分钟到数小时。所以脚本默认先用 COARSEN=4
把图像降采样到 100x100（RVE物理尺寸不变，只是网格粗一些）快速跑通整条流程、
看曲线形状和量级是否对得上，确认没问题后再把 COARSEN 改成 1、N_STEPS 加大到
40+ 去跑全分辨率的定量比较。
"""
import os

# 必须在 import numpy/petsc4py/dolfinx 之前设, 否则底层 BLAS 会自己开满所有核心
# 跑, 既没什么速度收益(design doc里测过LU求解基本是单线程瓶颈), 又白白发热。
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

# ------------------------------------------------------------------ 1. 输入文件路径
BASE = "/Users/anlan/Desktop/elastoplastic"
CASE_DIR = f"{BASE}/FEA/uniw/VF_2_ParVF_1_Agg_2"
INTPH_IMAGE_MAT = f"{CASE_DIR}/b=0.03_e=0.929_z=0_UniIP_IP1_20pix_intph.mat"  # 400x400, 0/1/2
UNI_LEVEL_TXT = f"{CASE_DIR}/uni_level.txt"          # 界面层 [NormE, deltaTg]
YIELD_MAT = f"{BASE}/FEA/uniw/files/Temp_Yield_PMMA.mat"  # Temp_sigmaY: [T, sigma_y, ...]
ABAQUS_SE_TXT = f"{BASE}/FEA/uniw/S-E.txt"            # 已有 ABAQUS 结果:
                                                        # 前半段=应力(MPa), 后半段=应变
                                                        # !!确认这份 S-E.txt 确实是
                                                        # VF_2_ParVF_1_Agg_2 这个微结构
                                                        # 跑出来的，而不是别的样本!!

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

COARSEN = 8               # 1=全分辨率400x400; 8=先粗网格50x50, 负载小、发热少
N_STEPS = 4 if COARSEN > 1 else 41  # 把 0 -> 0.105 拉伸拆成多少个 Static step
                          # (辐射回映射对单调比例加载是增量无关的，所以分多少步不
                          #  影响每个应变点的解，只影响曲线分辨率)


def bilinear_plastic_table(sigma_y0, E):
    """复刻 InputFile.write('*Plastic', ...) 里的两点表: (yield, 0), (sigma@15%, eps_pl@15%)."""
    sigma2 = sigma_y0 + Etan * (STRAIN_CAP - sigma_y0 / E)
    eps_pl2 = STRAIN_CAP - sigma2 / E
    return [sigma_y0, sigma2], [0.0, eps_pl2]


def build_simulation():
    # --- 微结构图像: 0=matrix, 1=interphase, 2=particle ---
    img = sio.loadmat(INTPH_IMAGE_MAT)["interphase_image"].astype(int)
    Y, X = img.shape
    assert (X, Y) == (400, 400), f"expected 400x400, got {img.shape}"

    eff_scale = scale
    if COARSEN > 1:
        # 简单最近邻降采样做冒烟测试: RVE 物理尺寸不变(1um), 只是网格更粗、更快solve
        img = img[::COARSEN, ::COARSEN]
        eff_scale = scale * COARSEN
        Y, X = img.shape
        print(f"COARSEN={COARSEN}: using {X}x{Y} mesh (smoke test, not the final "
              f"quantitative comparison)")

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
    nodes = GridNodes.from_matl_img(img, scale=eff_scale)
    elements = GridElements(nodes, type="CPE4")

    matrix_elset, interphase_elset, particle_elset = ElementSet.from_matl_img(img)

    materials = [
        Material(particle_elset, density=Rou_Ptc, poisson=v_Ptc, youngs=E_Ptc),
        PlasticMaterial(matrix_elset, density=Rou_mat, poisson=v0_mat, youngs=E0_mat,
                         yield_stress=mat_ys, plastic_strain=mat_eps),
        PlasticMaterial(interphase_elset, density=Rou_mat, poisson=v0_mat, youngs=E0_int,
                         yield_stress=int_ys, plastic_strain=int_eps),
    ]

    # --- 边界条件: confined 周期性 x 向单轴拉伸 (对应 legacy .inp 里
    #     BOTMLEFT=origin 全约束, BOTMRIGHT=x_drive 压 y, TOPLEFT=y_drive 压 x) ---
    origin, x_drive, y_drive = (nodes.nsets[k] for k in ("X0Y0", "X1Y0", "X0Y1"))
    bcs = [
        PeriodicBoundaryCondition(nodes=nodes),
        FixedBoundaryCondition(origin, dofs=[1, 2]),
        FixedBoundaryCondition(x_drive, dofs=[2]),
        FixedBoundaryCondition(y_drive, dofs=[1]),
        DisplacementBoundaryCondition(x_drive, 1, 1, 0.0),  # baseline
    ]
    model = Model(nodes=nodes, elements=elements, materials=materials, bcs=bcs)

    # --- 加载步: 0.105*scale*X 是 legacy 代码里 BOTMRIGHT 的最终 x 位移
    #     (等于 10.5% 工程应变), 这里拆成 N_STEPS 个递增的 Static step 来重建
    #     完整应力-应变曲线 (每个 step 各自 commit 一次塑性状态，历史正确累加) ---
    Lx = X * eff_scale  # == 400*scale == 1.0 um regardless of COARSEN
    max_disp = 0.105 * Lx
    disp_targets = np.linspace(0.0, max_disp, N_STEPS + 1)[1:]  # 跳过 0

    steps = [
        Step(subsections=[Static(), DisplacementBoundaryCondition(x_drive, 1, 1, float(d))],
             perturbation=False)
        for d in disp_targets
    ]

    sim = Simulation(heading=Heading("VF_2_ParVF_1_Agg_2 bilinear plastic vs ABAQUS"),
                      model=model, steps=steps)
    return sim, Lx


def main():
    t0 = time.time()
    sim, Lx = build_simulation()
    n_steps = len(list(sim.steps))
    print(f"built simulation: {n_steps} Static steps, "
          f"{sim.model.nodes.shape} node grid ({time.time()-t0:.1f}s)")

    # (可选) 先只用 numpy/scipy 生成 .inp, 跟 legacy 生成器的输出做人工比对
    # (材料卡/单元数/节点数/*Equation 数量应该完全一致, 除了单元类型 CPE4 vs CPE4R):
    from microstructure_ve.backends.abaqus import write_inp
    inp_path = f"{BASE}/vf2_parvf1_agg2_check.inp"
    write_inp(sim, inp_path)
    print(f"wrote {inp_path} for a manual sanity diff against the legacy .inp")

    # 用 DOLFINx 后端求解 (需要 fenicsx conda 环境, 见 docs/dolfinx_backend_design.md)
    # 注意: run() 内部没有逐 step 的进度回调, 41 个塑性 Static step 顺序求解,
    # 网格越大单个 step 里的牛顿迭代+LU分解越慢, 这一行可能跑很久, 请耐心等待
    # (可以先用脚本顶部 COARSEN=4 的粗网格冒烟测试确认流程没问题)。
    from microstructure_ve.backends.dolfinx import run
    t1 = time.time()
    rows = run(sim)  # (N_STEPS, 7): [frame, RF_x, RF_y, 0, 0, U_x, U_y]
    print(f"dolfinx solve done in {time.time()-t1:.1f}s")

    RF_x = rows[:, 1]
    U_x = rows[:, 5]
    stress = RF_x / Lx        # cross_area = Ly = Lx = 1.0 micron here, 平面应变单位厚度
    strain = U_x / Lx

    out_path = f"{BASE}/msve_stress_strain{'_coarse' if COARSEN > 1 else ''}.tsv"
    out = np.column_stack([strain, stress])
    np.savetxt(out_path, out, delimiter="\t", header="strain\tstress_MPa", comments="")
    print(f"wrote {out_path}")

    # 跟已有 ABAQUS S-E.txt 比较 (S-E.txt: 前半是应力数组, 后半是应变数组, 等长)
    se = np.loadtxt(ABAQUS_SE_TXT)
    n = len(se) // 2
    abq_stress, abq_strain = se[:n], se[n:]

    # 把 msve 曲线插值到 ABAQUS 的应变点上做逐点比较 (只在两条曲线都覆盖的应变范围内比)
    mask = abq_strain <= strain.max()
    msve_stress_on_abq_strain = np.interp(abq_strain[mask], strain, stress)
    rel_err = np.abs(msve_stress_on_abq_strain - abq_stress[mask]) / np.maximum(abq_stress[mask], 1e-6)
    print(f"compared over strain range up to {abq_strain[mask].max():.4f} "
          f"({mask.sum()}/{len(abq_strain)} ABAQUS points covered by the msve curve)")
    print(f"max relative stress error: {rel_err.max()*100:.2f}%")
    print(f"mean relative stress error: {rel_err.mean()*100:.2f}%")
    print(f"total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
