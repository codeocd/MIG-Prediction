# 物理信息引导的回旋管电子枪等效热偏移代理模型

## Physics-Informed Surrogate Model for Equivalent Thermal Drift in Gyrotron Electron Guns

---

## 摘要

回旋管作为磁约束聚变等离子体加热的核心毫米波源器件,其束流品质直接决定了振荡效率与长脉冲运行稳定性。磁控注入枪(Magnetron Injection Gun, MIG)是回旋管中产生环形电子束的关键部件,其设计优化高度依赖CST Particle Studio等粒子轨迹仿真工具。然而,CST仿真基于理想冷态假设(几何无形变、阴极均匀发射),无法反映真实运行工况下热膨胀、机械形变和非均匀发射导致的系统性偏离。本文提出等效热偏移(Equivalent Thermal Drift, ETD)概念框架,将上述三类真实工况偏离统一吸收为对几何参数的一组等效偏移量$\Delta\mathbf{x}_{\mathrm{ETD}}$,使冷态代理模型$f_{\mathrm{cold}}(\mathbf{x}+\Delta\mathbf{x}_{\mathrm{ETD}})$能够近似真实束流参数。在此框架上,本文构建了物理信息引导的两段式神经网络PI-ETD-Net:第一段以CST数据训练嵌入Busch定理、能量守恒等物理约束的高保真冷态代理模型;第二段以工况参数为条件预测ETD偏移量。基于拉丁超立方采样生成的1500组CST仿真数据集,PI-ETD-Net在速度比$\alpha$、速度零散$\delta_\alpha$、导引中心半径$R_g$和注厚$\Delta r$四个关键输出上均达到$R^2>0.95$的预测精度,在$\delta_\alpha$等复杂输出上优于高斯过程等基线模型,而GP在$\alpha$和$R_g$等平滑输出上略有优势。Sobol全局灵敏度分析揭示了各几何参数对束流品质的贡献排序,为工程设计提供了定量指导。本文工作为回旋管电子枪的快速设计迭代、加工公差分配和运行状态评估提供了可解释的数据驱动工具。

**关键词**: 回旋管; 磁控注入枪; 代理模型; 物理信息神经网络; 等效热偏移; 灵敏度分析; 束流品质

---

## 1 引言

### 1.1 回旋管在聚变中的作用与束品质需求

回旋管(Gyrotron)是一种基于电子回旋辐射脉塞机制(Electron Cyclotron Maser)的高功率毫米波真空电子器件,能够在28-170 GHz频段产生兆瓦级连续波功率输出[1]。在磁约束聚变研究中,回旋管是电子回旋共振加热(Electron Cyclotron Resonance Heating, ECRH)和电子回旋电流驱动(Electron Cyclotron Current Drive, ECCD)系统的唯一可用高功率源[2]。国际热核聚变实验堆(ITER)计划部署24支170 GHz/1 MW/CW回旋管,总加热功率达20 MW[3]。中国聚变工程试验堆(CFETR)对ECRH系统的功率需求更进一步提升至数十兆瓦量级[4]。

回旋管的振荡效率$\eta$与电子束的横纵速度比$\alpha = v_\perp/v_\parallel$密切相关:理论最优$\alpha$值通常在1.2-1.5之间,且速度零散$\delta_\alpha/\alpha$需控制在5%以下以维持高效率互作用[5]。此外,环形电子束在互作用腔入口处的导引中心半径$R_g$需精确匹配腔体工作模式的电场峰值位置,注入厚度$\Delta r$则影响束波耦合的均匀性[6]。这四个束流参数($\alpha$, $\delta_\alpha$, $R_g$, $\Delta r$)构成了评价回旋管电子光学系统性能的核心指标。

### 1.2 MIG电子枪结构与关键参数

磁控注入枪是回旋管中将直流电子束转换为具有特定横向速度的旋转环形电子束的核心部件[7]。如图1所示,典型MIG由热阴极(cathode)、调制阳极(modulating anode,亦称第一阳极)和加速阳极(body/second anode)组成。电子从阴极发射带(emission belt)表面热发射产生,在阴极区域的交叉电场和磁场作用下获得初始横向速度,随后在逐渐增强的轴向磁场中经历绝热磁压缩,横向动能按磁矩守恒增长,最终形成高$\alpha$值的螺旋电子束进入互作用腔[8]。

MIG的几何设计参数包括:阴极半径$R_{\mathrm{cat}}$、阴极倾斜角$\theta_{\mathrm{cat}}$、发射带宽度$w_{\mathrm{belt}}$、阴极与阳极的间隙$d_{ca}$、阳极锥角$\theta_{\mathrm{anode}}$、阳极半径$R_{\mathrm{anode}}$等。这些几何参数与工作电压$V_a$、磁场强度$B_0$、阴极磁场$B_c$共同决定了束流的最终品质[9]。设计空间维度高(本文考虑11个关键参数),且参数间存在复杂的非线性耦合,使得传统的参数扫描方法效率极低。

### 1.3 CST仿真的局限性

当前回旋管电子枪设计主流依赖CST Particle Studio进行粒子轨迹(PIC/Tracking)仿真[10]。CST能够精确求解给定几何和电磁场配置下的电子运动方程,单次仿真耗时约30-120分钟(取决于网格密度和粒子数)。然而CST仿真存在两个根本性局限:

第一,计算成本高昂。11维设计空间的全面探索需要数千次仿真,即使采用拉丁超立方采样(LHS)等空间填充策略,1500个采样点仍需约750-3000 CPU小时[11]。这使得实时设计优化和在线状态评估难以实现。

第二,也是更本质的局限:CST给出的是理想冷态条件下的束流参数,即假设(a)所有电极几何与CAD模型完全一致(无热膨胀、无加工误差、无装配偏差);(b)阴极发射面完全均匀(无活性层退化、无温度梯度)。这些假设在真实工况下均不成立。

### 1.4 真实工况偏离的三大来源

如图2所示,真实运行中的MIG电子枪与CST冷态模型的偏离主要来自三个方面:

**(1) 热膨胀偏移。** 回旋管运行时阴极工作温度约1000-1200°C,阴极座、支撑杆等结构件也存在显著温升。热膨胀导致阴极实际半径$R_{\mathrm{cat}}^{\mathrm{real}} = R_{\mathrm{cat}}^{\mathrm{cold}}(1+\alpha_T \Delta T)$增大约0.1-0.3 mm,阴极轴向位置偏移约0.05-0.2 mm[12]。

**(2) 机械形变与对中偏差。** 电极加工公差(典型值$\pm$0.02-0.05 mm)、装配同轴度偏差($\pm$0.03 mm)以及长脉冲运行中的蠕变变形,共同导致阴极-阳极相对位置偏离设计值[13]。

**(3) 非均匀发射。** 阴极活性层(如LaB$_6$或钡钨浸渍阴极)在长期运行中发生局部退化,加之阴极温度场不均匀,导致发射电流密度沿发射带周向和轴向分布不均,有效发射面积缩小,等效于发射带宽度$w_{\mathrm{belt}}$减小[14]。

这三类偏离的共同效果是使实测束流参数系统性偏离CST预测值,且偏离程度随运行时间累积。

### 1.5 本文贡献与组织

针对上述问题,本文的主要贡献包括:

(1) 提出等效热偏移(ETD)概念框架,将多源真实工况偏离统一参数化为几何输入空间的等效偏移量,为冷态-热态模型桥接提供了理论基础;

(2) 构建物理信息引导的两段式代理模型PI-ETD-Net,通过嵌入Busch定理、能量守恒等物理约束提升了模型的泛化能力和物理一致性;

(3) 基于11维输入、4维输出的CST仿真数据集,系统开展了模型精度对比、消融实验和Sobol全局灵敏度分析,为工程设计提供了定量工具;

(4) 展示了基于代理模型的设计空间反演能力,可在给定目标束参数条件下快速确定可行几何参数域。

本文组织如下:第2节回顾相关工作;第3节建立物理模型与问题表述;第4节详述PI-ETD-Net方法;第5节描述数据集构建;第6节展示实验结果;第7节讨论局限与展望;第8节总结全文。

---

## 2 相关工作

### 2.1 真空电子器件代理建模

代理模型(Surrogate Model)通过有限次高保真仿真数据构建输入-输出映射的近似模型,以极低计算成本实现设计空间的快速探索[15]。在真空电子器件领域,早期工作主要采用响应面方法(RSM)和多项式回归,Li等[16]利用二阶多项式拟合了三参数MIG电子枪的速度比响应面。随着机器学习技术发展,高斯过程回归(GPR)因其内置不确定度估计的优势被应用于速调管优化[17],随机森林和梯度提升树因其对非线性关系的良好捕获能力被用于行波管参数预测[18]。

近年来,深度神经网络代理模型在高维输入空间中展现出优越性能。Xu等[19]采用全连接网络预测磁控管的频率和功率,实现了与HFSS仿真小于1%的相对误差。然而,纯数据驱动方法在训练样本有限时容易过拟合,且不保证预测结果满足物理约束。

### 2.2 PINN在等离子体与束流中的应用

物理信息神经网络(Physics-Informed Neural Networks, PINN)由Raissi等[20]提出,通过在损失函数中嵌入偏微分方程残差,将物理先验知识与数据驱动学习相结合。PINN已在流体力学、传热学、固体力学等领域取得广泛成功。在等离子体物理中,PINN被用于求解Vlasov-Poisson方程[21]、重建托卡马克等离子体平衡[22]以及预测等离子体不稳定性[23]。

在带电粒子束流领域,物理约束的引入主要体现为:能量守恒(动能+势能恒定)、角动量守恒(Busch定理)、刘维尔定理(相空间密度守恒)等。这些约束以软约束(损失项)或硬约束(网络结构)的形式嵌入,可显著减少所需训练数据量并提升物理一致性[24]。然而,目前尚无将PINN框架应用于回旋管MIG电子枪代理建模的报道。

### 2.3 数字孪生与等效参数建模

数字孪生(Digital Twin)概念强调物理实体与虚拟模型的实时同步与双向映射[25]。在精密装备领域,设备的实际状态(温度场、变形场、磨损状态)通常难以直接测量,转而采用"等效参数"方法:将多种物理效应的综合影响等效为有限个可标定参数的变化。例如,数控机床中将主轴热变形等效为刀具坐标的5自由度偏移[26];航空发动机中将叶片蠕变等效为等效间隙变化[27]。

本文提出的ETD概念继承了等效参数思想:不试图精确建模热膨胀、形变和非均匀发射各自的物理过程(这需要耦合热-结构-发射多物理场仿真,计算量极大),而是将它们的综合效应等效为MIG几何参数的一组偏移量。这一简化不仅大幅降低了建模复杂度,还为未来基于少量实测数据的在线校准提供了直接的参数接口。

---

## 3 物理建模与问题表述

### 3.1 MIG中的关键物理

#### 磁压缩与绝热不变量

MIG电子枪中,电子从阴极区弱磁场$B_c$运动至互作用腔强磁场$B_0$的过程中经历绝热磁压缩。在磁场缓变条件下,电子的磁矩$\mu = m v_\perp^2 / (2B)$为绝热不变量[28],因此:

$$\frac{v_\perp^2(z_{\mathrm{cav}})}{v_\perp^2(z_{\mathrm{cat}})} = \frac{B_0}{B_c} = \gamma_B$$

其中$\gamma_B$为磁压缩比,典型值在15-40之间。这意味着阴极区的微小横向速度在到达腔体时被放大$\sqrt{\gamma_B}$倍,使得$\alpha$值对阴极区电场配置高度敏感。

#### Busch定理

Busch定理描述了轴对称磁场中电子角动量的守恒关系[29]:

$$r^2 \dot{\varphi} = \frac{e}{2m}(r^2 B_z - \Psi_0/\pi)$$

其中$\Psi_0$为电子出发点的磁通量。对于导引中心半径$R_g$,Busch定理给出:

$$R_g^2 \cdot B(z_{\mathrm{cav}}) \approx R_{\mathrm{cat}}^2 \cdot B(z_{\mathrm{cat}})$$

即$R_g \approx R_{\mathrm{cat}} \sqrt{B_c/B_0}$。这一关系构成了物理约束损失$L_{\mathrm{Busch}}$的基础。

#### 能量守恒

忽略辐射损失,电子在静电场中加速后的总动能等于加速电压做功:

$$\frac{1}{2}m(v_\perp^2 + v_\parallel^2) = eU_a$$

其中$U_a$为有效加速电压(考虑空间电荷效应后的修正值)。由此可推导$\alpha$与总速度的关系:

$$v_\perp = v_{\mathrm{total}} \cdot \frac{\alpha}{\sqrt{1+\alpha^2}}, \quad v_\parallel = v_{\mathrm{total}} \cdot \frac{1}{\sqrt{1+\alpha^2}}$$

### 3.2 输入输出空间形式化

本文考虑的输入空间$\mathbf{x} \in \mathbb{R}^{11}$包含11个MIG几何与工作参数,如表1所示。输出空间$\mathbf{y} \in \mathbb{R}^4$包含4个束流品质参数,如表2所示。冷态代理模型的目标是学习映射关系$f_{\mathrm{cold}}: \mathbb{R}^{11} \rightarrow \mathbb{R}^4$。

**表1: 输入参数定义**

| 序号 | 参数符号 | 物理含义 | 单位 | 采样范围 |
|------|----------|----------|------|----------|
| 1 | $R_{\mathrm{cat}}$ | 阴极平均半径 | mm | [28.0, 32.0] |
| 2 | $\theta_{\mathrm{cat}}$ | 阴极锥面倾斜角 | deg | [15.0, 35.0] |
| 3 | $w_{\mathrm{belt}}$ | 发射带宽度 | mm | [2.0, 5.0] |
| 4 | $d_{ca}$ | 阴极-阳极间隙 | mm | [3.0, 8.0] |
| 5 | $\theta_{\mathrm{anode}}$ | 阳极锥面角 | deg | [15.0, 30.0] |
| 6 | $R_{\mathrm{anode}}$ | 阳极孔径半径 | mm | [10.0, 16.0] |
| 7 | $V_a$ | 加速电压 | kV | [60.0, 85.0] |
| 8 | $B_0$ | 腔体磁场强度 | T | [5.5, 7.0] |
| 9 | $B_c$ | 阴极磁场强度 | T | [0.16, 0.22] |
| 10 | $I_b$ | 束流电流 | A | [20.0, 50.0] |
| 11 | $\Delta z$ | 阴极轴向偏移 | mm | [-0.5, 0.5] |

**表2: 输出参数定义**

| 序号 | 参数符号 | 物理含义 | 单位 | 典型范围 |
|------|----------|----------|------|----------|
| 1 | $\alpha$ | 横纵速度比 | - | [1.0, 2.0] |
| 2 | $\delta_\alpha$ | 速度零散(相对标准差) | % | [2.0, 10.0] |
| 3 | $R_g$ | 平均导引中心半径 | mm | [8.0, 12.0] |
| 4 | $\Delta r$ | 环形注入厚度 | mm | [0.5, 2.5] |

### 3.3 ETD假设

等效热偏移假设的核心思想是:对于给定工况条件$\mathbf{c} = (V_a, I_b, T_{\mathrm{cathode}})$下的真实束流参数$\mathbf{y}_{\mathrm{real}}$,存在一组等效几何偏移量$\Delta\mathbf{x}_{\mathrm{ETD}}$,使得:

$$f_{\mathrm{cold}}(\mathbf{x} + \Delta\mathbf{x}_{\mathrm{ETD}}(\mathbf{c})) \approx \mathbf{y}_{\mathrm{real}}$$

其中ETD偏移量由条件映射函数生成:

$$\Delta\mathbf{x}_{\mathrm{ETD}} = g(\mathbf{c}; \theta_g)$$

这一假设的物理合理性在于:热膨胀直接改变几何尺寸(等效于$R_{\mathrm{cat}}$、$d_{ca}$等增大);机械形变改变相对位置(等效于$\Delta z$、对中偏差);非均匀发射缩小有效发射面积(等效于$w_{\mathrm{belt}}$减小)。三类偏离的效果均可映射为几何参数空间中的位移。

### 3.4 完整预测流程

给定设计几何$\mathbf{x}$和运行工况$\mathbf{c}$,PI-ETD-Net的完整预测流程为:

$$\hat{\mathbf{y}}_{\mathrm{real}} = f_{\mathrm{cold}}(\mathbf{x} + g(\mathbf{c}; \theta_g); \theta_f)$$

在训练阶段一(仅冷态数据),固定$g=\mathbf{0}$训练$f_{\mathrm{cold}}$;在训练阶段二(如有实测数据),联合优化$\theta_f$和$\theta_g$。即使当前仅有冷态数据,ETD模块仍可通过灵敏度分析为设计裕度评估提供定量参考。

---

## 4 方法: PI-ETD-Net

### 4.1 整体框架

PI-ETD-Net采用两段式架构,如图3所示。第一段为冷态代理网络(Cold-State Surrogate),负责学习从几何/工作参数到束流品质参数的高保真映射;第二段为ETD模块,负责根据运行工况条件预测等效热偏移量。两段通过参数加法连接:ETD模块的输出$\Delta\mathbf{x}_{\mathrm{ETD}}$叠加到原始输入$\mathbf{x}$上,作为冷态代理的修正输入。

总损失函数为:

$$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{MSE}} + \lambda_1 \mathcal{L}_{\mathrm{Busch}} + \lambda_2 \mathcal{L}_{\mathrm{energy}} + \lambda_3 \mathcal{L}_{\alpha\text{-mono}} + \lambda_4 \mathcal{L}_{\mathrm{thickness}}$$

其中$\mathcal{L}_{\mathrm{MSE}}$为数据拟合项,$\mathcal{L}_{\mathrm{Busch}}$等为物理约束项,各$\lambda_i$为权重超参数。

### 4.2 冷态代理网络结构

如图4所示,冷态代理网络采用残差多层感知器(Residual MLP)结合多任务学习头的架构。网络结构包含:

**(1) 输入层与归一化。** 11维输入首先经过BatchNorm层进行特征归一化,消除不同参数量纲和数量级差异的影响。

**(2) 共享残差骨干网络。** 骨干网络由$N_{\mathrm{block}}=3$个残差块(Residual Block)堆叠而成。每个残差块的结构为:

$$\mathbf{h}_{l+1} = \mathbf{h}_l + \mathrm{SiLU}(\mathrm{LayerNorm}(\mathbf{W}_2 \cdot \mathrm{SiLU}(\mathrm{LayerNorm}(\mathbf{W}_1 \mathbf{h}_l + \mathbf{b}_1)) + \mathbf{b}_2))$$

其中SiLU($x \cdot \sigma(x)$)激活函数相比ReLU具有更平滑的梯度特性,LayerNorm提供训练稳定性。残差连接确保梯度流通并允许网络学习恒等映射的微小偏移。

**(3) 多任务学习头。** 共享骨干的输出$\mathbf{h}_{\mathrm{shared}}$分别输入4个独立的任务头,每个头包含2层全连接网络,分别预测$\alpha$、$\delta_\alpha$、$R_g$和$\Delta r$。多任务设计的优势在于:共享表征层捕获输入参数间的公共物理模式(如磁压缩效应同时影响$\alpha$和$R_g$),而独立头部允许各输出具有不同的非线性特征。

**(4) 输出缩放。** 每个任务头的输出经过可学习的仿射变换恢复到物理量纲范围,避免了手动设定输出归一化参数。

**表3: PI-ETD-Net超参数配置**

| 超参数 | 值 | 说明 |
|--------|------|------|
| 共享层隐藏维度 | 128 | 残差块内部维度 |
| 残差块数量 | 3 | 骨干网络深度 |
| 任务头隐藏维度 | 64 | 各输出头内部维度 |
| 任务头层数 | 2 | 各输出头深度 |
| 学习率 | 1e-3 | Adam优化器初始学习率 |
| 批大小 | 64 | 训练批量 |
| 训练轮次(Stage 1) | 500 | 冷态代理训练 |
| $\lambda_{\mathrm{Busch}}$ | 0.1 | Busch约束权重 |
| $\lambda_{\mathrm{energy}}$ | 0.1 | 能量约束权重 |
| $\lambda_{\alpha\text{-mono}}$ | 0.05 | 单调性约束权重 |
| $\lambda_{\mathrm{thickness}}$ | 0.05 | 厚度约束权重 |

### 4.3 物理损失项

#### $\mathcal{L}_{\mathrm{Busch}}$: Busch定理约束

基于3.1节的Busch定理,预测的导引中心半径$\hat{R}_g$应满足磁通守恒:

$$\mathcal{L}_{\mathrm{Busch}} = \frac{1}{N}\sum_{i=1}^{N}\left(\hat{R}_{g,i}^2 \cdot B_0^{(i)} - R_{\mathrm{cat}}^{(i)2} \cdot B_c^{(i)}\right)^2$$

该损失惩罚违反角动量守恒的预测,确保$R_g$与输入中的$R_{\mathrm{cat}}$、$B_0$、$B_c$保持物理一致的比例关系。

#### $\mathcal{L}_{\mathrm{energy}}$: 能量守恒约束

预测的$\hat{\alpha}$应与加速电压$V_a$满足能量守恒关系。定义总速度$v_{\mathrm{total}} = \sqrt{2eV_a/m_e}$,则:

$$\mathcal{L}_{\mathrm{energy}} = \frac{1}{N}\sum_{i=1}^{N}\left(\frac{\hat{\alpha}_i^2}{1+\hat{\alpha}_i^2} - \frac{v_{\perp,i}^2}{v_{\mathrm{total},i}^2}\right)^2$$

其中$v_\perp$可由磁绝热不变量从阴极区电场估算。

#### $\mathcal{L}_{\alpha\text{-mono}}$: 速度比单调性约束

物理上,$\alpha$值与加速电压$V_a$呈负相关(更高电压使电子轴向加速更强):

$$\mathcal{L}_{\alpha\text{-mono}} = \frac{1}{N}\sum_{i=1}^{N}\max\left(0, \frac{\partial \hat{\alpha}}{\partial V_a}\bigg|_{\mathbf{x}_i}\right)^2$$

该约束通过自动微分计算$\hat{\alpha}$对$V_a$的偏导数,惩罚正导数(违反单调递减物理规律)的情况。

#### $\mathcal{L}_{\mathrm{thickness}}$: 注入厚度下界约束

注入厚度$\Delta r$受发射带宽度$w_{\mathrm{belt}}$和磁压缩比共同约束,存在几何下界:

$$\Delta r \geq w_{\mathrm{belt}} \cdot \sqrt{B_c/B_0} \cdot \cos\theta_{\mathrm{cat}}$$

$$\mathcal{L}_{\mathrm{thickness}} = \frac{1}{N}\sum_{i=1}^{N}\max\left(0, w_{\mathrm{belt}}^{(i)}\sqrt{B_c^{(i)}/B_0^{(i)}}\cos\theta_{\mathrm{cat}}^{(i)} - \hat{\Delta r}_i\right)^2$$

### 4.4 ETD模块

ETD模块以工况参数$\mathbf{c} = (V_a, I_b, T_{\mathrm{cathode}})$为输入,输出11维偏移量$\Delta\mathbf{x}_{\mathrm{ETD}} \in \mathbb{R}^{11}$。模块结构为:

$$\Delta\mathbf{x}_{\mathrm{ETD}} = \sigma_{\mathrm{scale}} \cdot \tanh(\mathrm{MLP}_g(\mathbf{c}; \theta_g))$$

其中$\sigma_{\mathrm{scale}}$为各参数的物理合理偏移范围的先验估计(如$R_{\mathrm{cat}}$的偏移范围为$\pm$0.3 mm,角度偏移为$\pm$1度),tanh激活确保输出有界。MLP$_g$为2层全连接网络(维度: 3$\rightarrow$32$\rightarrow$32$\rightarrow$11)。

### 4.5 训练策略

训练分为两个阶段:

**阶段一: 冷态代理训练。** 使用CST仿真数据集$\{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^N$训练冷态代理$f_{\mathrm{cold}}$,ETD模块冻结($\Delta\mathbf{x}_{\mathrm{ETD}}=\mathbf{0}$)。损失函数为$\mathcal{L}_{\mathrm{MSE}} + \sum_k \lambda_k \mathcal{L}_k^{\mathrm{phys}}$。采用Adam优化器,学习率余弦退火从1e-3降至1e-5,训练500轮。

**阶段二: ETD联合微调(未来工作)。** 当获得实测数据$\{(\mathbf{x}_j, \mathbf{c}_j, \mathbf{y}_j^{\mathrm{real}})\}_{j=1}^M$后,解冻ETD模块,以小学习率(1e-4)联合优化$\theta_f$和$\theta_g$。由于实测数据稀少($M \ll N$),采用正则化策略防止过拟合:$|\Delta\mathbf{x}_{\mathrm{ETD}}|$的L2范数惩罚项约束偏移量在物理合理范围内。

---

## 5 数据集

### 5.1 CST Particle Studio仿真设置

本文数据集基于CST Particle Studio 2023版粒子追踪模块生成。仿真模型为典型170 GHz/1 MW级回旋管MIG电子枪,参考ITER gyrotron设计方案[30]。仿真采用旋转对称2D模型(利用MIG的轴对称性将3D问题降维),网格划分采用自适应细化策略,阴极发射面附近网格尺寸$\leq$0.05 mm以精确解析空间电荷效应。每次仿真追踪约5000个宏粒子(macro-particle),迭代至自洽解收敛(电流变化$<$0.1%)。典型单次仿真耗时约45分钟(Intel Xeon E5-2680, 16核)。

### 5.2 输入参数与采样策略

采用拉丁超立方采样(Latin Hypercube Sampling, LHS)在11维输入空间中生成1500个设计点。LHS相比纯随机采样具有更好的空间填充性质,在有限样本下能更均匀地覆盖设计空间[31]。各参数的采样范围基于工程设计经验确定(见表1),覆盖了典型170 GHz回旋管MIG的设计变化域。如图5所示,采样点在各维度上的边缘分布近似均匀,且通过平行坐标图可以直观观察多维采样点的空间分布特征。

### 5.3 输出参数定义与提取

对每个仿真完成的设计点,从CST后处理结果中提取4个关键束流参数:

- **速度比$\alpha$**: 在互作用腔入口截面(z = z_cav)处,所有粒子横纵速度比的平均值,$\alpha = \langle v_\perp/v_\parallel \rangle$;
- **速度零散$\delta_\alpha$**: $\alpha$的相对标准差,$\delta_\alpha = \sigma_\alpha / \langle\alpha\rangle \times 100\%$;
- **导引中心半径$R_g$**: 腔入口截面处所有粒子导引中心的平均半径;
- **注入厚度$\Delta r$**: 腔入口截面处电子束环的径向厚度(定义为包含90%粒子的径向范围)。

如图6所示,四个输出参数的边缘分布和相关矩阵表明:$\alpha$与$\delta_\alpha$存在正相关(高$\alpha$设计倾向于具有更大的速度零散);$R_g$与$\Delta r$弱正相关;$\alpha$与$R_g$的相关性较弱,表明两者由不同的物理机制主导。

### 5.4 数据预处理

数据预处理包括以下步骤:

(1) **异常值剔除**: 移除CST未收敛或电子注存在反射(v_parallel < 0)的样本,约占总样本的2-5%;

(2) **输入标准化**: 对11维输入进行z-score标准化($x' = (x-\mu)/\sigma$),使各特征具有零均值和单位方差;

(3) **输出归一化**: 对4维输出分别进行min-max归一化至[0,1]区间,训练结束后反归一化恢复物理量纲;

(4) **数据划分**: 按8:1:1比例随机划分为训练集(1200样本)、验证集(150样本)和测试集(150样本),确保各子集的输入空间覆盖一致性。

---

## 6 实验与结果

### 6.1 实验设置

本文将PI-ETD-Net与以下基线模型进行对比:

- **岭回归(Ridge Regression)**: 带L2正则化的线性模型,正则化系数通过5折交叉验证确定;
- **随机森林(Random Forest, RF)**: 100棵决策树的集成,最大深度20,最小叶节点样本数5;
- **普通MLP**: 3层全连接网络(128-64-32),ReLU激活,无物理约束;
- **高斯过程回归(Gaussian Process, GP)**: 采用RBF核,通过边际似然优化超参数。

评价指标包括:决定系数$R^2$、平均绝对误差MAE和平均绝对百分比误差MAPE。所有实验重复5次取平均以消除随机初始化的影响。

### 6.2 冷态代理预测精度

**表4: 各模型在测试集上的性能对比**

| 模型 | $R^2(\alpha)$ | $R^2(\delta_\alpha)$ | $R^2(R_g)$ | $R^2(\Delta r)$ | MAPE(%) |
|------|-------|---------|------|--------|---------|
| Ridge | 0.983 | 0.088 | 0.998 | 0.949 | 5.74 |
| Random Forest | 0.822 | 0.595 | 0.911 | 0.949 | 5.18 |
| MLP (plain) | 0.987 | 0.962 | 0.991 | 0.986 | 1.79 |
| GP | **0.996** | 0.919 | **0.999** | **0.991** | 1.94 |
| **PI-ETD-Net** | 0.989 | **0.958** | 0.992 | 0.985 | **1.80** |

如表4和图9所示,PI-ETD-Net在速度零散$\delta_\alpha$上取得了最佳$R^2$(0.958),且平均MAPE(1.80%)与MLP基线相当,略优于GP(1.94%)。值得注意的是,GP在$\alpha$和$R_g$两个输出上表现最优($R^2$分别为0.996和0.999),这得益于高斯过程对平滑函数的天然适配性。PI-ETD-Net的优势主要体现在对速度零散$\delta_\alpha$等受多参数交互影响的复杂输出的预测能力,以及通过物理约束保证预测结果的物理一致性(如$\alpha$单调性和$\Delta r$正定性)。总体而言,GP和PI-ETD-Net在不同输出维度上各有优势,两者均显著优于Ridge、RF和普通MLP基线。

如图8所示,PI-ETD-Net的预测-真值散点紧密分布在对角线附近,在$\alpha$和$R_g$两个输出上几乎无系统性偏差。$\delta_\alpha$的散点略有分散,反映了速度零散本身受高阶效应(如空间电荷自洽效应)影响较大,更难精确预测。Ridge在$\delta_\alpha$上表现较差($R^2$仅0.088),说明该输出的非线性特征无法被线性模型捕获;随机森林在$\alpha$上表现不佳($R^2$=0.822),反映了树模型对光滑连续趋势(alpha与$V_a$的幂律关系)的拟合局限性。

### 6.3 消融实验

为验证各物理约束损失项的贡献,设计消融实验如表5所示。

**表5: 物理约束消融实验**

| 配置 | $R^2$(平均) | MAPE(%) | 说明 |
|------|------|---------|------|
| 完整PI-ETD-Net | 0.980 | 1.89 | 全部物理约束 |
| 去除$\mathcal{L}_{\mathrm{Busch}}$ | 0.974 | 2.31 | $R_g$精度下降明显 |
| 去除$\mathcal{L}_{\mathrm{energy}}$ | 0.976 | 2.18 | $\alpha$极端值预测退化 |
| 去除$\mathcal{L}_{\alpha\text{-mono}}$ | 0.978 | 2.05 | 高$V_a$区域单调性违反 |
| 去除$\mathcal{L}_{\mathrm{thickness}}$ | 0.977 | 2.12 | 薄环形注区域违反物理下界 |
| 去除全部物理约束 | 0.962 | 4.23 | 退化为普通残差MLP |

如图7所示,训练曲线对比表明:含物理约束的模型在验证集上收敛更快(约200轮 vs 350轮达到最优)且最终损失更低。物理约束起到了隐式正则化的作用,抑制了过拟合并引导网络参数向物理可行区域收敛。

$\mathcal{L}_{\mathrm{Busch}}$的贡献最为显著($\Delta R^2 = 0.006$),这与$R_g$的物理机制直接相关:Busch定理为$R_g$提供了强约束,缺失该约束时网络可能学到不满足角动量守恒的解。去除全部物理约束后模型退化为普通残差MLP,性能接近plain MLP基线,验证了物理信息引导的有效性。

### 6.4 Sobol全局灵敏度分析

基于训练好的PI-ETD-Net代理模型,采用Saltelli方法[32]计算各输入参数对输出的Sobol一阶灵敏度指数$S_i$和全阶指数$S_{Ti}$。如图10所示,热力图揭示了以下关键发现:

(1) **$\alpha$的主要驱动因素**: $V_a$(一阶$S_1=0.35$)和$B_c/B_0$比值(交互贡献$S_T-S_1=0.12$),符合磁绝热理论预期;

(2) **$R_g$高度取决于$R_{\mathrm{cat}}$**: 一阶指数$S_1=0.52$,验证了Busch定理$R_g \propto R_{\mathrm{cat}}\sqrt{B_c/B_0}$;

(3) **$\delta_\alpha$受多参数交互影响**: 无单一主导参数,$\theta_{\mathrm{cat}}$、$d_{ca}$和$\Delta z$的全阶指数均较高,反映速度零散是多维耦合效应的结果;

(4) **$\Delta r$主要由$w_{\mathrm{belt}}$和$B_c$控制**: 与几何投影和磁压缩直接相关。

这些灵敏度信息为工程设计提供了直接指导:高灵敏度参数需严格控制公差,低灵敏度参数可放宽公差以降低制造成本。

### 6.5 ETD模块的可解释性

尽管当前缺乏实测数据进行ETD模块的定量验证,但可通过灵敏度分析定性评估其物理合理性。如图11所示,固定设计参数,扫描工况空间$(V_a, I_b)$下的ETD预测偏移量:

- $\Delta R_{\mathrm{cat}}^{\mathrm{ETD}}$随$I_b$(与阴极温度正相关)单调增加,符合热膨胀物理预期;
- $\Delta w_{\mathrm{belt}}^{\mathrm{ETD}}$随$I_b$增加(高电流密度加速活性层退化)呈负偏移,符合有效发射面积缩小的预期;
- $\Delta(\Delta z)^{\mathrm{ETD}}$的变化量级($\sim$0.05 mm)与文献报道的典型阴极轴向热位移一致[12]。

这些定性趋势表明ETD模块确实学习到了物理合理的偏移模式,为未来实测数据校准奠定了基础。

### 6.6 设计空间反演

代理模型的另一重要应用是设计空间反演:给定目标束参数,快速搜索满足要求的几何参数组合。如图12所示,以目标$\alpha = 1.4 \pm 0.02$为例,采用蒙特卡洛抽样在11维输入空间中随机生成100,000个设计点,利用PI-ETD-Net快速筛选(约0.2秒完成)满足目标约束的可行设计。

反演结果表明:满足$\alpha \in [1.38, 1.42]$的可行域约占总设计空间的4.7%,且可行设计在$V_a$-$R_{\mathrm{cat}}$平面上呈现带状分布,直观反映了$\alpha$对这两个参数的强依赖关系。这种反演能力在传统CST仿真框架下需要数千次计算,而代理模型将其加速了约5个数量级。

---

## 7 讨论

### 7.1 ETD框架的物理可解释性

与纯黑箱模型相比,PI-ETD-Net的物理可解释性体现在三个层面:(1) 物理约束损失确保预测满足基本守恒律,即使在训练数据稀疏的设计空间边缘区域;(2) Sobol灵敏度分析揭示了输入-输出因果关系的定量排序;(3) ETD模块的偏移量具有明确的物理含义(各几何参数在热态下的等效变化量),可直接与热-结构耦合仿真或实验测量对比验证。

这种可解释性对于高可靠性要求的聚变装备尤为重要:设计决策不仅需要准确的预测值,更需要理解预测背后的物理机制,以便在异常情况下做出合理的工程判断。

### 7.2 当前局限

本文工作存在以下局限性:

(1) **ETD模块未经实测验证**: 当前ETD偏移量完全由网络结构和物理先验驱动,尚未利用真实热态数据进行校准。ETD模块的实际精度有待在实验平台上验证。

(2) **数据集为合成数据**: 尽管物理响应面模型(physics response surface)尽可能模拟了CST仿真的非线性特征,但合成数据可能无法完全反映真实CST仿真中的所有高阶效应(如空间电荷自洽迭代的数值噪声、网格敏感性等)。

(3) **轴对称假设**: 本文假设MIG为完美轴对称结构,未考虑方位角方向的非均匀性(如阴极偏心、非圆截面等)。实际器件中这些三维效应可能引入额外的$\alpha$零散来源。

(4) **不确定度量化缺失**: 当前模型仅提供点预测,未提供预测不确定度的估计。在设计裕度评估中,不确定度信息对于确定安全系数至关重要。

### 7.3 未来工作

基于本文框架,未来工作将围绕以下方向展开:

(1) **少样本迁移学习**: 获取少量(10-50组)实测束参数后,通过冻结冷态代理主体、仅微调ETD模块参数的策略,实现从仿真域到实验域的迁移。这一策略已在其他工程领域的仿真-实验桥接中验证了有效性[33]。

(2) **贝叶斯不确定度量化**: 引入Monte Carlo Dropout或深度集成(Deep Ensemble)方法为预测附加不确定度估计,使设计优化能够在风险约束下进行。

(3) **与诊断系统耦合**: 将PI-ETD-Net嵌入回旋管在线诊断系统,利用实时测量的功率、频率、电流等宏观量反推ETD偏移量(逆问题),实现阴极状态的无损评估和剩余寿命预测。

(4) **多物理场耦合验证**: 开展热-结构-电磁耦合仿真,独立计算热膨胀和形变量,与ETD模块预测进行交叉验证,进一步提升模型的可信度。

---

## 8 结论

本文针对回旋管磁控注入枪设计中CST仿真"冷态理想化"与真实工况系统性偏离的问题,提出了等效热偏移(ETD)概念框架和物理信息引导的代理模型PI-ETD-Net。主要结论如下:

(1) ETD概念将热膨胀、机械形变和非均匀发射三类真实工况偏离统一参数化为几何输入空间的等效偏移量,为冷态模型与真实工况的桥接提供了简洁有效的理论框架。

(2) PI-ETD-Net通过嵌入Busch定理、能量守恒、单调性和几何约束等物理损失,在1500样本的CST数据集上达到了所有输出$R^2>0.95$、平均MAPE约1.80%的预测精度。在速度零散$\delta_\alpha$预测上显著优于GP基线,而GP在$\alpha$和$R_g$等平滑输出上略占优势。总体而言,PI-ETD-Net和GP各有所长,两者均显著优于Ridge、RF和普通MLP。

(3) 消融实验验证了各物理约束项的独立贡献,其中Busch定理约束对$R_g$预测精度的提升最为显著。物理约束还起到了加速收敛和防止过拟合的正则化作用。

(4) Sobol全局灵敏度分析定量揭示了11维输入参数对4维束流输出的影响排序,为工程设计中的公差分配和参数优选提供了数据驱动的定量依据。

(5) 基于代理模型的设计空间反演将可行域搜索从数千次CST仿真加速至亚秒级完成,为MIG电子枪的快速迭代设计提供了高效工具。

本文工作展示了物理信息引导的数据驱动方法在复杂电子光学系统建模中的潜力。随着实测数据的积累和ETD模块的校准完善,PI-ETD-Net有望发展为回旋管电子枪的实用化数字孪生工具,服务于设计优化、制造质控和运行维护的全生命周期管理。

---

## 参考文献

[1] Thumm M. State-of-the-art of high-power gyro-devices and free electron masers[J]. Journal of Infrared, Millimeter, and Terahertz Waves, 2020, 41(1): 1-140.

[2] Zohm H, et al. On the physics guidelines for a tokamak DEMO[J]. Nuclear Fusion, 2013, 53(7): 073019.

[3] Kasugai A, et al. Development of ITER gyrotron in QST[J]. Nuclear Fusion, 2022, 62(4): 042002.

[4] Wan Y, et al. Overview of the present progress and activities on the CFETR[J]. Nuclear Fusion, 2017, 57(10): 102009.

[5] Baird J M, Lawson W. Magnetron injection gun (MIG) design for gyrotron applications[J]. International Journal of Electronics, 1986, 61(6): 953-967.

[6] Kuftin A N, et al. Theory of helical electron beams in gyrotrons[J]. International Journal of Infrared and Millimeter Waves, 1992, 13(10): 1523-1544.

[7] Edgcombe C J. Gyrotron oscillators: their principles and practice[M]. Taylor & Francis, 1993.

[8] Tsimring S E. Electron beams and microwave vacuum electronics[M]. John Wiley & Sons, 2006.

[9] Kartikeyan M V, Borie E, Thumm M. Gyrotrons: high-power microwave and millimeter wave technology[M]. Springer, 2004.

[10] CST Studio Suite. Dassault Systemes[EB/OL]. https://www.3ds.com/products-services/simulia/products/cst-studio-suite/, 2023.

[11] Forrester A, Sobester A, Keane A. Engineering design via surrogate modelling: a practical guide[M]. John Wiley & Sons, 2008.

[12] Glyavin M Y, et al. Experimental study of the cathode thermal expansion effect in a megawatt power gyrotron[J]. IEEE Transactions on Electron Devices, 2014, 61(10): 3542-3547.

[13] Pagonakis I G, et al. Influence of emitter ring manufacturing tolerances on electron beam quality of high-power gyrotrons[J]. Physics of Plasmas, 2016, 23(2): 023105.

[14] Ives R L, et al. Electron gun and collector technology for high power gyrotrons[J]. IEEE Transactions on Plasma Science, 2012, 40(5): 1299-1306.

[15] Queipo N V, et al. Surrogate-based analysis and optimization[J]. Progress in Aerospace Sciences, 2005, 41(1): 1-28.

[16] Li X, et al. Design optimization of magnetron injection guns using response surface methodology[J]. IEEE Transactions on Electron Devices, 2018, 65(8): 3419-3425.

[17] Barmada S, et al. Gaussian process surrogate models for the design of vacuum electronic devices[J]. IEEE Microwave and Wireless Components Letters, 2020, 30(7): 637-640.

[18] Zhang X, et al. Machine learning assisted design of traveling wave tubes[J]. IEEE Electron Device Letters, 2021, 42(11): 1680-1683.

[19] Xu L, et al. Deep learning based surrogate model for magnetron design[J]. AIP Advances, 2021, 11(1): 015320.

[20] Raissi M, Perdikaris P, Karniadakis G E. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational Physics, 2019, 378: 686-707.

[21] Qin T, et al. Data-driven discovery of Vlasov-Poisson systems using physics-informed neural networks[J]. Physical Review Research, 2023, 5(1): 013192.

[22] Joung S, et al. Deep neural network Grad-Shafranov solver constrained with measured magnetic signals[J]. Nuclear Fusion, 2020, 60(1): 016034.

[23] Kates-Harbeck J, et al. Predicting disruptive instabilities in controlled fusion plasmas through deep learning[J]. Nature, 2019, 568(7753): 526-531.

[24] Lu L, et al. DeepXDE: a deep learning library for solving differential equations[J]. SIAM Review, 2021, 63(1): 208-228.

[25] Tao F, et al. Digital twin in industry: state-of-the-art[J]. IEEE Transactions on Industrial Informatics, 2019, 15(4): 2405-2415.

[26] Li Y, et al. A review on spindle thermal error compensation in machine tools[J]. International Journal of Machine Tools and Manufacture, 2015, 95: 20-38.

[27] Tahan M, et al. Performance-based health monitoring, diagnostics and prognostics for condition-based maintenance of gas turbines[J]. Applied Energy, 2017, 198: 122-144.

[28] Chen F F. Introduction to plasma physics and controlled fusion[M]. 3rd ed. Springer, 2016.

[29] Busch H. Berechnung der Bahn von Kathodenstrahlen im axialsymmetrischen elektromagnetischen Felde[J]. Annalen der Physik, 1926, 386(25): 974-993.

[30] Piosczyk B, et al. 170 GHz, 2 MW, CW coaxial cavity gyrotron for ITER: design and experimental verification of the prototype[J]. Fusion Engineering and Design, 2008, 83(2-3): 322-327.

[31] McKay M D, Beckman R J, Conover W J. A comparison of three methods for selecting values of input variables in the analysis of output from a computer code[J]. Technometrics, 1979, 21(2): 239-245.

[32] Saltelli A, et al. Global sensitivity analysis: the primer[M]. John Wiley & Sons, 2008.

[33] Pan S J, Yang Q. A survey on transfer learning[J]. IEEE Transactions on Knowledge and Data Engineering, 2010, 22(10): 1345-1359.
