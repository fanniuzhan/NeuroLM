# NeuroLM 入门数据下载

日期：2026-11-18。目标：用小规模真实 EEG 理解数据输入、tokenizer、预训练和指令微调的衔接。

## 选择的数据

| 本地目录（项目根目录下） | 范围 | 论文中的用途 | 入门用途 |
| --- | --- | --- | --- |
| `data/raw/eegmat-1.0.0/` | 完整 36 人、72 条 EDF；保留说明与受试者信息 | 下游 Workload，二分类 | 首先跑通一个完整下游任务 |
| `data/raw/hmc-1.1/` | SN001、SN103、SN128；每人信号 EDF、评分 EDF、评分 TXT | 下游 HMC，五分类 | 再增加睡眠分期，理解多任务 |
| `data/raw/eegmmidb-1.0.0/` | S001–S003，每人 R01、R02、R04，含事件文件 | 附录 C 的 Motor Movement/Imagery 预训练来源 | 小规模 tokenizer / 自回归预训练输入 |

三者均可从 PhysioNet 官方地址免登录下载。Workload 与 EEGMMIDB 使用 Open Data Commons Attribution License v1.0；HMC 1.1 使用 CC BY 4.0。请按源站要求引用原数据。

本次下载已完成：Workload 75 个文件 / 183.61 MB，HMC 13 个文件 / 340.25 MB，EEGMMIDB 24 个文件 / 15.57 MB，总计 **112 个文件、539,436,027 字节（539.44 MB）**，另有官方校验清单及本地日志。全部 112 个文件通过官方 SHA256 比对，87 个 EDF 文件通过文件头与声明长度检查；详见 `data/verification.log` 和 `data/download_manifest.json`。这里只验证了文件完整性与头部信息，没有验证波形质量或训练效果。

实测：Workload 72 条信号记录共约 2.41 小时，静息记录 80–188 秒、心算记录 62 秒；HMC 三晚约 23.05 小时，三人的评分各自均覆盖五个阶段；EEGMMIDB 九条记录约 12.32 分钟。EDF 文件头中的信号总数可能包含辅助导联或标注：Workload 的 22 路包含 19 EEG、参考、ECG 和标注；HMC 的 8 路中有 4 EEG；EEGMMIDB 的 65 路包含 64 EEG 与标注。

- [Workload / EEGMAT 1.0.0](https://physionet.org/content/eegmat/1.0.0/)
- [HMC 1.1](https://physionet.org/content/hmc-sleep-staging/1.1/)
- [EEGMMIDB 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/)
- [论文：下游任务及附录 C](https://arxiv.org/html/2409.00101v3)

HMC 全集约 15.7 GB，本次只取三人。按官方全部 151 条信号记录排序，SN001、SN103、SN128 的零起始位置分别是 0、100、125，对应论文训练、验证、测试范围。编号有缺号，不能直接把 SN101 当成第 101 人。**只对这三人调用原脚本的 `[:100]` 切分，会全部落入训练集；处理子集时应显式指定三组。**

EEGMMIDB 的 R01/R02 是睁眼/闭眼基线，R04 是左右拳运动想象。它是论文预训练来源之一，**不是论文六项下游评测中的一个任务**；当前仓库没有它的专用转换器或指令任务。

## 文件验证与重下载

下载器保存每个源的 `SHA256SUMS.txt`，逐文件比对官方摘要；成功后写入 `data/download_manifest.json`（URL、大小、SHA256）。`data/download.log` 保留本次下载日志。原始数据和生成文件通过 `.gitignore` 排除。

```bash
/home_data/home/v-niuzhk/anaconda3/bin/python \
  /home_data/home/v-niuzhk/project/NeuroLM/dataset_maker/download_learning_data.py

# 本地检查，不联网；缺失或校验失败时返回错误
/home_data/home/v-niuzhk/anaconda3/bin/python \
  /home_data/home/v-niuzhk/project/NeuroLM/dataset_maker/download_learning_data.py --verify-only
```

下载中断可重新运行，使用 `.part` 续传。`RECORDS` 和官方校验清单属于原始全集元数据；HMC / EEGMMIDB 的实际下载范围以 `download_manifest.json` 为准。

本次 SN103 / SN128 的大文件下载因源站传输超时，改用源站页面提供的公开 S3 镜像 `https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/1.1/` 分段续传，合并后仍对照原站 SHA256 校验。过程保留于 `data/resume_hmc.log`。下载器已修正重试方式：每次重试重新计算本地续传偏移，避免大文件从最初偏移重传。

## 建议跑通顺序与代码入口

1. **先 Workload**：参考 `dataset_maker/prepare_workload.py`。`_1` 为静息（y=0），`_2` 为心算（y=1）；两者都取末尾 60 秒，滤波后重采样为 200 Hz，保留 19 EEG 通道，单位 µV。按原脚本 4 秒窗、2 秒步长，每条记录 29 个窗口，完整 72 条应产生 2,088 个样本。原始信号的时长、通道与单位还需在预处理时检查。
2. 原脚本的 `root` 和 `out_dir` 是写死路径。应分别指向这里的 `data/raw/eegmat-1.0.0` 和 `data/processed/EEGWorkload`。36 人按 00–25 / 26–30 / 31–35 划为 train / eval / test，先分人再切窗，重叠窗口不跨集合。
3. **再 HMC**：参考 `dataset_maker/prepare_HMC.py`，输入为 `data/raw/hmc-1.1/recordings`，保留 F4、C4、O2、C3 四个 EEG 导联及原始参考信息，以评分 TXT 对齐 30 秒窗口。输出放 `data/processed/HMC/{train,eval,test}`，按上述三人显式划分。应检查评分是否包含未识别阶段、末端窗口越界及每一组的类别覆盖。
4. **最后贯通预训练两阶段**：EEGMMIDB 原始采样率为 160 Hz，需要真正重采样为 200 Hz；电极名去掉尾部句点并转大写。参考 TUH 转换器输出 `{"X": ..., "ch_names": ...}`，而不要直接套用 TUH 的通道前缀和参考方式判断。64 通道按现有示例可切 16 秒窗（1024 patch）。按受试者划分 train / val，供 `train_vq.py` 和 `train_pretrain.py` 读取。
5. 下游加载由 `downstream_dataset.py` 完成，`train_instruction.py --dataset_dir` 应指向 `data/processed`。训练脚本内任务列表也需要缩减为本次实际准备的 Workload / HMC，否则会查找未下载的数据。完整三阶段还需要 GPT-2 / 文字 token 数据，以及相应阶段的模型权重或训练产物。

本次任务只下载和验证原始文件，没有运行滤波、生成训练样本或训练模型。上述步骤是接入指引，不代表 pipeline 已经运行成功。预处理与训练须通过 `sbatch` 在计算节点运行；三人子集适合检查流程，不能用于复现论文指标。
