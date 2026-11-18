"""Render measured JSON traces as a beginner-readable, versioned experiment record."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'data/runs/learning'


def main():
    load = lambda name: json.loads((RUN/(name+'.json')).read_text())
    env = load('environment')
    lines = ['# 入门实验实测记录', '',
             '本页由 `scripts/render_learning_report.py` 从实际运行的 JSON 追踪生成。完整概念讲解见 [逐步讲解](learning_pipeline_walkthrough.md)。', '',
             f"- Slurm 作业：`{env['slurm_job']}`；GPU：{env['gpu']}；PyTorch：{env['torch']}。",
             f"- 运行代码提交：`{env['git_commit']}`；随机种子：{env['seed']}。",
             '- 官方 NeuroLM-B 文件 SHA256：`ffe098bc138b89f8817d3710a3604498d8ecd15135080e2ca27735d05c6d29ab`。',
             '- 每阶段 16 次更新，只验证流程。验证/测试展示仅各类别两个固定窗口，不是完整数据集评估。', '',
             '## 运行中发现并处理的问题', '',
             '- 作业 4349987：样本切窗完成，但 NumPy int64 无法写入 JSON。转换为 Python int 后作业 4349989 完整完成。',
             '- 作业 4350001：初始化模型时 MKL/OpenMP 符号冲突。通过作业环境 `MKL_THREADING_LAYER=GNU` 修复后重新运行。',
             '- Git HTTPS 无凭据；改用已可访问的 SSH 连接，把独立分支推送到用户 fork。', '']
    lines += ['- 作业 4350009 四阶段均执行成功，但 HMC 的答案格式无效。独立检查作业 4350021 证明原 HMCLoader 的监督向左错一位：问题已有左括号，目标却仍从左括号开始。修复为从完整问题长度之后监督；修复前的 JSON 保存在 `data/runs/learning/attempt-4350009/`。',
              '- 修复后的七类别对齐检查与完整四阶段重跑构成最终验证；这只修正监督位置，不保证少量训练后的预测准确。','']
    explanations = {
        'tokenizer.patch_embed':'每个一秒波形经卷积和线性投影得到连续向量',
        'tokenizer.blocks.0':'EEG encoder 第一层处理电极与时间信息',
        'tokenizer':'完整 EEG encoder 输出；连续向量进入 GPT',
        'encode_transform_layer':'EEG 表示与 GPT 隐藏维度对接',
        'GPT2.transformer.h.0':'GPT 第一层整合可见的 EEG / 文字上下文',
        'GPT2.lm_head':'将隐藏向量映射为合并词表上的分数',
        'VQ.encoder.patch_embed':'波形 patch 变成 768 维',
        'VQ.encoder':'跨 patch 的 EEG 表示',
        'VQ.encode_task_layer':'768 维投影到 128 维，准备查码本',
        'VQ.quantize':'返回量化向量、量化损失与离散编号',
        'VQ.decoder_freq':'量化向量送入频率 decoder',
        'VQ.decoder_raw':'量化向量送入波形 decoder',
        'domain_classifier':'判别 EEG / 文字来自哪个域',
    }
    for stage,title in [('workload','Workload 单任务'),('multitask','Workload + HMC 多任务'),
                         ('tokenizer','运动想象：训练 tokenizer'),('autoregression','运动想象：多通道自回归')]:
        d = load(stage)
        lines += [f'## {title}', '', f"实际执行 {d['steps']} 次权重更新。首末损失记录如下（不同训练 batch，不表示严格收敛趋势）：",'',
                  '```json',json.dumps([d['losses'][0],d['losses'][-1]],ensure_ascii=False,indent=2),'```','',
                  '### 数据与关键层实际输出','']
        for name,value in d['trace'].items():
            meaning = explanations.get(name,name)
            lines += [f'**{name}：{meaning}**','', '```json',json.dumps(value,ensure_ascii=False,indent=2),'```','']
        if 'evaluation' in d:
            lines += ['### 推理输出：只有问题，没有把真实答案喂给模型','']
            for task,splits in d['evaluation'].items():
                for split,ev in splits.items():
                    lines += [f"{task} / {split}：展示窗口准确率 {ev['accuracy']:.3f}，无法解析 {ev['invalid_outputs']} 个。",'',
                              '| 文件 | 真值 | 模型原始回答 | 解析类别 |','| --- | --- | --- | --- |']
                    for r in ev['predictions']:
                        answer = r['generated_answer'].replace('\n','\\n').replace('|','\\|')
                        lines.append(f"| {r['file']} | {r['true_label']} | `{answer}` | {r['predicted_label']} |")
                    lines += ['', '第一条问题及生成 token：','```json',
                              json.dumps(ev['predictions'][0],ensure_ascii=False,indent=2),'```','']
        if 'heldout' in d:
            lines += ['### S002 留出记录的下一秒预测', '', '```json',
                      json.dumps(d['heldout'],ensure_ascii=False,indent=2),'```','']
    lines += ['## 如何解释这次结果','',
              '四阶段都有有限损失、反向更新、保存的 checkpoint 及留出样本输出，才能说教学流程已跑通。少量更新的损失与少数窗口的准确率不能证明模型已经学会任务，也不能证明多任务或预训练优于原始模型。',
              '文字辅助数据是显式保存的简短英文，不是 OpenWebText；EEG tokenizer 随机初始化后只训练少量步。自回归阶段的 GPT 复用作者权重，不能称为从零预训练。', '']
    destination = ROOT/'research/learning_run_results.md'
    destination.write_text('\n'.join(lines))
    print(destination)


if __name__ == '__main__':main()
