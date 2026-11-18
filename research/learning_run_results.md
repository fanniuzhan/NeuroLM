# 入门实验实测记录

本页由 `scripts/render_learning_report.py` 从实际运行的 JSON 追踪生成。完整概念讲解见 [逐步讲解](learning_pipeline_walkthrough.md)。

- Slurm 作业：`4350032`；GPU：NVIDIA A100 80GB PCIe；PyTorch：2.5.1+cu121。
- 运行代码提交：`ace4e8bb05c5bffbc4c1ac7b5cd5e5e4cba1c559`；随机种子：42。
- 官方 NeuroLM-B 文件 SHA256：`ffe098bc138b89f8817d3710a3604498d8ecd15135080e2ca27735d05c6d29ab`。
- 每阶段 16 次更新，只验证流程。验证/测试展示仅各类别两个固定窗口，不是完整数据集评估。

## 运行中发现并处理的问题

- 作业 4349987：样本切窗完成，但 NumPy int64 无法写入 JSON。转换为 Python int 后作业 4349989 完整完成。
- 作业 4350001：初始化模型时 MKL/OpenMP 符号冲突。通过作业环境 `MKL_THREADING_LAYER=GNU` 修复后重新运行。
- Git HTTPS 无凭据；改用已可访问的 SSH 连接，把独立分支推送到用户 fork。

- 作业 4350009 四阶段均执行成功，但 HMC 的答案格式无效。独立检查作业 4350021 证明原 HMCLoader 的监督向左错一位：问题已有左括号，目标却仍从左括号开始。修复为从完整问题长度之后监督；修复前的 JSON 保存在 `data/runs/learning/attempt-4350009/`。
- 修复后的七类别对齐检查与完整四阶段重跑构成最终验证；这只修正监督位置，不保证少量训练后的预测准确。

## 先看实际结论

四阶段运行完成。Workload 与 HMC 生成仍有错误和多余文字；只有 16 步更新，不能说模型已学会任务。HMC 对齐修复后，验证与测试各 10 个展示窗口中各有 2 个无法解析的输出，修复前各有 10 个；该前后变化还受教学抽样变化影响，不作为效果比较。
下一秒 EEG 预测也有错误；本次验证的是数据到模型、损失、梯度更新与输出的通路。

## Workload 单任务

实际执行 16 次权重更新。逐步损失记录如下（不同训练 batch，不表示严格收敛趋势）：

```json
[
  {
    "step": 0,
    "task": "Workload",
    "files": [
      "Subject21_1-023.pkl",
      "Subject01_2-015.pkl"
    ],
    "loss": 16.70833396911621,
    "gradient_norm": 2516.57177734375
  },
  {
    "step": 1,
    "task": "Workload",
    "files": [
      "Subject07_2-023.pkl",
      "Subject15_1-000.pkl"
    ],
    "loss": 6.932291507720947,
    "gradient_norm": 162.4114990234375
  },
  {
    "step": 2,
    "task": "Workload",
    "files": [
      "Subject18_1-001.pkl",
      "Subject07_2-028.pkl"
    ],
    "loss": 4.791666507720947,
    "gradient_norm": 142.0631866455078
  },
  {
    "step": 3,
    "task": "Workload",
    "files": [
      "Subject24_1-017.pkl",
      "Subject14_1-008.pkl"
    ],
    "loss": 3.09765625,
    "gradient_norm": 138.94847106933594
  },
  {
    "step": 4,
    "task": "Workload",
    "files": [
      "Subject01_1-007.pkl",
      "Subject10_2-025.pkl"
    ],
    "loss": 2.1544597148895264,
    "gradient_norm": 174.16549682617188
  },
  {
    "step": 5,
    "task": "Workload",
    "files": [
      "Subject16_2-012.pkl",
      "Subject09_1-016.pkl"
    ],
    "loss": 1.2438150644302368,
    "gradient_norm": 68.01143646240234
  },
  {
    "step": 6,
    "task": "Workload",
    "files": [
      "Subject18_1-000.pkl",
      "Subject02_2-012.pkl"
    ],
    "loss": 0.9337565302848816,
    "gradient_norm": 105.52350616455078
  },
  {
    "step": 7,
    "task": "Workload",
    "files": [
      "Subject17_2-028.pkl",
      "Subject04_1-021.pkl"
    ],
    "loss": 0.7400715947151184,
    "gradient_norm": 93.17747497558594
  },
  {
    "step": 8,
    "task": "Workload",
    "files": [
      "Subject23_1-023.pkl",
      "Subject16_2-008.pkl"
    ],
    "loss": 0.845947265625,
    "gradient_norm": 100.9796142578125
  },
  {
    "step": 9,
    "task": "Workload",
    "files": [
      "Subject16_2-013.pkl",
      "Subject07_2-009.pkl"
    ],
    "loss": 1.1124674081802368,
    "gradient_norm": 157.71876525878906
  },
  {
    "step": 10,
    "task": "Workload",
    "files": [
      "Subject08_2-025.pkl",
      "Subject15_1-009.pkl"
    ],
    "loss": 0.5775960087776184,
    "gradient_norm": 101.73121643066406
  },
  {
    "step": 11,
    "task": "Workload",
    "files": [
      "Subject16_1-008.pkl",
      "Subject06_2-006.pkl"
    ],
    "loss": 0.4734903872013092,
    "gradient_norm": 45.24417495727539
  },
  {
    "step": 12,
    "task": "Workload",
    "files": [
      "Subject22_1-028.pkl",
      "Subject04_1-004.pkl"
    ],
    "loss": 0.3108622133731842,
    "gradient_norm": 52.85708236694336
  },
  {
    "step": 13,
    "task": "Workload",
    "files": [
      "Subject12_2-022.pkl",
      "Subject25_2-000.pkl"
    ],
    "loss": 1.1312662363052368,
    "gradient_norm": 133.64956665039062
  },
  {
    "step": 14,
    "task": "Workload",
    "files": [
      "Subject11_2-005.pkl",
      "Subject25_2-003.pkl"
    ],
    "loss": 0.712890625,
    "gradient_norm": 69.7890625
  },
  {
    "step": 15,
    "task": "Workload",
    "files": [
      "Subject13_2-001.pkl",
      "Subject05_1-008.pkl"
    ],
    "loss": 0.3996785581111908,
    "gradient_norm": 19.955455780029297
  }
]
```

### 数据与关键层实际输出

**task_inputs：task_inputs**

```json
{
  "Workload": {
    "sources": [
      {
        "processed_file": "Subject21_1-023.pkl",
        "raw_file": "Subject21_1.edf",
        "start_seconds": 168.0,
        "label": 0
      },
      {
        "processed_file": "Subject01_2-015.pkl",
        "raw_file": "Subject01_2.edf",
        "start_seconds": 32.0,
        "label": 1
      }
    ],
    "tensors": [
      {
        "shape": [
          2,
          276,
          200
        ],
        "dtype": "torch.float32",
        "first_values": [
          0.05894952639937401,
          0.06324413418769836,
          0.07306116819381714,
          0.07087662070989609,
          0.04668058827519417,
          0.008831515908241272,
          -0.020990362390875816,
          -0.023470820859074593
        ]
      },
      {
        "shape": [
          2,
          80
        ],
        "dtype": "torch.int64",
        "first_values": [
          50257.0,
          24361.0,
          25.0,
          1148.0,
          428.0,
          48749.0,
          10618.0,
          286.0
        ]
      },
      {
        "shape": [
          2,
          80
        ],
        "dtype": "torch.int64",
        "first_values": [
          -1.0,
          -1.0,
          -1.0,
          -1.0,
          -1.0,
          -1.0,
          -1.0,
          -1.0
        ]
      },
      {
        "shape": [
          2,
          276
        ],
        "dtype": "torch.int32",
        "first_values": [
          0.0,
          2.0,
          17.0,
          21.0,
          15.0,
          23.0,
          88.0,
          90.0
        ]
      },
      {
        "shape": [
          2,
          276
        ],
        "dtype": "torch.int32",
        "first_values": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ]
      },
      {
        "shape": [
          2,
          276
        ],
        "dtype": "torch.bool",
        "first_values": [
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0
        ]
      },
      {
        "shape": [
          2,
          1,
          356,
          356
        ],
        "dtype": "torch.bool",
        "first_values": [
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0
        ]
      }
    ],
    "valid_patch_counts": [
      76,
      76
    ],
    "full_training_text": "Question: Is this EEG segment of high workload? Answer: No <|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>",
    "supervised_answer": " No <|endoftext|>"
  }
}
```

**tokenizer.patch_embed：每个一秒波形经卷积和线性投影得到连续向量**

```json
{
  "input": [
    {
      "shape": [
        2,
        276,
        200
      ],
      "dtype": "torch.float32",
      "first_values": [
        0.05894952639937401,
        0.06324413418769836,
        0.07306116819381714,
        0.07087662070989609,
        0.04668058827519417,
        0.008831515908241272,
        -0.020990362390875816,
        -0.023470820859074593
      ]
    }
  ],
  "output": {
    "shape": [
      2,
      276,
      768
    ],
    "dtype": "torch.bfloat16",
    "first_values": [
      -0.09130859375,
      -0.0,
      -0.0849609375,
      -0.1455078125,
      0.38671875,
      0.0947265625,
      -0.1669921875,
      -0.1611328125
    ]
  }
}
```

**tokenizer.blocks.0：EEG encoder 第一层处理电极与时间信息**

```json
{
  "input": [
    {
      "shape": [
        2,
        276,
        768
      ],
      "dtype": "torch.float32",
      "first_values": [
        -0.1150294840335846,
        0.7118448615074158,
        -0.12779295444488525,
        0.9589900970458984,
        -0.05526241660118103,
        2.242778778076172,
        -1.8447097539901733,
        -0.3071565330028534
      ]
    },
    {
      "shape": [
        2,
        1,
        276,
        276
      ],
      "dtype": "torch.bool",
      "first_values": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ]
    }
  ],
  "output": {
    "shape": [
      2,
      276,
      768
    ],
    "dtype": "torch.float32",
    "first_values": [
      0.2702244222164154,
      -0.14020591974258423,
      0.5850976705551147,
      0.7910213470458984,
      -0.5269421339035034,
      0.8191947937011719,
      -0.9697097539901733,
      -0.5722932815551758
    ]
  }
}
```

**tokenizer：完整 EEG encoder 输出；连续向量进入 GPT**

```json
{
  "input": [
    {
      "shape": [
        2,
        276,
        200
      ],
      "dtype": "torch.float32",
      "first_values": [
        0.05894952639937401,
        0.06324413418769836,
        0.07306116819381714,
        0.07087662070989609,
        0.04668058827519417,
        0.008831515908241272,
        -0.020990362390875816,
        -0.023470820859074593
      ]
    },
    {
      "shape": [
        2,
        276
      ],
      "dtype": "torch.int32",
      "first_values": [
        0.0,
        2.0,
        17.0,
        21.0,
        15.0,
        23.0,
        88.0,
        90.0
      ]
    },
    {
      "shape": [
        2,
        276
      ],
      "dtype": "torch.int32",
      "first_values": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ]
    },
    {
      "shape": [
        2,
        1,
        276,
        276
      ],
      "dtype": "torch.bool",
      "first_values": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ]
    }
  ],
  "output": {
    "shape": [
      2,
      276,
      768
    ],
    "dtype": "torch.float32",
    "first_values": [
      -0.6138486266136169,
      0.6228427886962891,
      -0.7435414791107178,
      -1.093909740447998,
      0.6224631071090698,
      1.2145086526870728,
      1.1645214557647705,
      1.467737078666687
    ]
  }
}
```

**encode_transform_layer：EEG 表示与 GPT 隐藏维度对接**

```json
{
  "input": [
    {
      "shape": [
        2,
        276,
        768
      ],
      "dtype": "torch.float32",
      "first_values": [
        -0.6138486266136169,
        0.6228427886962891,
        -0.7435414791107178,
        -1.093909740447998,
        0.6224631071090698,
        1.2145086526870728,
        1.1645214557647705,
        1.467737078666687
      ]
    }
  ],
  "output": {
    "shape": [
      2,
      276,
      768
    ],
    "dtype": "torch.float32",
    "first_values": [
      -0.6138486266136169,
      0.6228427886962891,
      -0.7435414791107178,
      -1.093909740447998,
      0.6224631071090698,
      1.2145086526870728,
      1.1645214557647705,
      1.467737078666687
    ]
  }
}
```

**GPT2.transformer.h.0：GPT 第一层整合可见的 EEG / 文字上下文**

```json
{
  "input": [
    {
      "shape": [
        2,
        356,
        768
      ],
      "dtype": "torch.float32",
      "first_values": [
        -0.5539412498474121,
        0.6172578930854797,
        -0.7442235946655273,
        -1.105816125869751,
        0.5546974539756775,
        1.2303917407989502,
        1.2316498756408691,
        1.4927549362182617
      ]
    },
    {
      "shape": [
        2,
        1,
        356,
        356
      ],
      "dtype": "torch.bool",
      "first_values": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ]
    }
  ],
  "output": {
    "shape": [
      2,
      356,
      768
    ],
    "dtype": "torch.float32",
    "first_values": [
      0.2292618751525879,
      -0.17766398191452026,
      -0.17293453216552734,
      -0.03257393836975098,
      0.1279396414756775,
      0.050704240798950195,
      0.16426706314086914,
      0.19978618621826172
    ]
  }
}
```

**GPT2.lm_head：将隐藏向量映射为合并词表上的分数**

```json
{
  "input": [
    {
      "shape": [
        2,
        356,
        768
      ],
      "dtype": "torch.float32",
      "first_values": [
        0.09258821606636047,
        0.9507675170898438,
        -9.000406265258789,
        -0.2862142026424408,
        -0.4948793947696686,
        -5.504533290863037,
        -15.436546325683594,
        0.11041878908872604
      ]
    }
  ],
  "output": {
    "shape": [
      2,
      356,
      58496
    ],
    "dtype": "torch.bfloat16",
    "first_values": [
      21.5,
      18.25,
      30.125,
      31.625,
      24.25,
      36.75,
      19.75,
      16.875
    ]
  }
}
```

**loader：loader**

```json
{
  "eeg": {
    "shape": [
      2,
      276,
      200
    ],
    "dtype": "torch.float32",
    "first_values": [
      0.05894952639937401,
      0.06324413418769836,
      0.07306116819381714,
      0.07087662070989609,
      0.04668058827519417,
      0.008831515908241272,
      -0.020990362390875816,
      -0.023470820859074593
    ]
  },
  "text": {
    "shape": [
      2,
      80
    ],
    "dtype": "torch.int64",
    "first_values": [
      50257.0,
      24361.0,
      25.0,
      1148.0,
      428.0,
      48749.0,
      10618.0,
      286.0
    ]
  },
  "answer_targets": {
    "shape": [
      2,
      80
    ],
    "dtype": "torch.int64",
    "first_values": [
      -1.0,
      -1.0,
      -1.0,
      -1.0,
      -1.0,
      -1.0,
      -1.0,
      -1.0
    ]
  },
  "channels": {
    "shape": [
      2,
      276
    ],
    "dtype": "torch.int32",
    "first_values": [
      0.0,
      2.0,
      17.0,
      21.0,
      15.0,
      23.0,
      88.0,
      90.0
    ]
  },
  "time": {
    "shape": [
      2,
      276
    ],
    "dtype": "torch.int32",
    "first_values": [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  },
  "valid": {
    "shape": [
      2,
      276
    ],
    "dtype": "torch.bool",
    "first_values": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  },
  "attention": {
    "shape": [
      2,
      1,
      356,
      356
    ],
    "dtype": "torch.bool",
    "first_values": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  }
}
```

**decoded_training_text：decoded_training_text**

```json
"Question: Is this EEG segment of high workload? Answer: No <|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
```

**supervised_text_positions：supervised_text_positions**

```json
[
  12,
  13,
  14
]
```

**supervised_target_ids：supervised_target_ids**

```json
[
  1400,
  220,
  50256
]
```

**supervised_targets_decoded：supervised_targets_decoded**

```json
" No <|endoftext|>"
```

**logits：logits**

```json
{
  "shape": [
    2,
    356,
    58496
  ],
  "dtype": "torch.bfloat16",
  "first_values": [
    21.5,
    18.25,
    30.125,
    31.625,
    24.25,
    36.75,
    19.75,
    16.875
  ]
}
```

### 推理输出：只有问题，没有把真实答案喂给模型

Workload / eval：展示窗口准确率 0.500，无法解析 0 个。

| 文件 | 真值 | 模型原始回答 | 解析类别 |
| --- | --- | --- | --- |
| Subject26_1-000.pkl | 0 | ` Yes� sidewalks� sidewalks` | 1 |
| Subject26_1-001.pkl | 0 | ` Yes’’` | 1 |
| Subject26_2-000.pkl | 1 | ` Yes’’` | 1 |
| Subject26_2-001.pkl | 1 | ` Yes’’` | 1 |

第一条问题及生成 token：
```json
{
  "file": "Subject26_1-000.pkl",
  "true_label": 0,
  "prompt": "Question: Is this EEG segment of high workload? Answer:",
  "generated_token_ids": [
    3363,
    447,
    38282,
    447,
    38282
  ],
  "generated_answer": " Yes� sidewalks� sidewalks",
  "predicted_label": 1
}
```

Workload / test：展示窗口准确率 0.250，无法解析 0 个。

| 文件 | 真值 | 模型原始回答 | 解析类别 |
| --- | --- | --- | --- |
| Subject31_1-000.pkl | 0 | ` Yes� sidewalks�Aug` | 1 |
| Subject31_1-001.pkl | 0 | ` Yes� sidewalks� sidewalks` | 1 |
| Subject31_2-000.pkl | 1 | ` Yes� sidewalks� sidewalks` | 1 |
| Subject31_2-001.pkl | 1 | ` No-B�Aug` | 0 |

第一条问题及生成 token：
```json
{
  "file": "Subject31_1-000.pkl",
  "true_label": 0,
  "prompt": "Question: Is this EEG segment of high workload? Answer:",
  "generated_token_ids": [
    3363,
    447,
    38282,
    447,
    12512
  ],
  "generated_answer": " Yes� sidewalks�Aug",
  "predicted_label": 1
}
```

## Workload + HMC 多任务

实际执行 16 次权重更新。逐步损失记录如下（不同训练 batch，不表示严格收敛趋势）：

```json
[
  {
    "step": 0,
    "task": "Workload",
    "files": [
      "Subject01_1-014.pkl",
      "Subject24_1-014.pkl"
    ],
    "loss": 13.515625,
    "gradient_norm": 1916.777587890625
  },
  {
    "step": 1,
    "task": "HMC",
    "files": [
      "SN001-0003.pkl",
      "SN001-0152.pkl"
    ],
    "loss": 7.87109375,
    "gradient_norm": 389.78717041015625
  },
  {
    "step": 2,
    "task": "Workload",
    "files": [
      "Subject03_2-003.pkl",
      "Subject09_2-004.pkl"
    ],
    "loss": 6.614583492279053,
    "gradient_norm": 178.78543090820312
  },
  {
    "step": 3,
    "task": "HMC",
    "files": [
      "SN001-0514.pkl",
      "SN001-0666.pkl"
    ],
    "loss": 3.2431640625,
    "gradient_norm": 158.7889404296875
  },
  {
    "step": 4,
    "task": "Workload",
    "files": [
      "Subject21_2-005.pkl",
      "Subject25_1-008.pkl"
    ],
    "loss": 3.9453125,
    "gradient_norm": 126.15695190429688
  },
  {
    "step": 5,
    "task": "HMC",
    "files": [
      "SN001-0531.pkl",
      "SN001-0679.pkl"
    ],
    "loss": 2.628173828125,
    "gradient_norm": 294.5879821777344
  },
  {
    "step": 6,
    "task": "Workload",
    "files": [
      "Subject02_1-019.pkl",
      "Subject06_2-016.pkl"
    ],
    "loss": 2.2957355976104736,
    "gradient_norm": 152.42813110351562
  },
  {
    "step": 7,
    "task": "HMC",
    "files": [
      "SN001-0164.pkl",
      "SN001-0815.pkl"
    ],
    "loss": 2.47216796875,
    "gradient_norm": 181.14744567871094
  },
  {
    "step": 8,
    "task": "Workload",
    "files": [
      "Subject13_2-006.pkl",
      "Subject00_1-009.pkl"
    ],
    "loss": 1.09765625,
    "gradient_norm": 63.7442512512207
  },
  {
    "step": 9,
    "task": "HMC",
    "files": [
      "SN001-0249.pkl",
      "SN001-0051.pkl"
    ],
    "loss": 2.603759765625,
    "gradient_norm": 130.0600128173828
  },
  {
    "step": 10,
    "task": "Workload",
    "files": [
      "Subject09_1-012.pkl",
      "Subject01_1-003.pkl"
    ],
    "loss": 0.7294921875,
    "gradient_norm": 77.39608001708984
  },
  {
    "step": 11,
    "task": "HMC",
    "files": [
      "SN001-0018.pkl",
      "SN001-0032.pkl"
    ],
    "loss": 2.5458984375,
    "gradient_norm": 263.8555603027344
  },
  {
    "step": 12,
    "task": "Workload",
    "files": [
      "Subject08_2-000.pkl",
      "Subject22_2-005.pkl"
    ],
    "loss": 2.75359845161438,
    "gradient_norm": 266.85418701171875
  },
  {
    "step": 13,
    "task": "HMC",
    "files": [
      "SN001-0750.pkl",
      "SN001-0214.pkl"
    ],
    "loss": 1.714111328125,
    "gradient_norm": 175.68148803710938
  },
  {
    "step": 14,
    "task": "Workload",
    "files": [
      "Subject19_1-020.pkl",
      "Subject08_1-021.pkl"
    ],
    "loss": 0.1014404296875,
    "gradient_norm": 16.123394012451172
  },
  {
    "step": 15,
    "task": "HMC",
    "files": [
      "SN001-0508.pkl",
      "SN001-0834.pkl"
    ],
    "loss": 1.3458251953125,
    "gradient_norm": 92.52975463867188
  }
]
```

### 数据与关键层实际输出

**task_inputs：task_inputs**

```json
{
  "Workload": {
    "sources": [
      {
        "processed_file": "Subject01_1-014.pkl",
        "raw_file": "Subject01_1.edf",
        "start_seconds": 150.0,
        "label": 0
      },
      {
        "processed_file": "Subject24_1-014.pkl",
        "raw_file": "Subject24_1.edf",
        "start_seconds": 150.0,
        "label": 0
      }
    ],
    "tensors": [
      {
        "shape": [
          2,
          276,
          200
        ],
        "dtype": "torch.float32",
        "first_values": [
          0.08219562470912933,
          0.05718623101711273,
          0.009778597392141819,
          -0.050105780363082886,
          -0.08476539701223373,
          -0.0890493169426918,
          -0.07969159632921219,
          -0.05026078224182129
        ]
      },
      {
        "shape": [
          2,
          80
        ],
        "dtype": "torch.int64",
        "first_values": [
          50257.0,
          24361.0,
          25.0,
          1148.0,
          428.0,
          48749.0,
          10618.0,
          286.0
        ]
      },
      {
        "shape": [
          2,
          80
        ],
        "dtype": "torch.int64",
        "first_values": [
          -1.0,
          -1.0,
          -1.0,
          -1.0,
          -1.0,
          -1.0,
          -1.0,
          -1.0
        ]
      },
      {
        "shape": [
          2,
          276
        ],
        "dtype": "torch.int32",
        "first_values": [
          0.0,
          2.0,
          17.0,
          21.0,
          15.0,
          23.0,
          88.0,
          90.0
        ]
      },
      {
        "shape": [
          2,
          276
        ],
        "dtype": "torch.int32",
        "first_values": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ]
      },
      {
        "shape": [
          2,
          276
        ],
        "dtype": "torch.bool",
        "first_values": [
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0
        ]
      },
      {
        "shape": [
          2,
          1,
          356,
          356
        ],
        "dtype": "torch.bool",
        "first_values": [
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0
        ]
      }
    ],
    "valid_patch_counts": [
      76,
      76
    ],
    "full_training_text": "Question: Is this EEG segment of high workload? Answer: No <|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>",
    "supervised_answer": " No <|endoftext|>"
  },
  "HMC": {
    "sources": [
      {
        "processed_file": "SN001-0003.pkl",
        "raw_file": "SN001.edf",
        "start_seconds": 60.0,
        "label": 0
      },
      {
        "processed_file": "SN001-0152.pkl",
        "raw_file": "SN001.edf",
        "start_seconds": 4530.0,
        "label": 2
      }
    ],
    "tensors": [
      {
        "shape": [
          2,
          276,
          200
        ],
        "dtype": "torch.float32",
        "first_values": [
          0.02089897356927395,
          0.017094749957323074,
          0.011269125156104565,
          -0.036120664328336716,
          -0.05309401452541351,
          -0.03557148575782776,
          -0.07246166467666626,
          -0.10414411872625351
        ]
      },
      {
        "shape": [
          2,
          80
        ],
        "dtype": "torch.int64",
        "first_values": [
          50257.0,
          24361.0,
          25.0,
          9022.0,
          3993.0,
          2099.0,
          857.0,
          428.0
        ]
      },
      {
        "shape": [
          2,
          80
        ],
        "dtype": "torch.int64",
        "first_values": [
          -1.0,
          -1.0,
          -1.0,
          -1.0,
          -1.0,
          -1.0,
          -1.0,
          -1.0
        ]
      },
      {
        "shape": [
          2,
          276
        ],
        "dtype": "torch.int32",
        "first_values": [
          21.0,
          43.0,
          82.0,
          39.0,
          21.0,
          43.0,
          82.0,
          39.0
        ]
      },
      {
        "shape": [
          2,
          276
        ],
        "dtype": "torch.int32",
        "first_values": [
          0.0,
          0.0,
          0.0,
          0.0,
          1.0,
          1.0,
          1.0,
          1.0
        ]
      },
      {
        "shape": [
          2,
          276
        ],
        "dtype": "torch.bool",
        "first_values": [
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0,
          1.0
        ]
      },
      {
        "shape": [
          2,
          1,
          356,
          356
        ],
        "dtype": "torch.bool",
        "first_values": [
          1.0,
          1.0,
          1.0,
          1.0,
          0.0,
          0.0,
          0.0,
          0.0
        ]
      }
    ],
    "valid_patch_counts": [
      120,
      120
    ],
    "full_training_text": "Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (A) <|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>",
    "supervised_answer": "A) <|endoftext|>"
  }
}
```

**tokenizer.patch_embed：每个一秒波形经卷积和线性投影得到连续向量**

```json
{
  "input": [
    {
      "shape": [
        2,
        276,
        200
      ],
      "dtype": "torch.float32",
      "first_values": [
        0.08219562470912933,
        0.05718623101711273,
        0.009778597392141819,
        -0.050105780363082886,
        -0.08476539701223373,
        -0.0890493169426918,
        -0.07969159632921219,
        -0.05026078224182129
      ]
    }
  ],
  "output": {
    "shape": [
      2,
      276,
      768
    ],
    "dtype": "torch.bfloat16",
    "first_values": [
      0.283203125,
      -0.0,
      -0.1318359375,
      -0.138671875,
      0.3046875,
      -0.03369140625,
      -0.16015625,
      -0.0927734375
    ]
  }
}
```

**tokenizer.blocks.0：EEG encoder 第一层处理电极与时间信息**

```json
{
  "input": [
    {
      "shape": [
        2,
        276,
        768
      ],
      "dtype": "torch.float32",
      "first_values": [
        0.2594822645187378,
        0.7118448615074158,
        -0.17466795444488525,
        0.9658260345458984,
        -0.13729366660118103,
        2.114360809326172,
        -1.8378738164901733,
        -0.2387971580028534
      ]
    },
    {
      "shape": [
        2,
        1,
        276,
        276
      ],
      "dtype": "torch.bool",
      "first_values": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ]
    }
  ],
  "output": {
    "shape": [
      2,
      276,
      768
    ],
    "dtype": "torch.float32",
    "first_values": [
      0.6225193738937378,
      0.01750892400741577,
      0.38001954555511475,
      0.7529354095458984,
      -0.5620983839035034,
      0.6915092468261719,
      -0.9023269414901733,
      -0.7309846878051758
    ]
  }
}
```

**tokenizer：完整 EEG encoder 输出；连续向量进入 GPT**

```json
{
  "input": [
    {
      "shape": [
        2,
        276,
        200
      ],
      "dtype": "torch.float32",
      "first_values": [
        0.08219562470912933,
        0.05718623101711273,
        0.009778597392141819,
        -0.050105780363082886,
        -0.08476539701223373,
        -0.0890493169426918,
        -0.07969159632921219,
        -0.05026078224182129
      ]
    },
    {
      "shape": [
        2,
        276
      ],
      "dtype": "torch.int32",
      "first_values": [
        0.0,
        2.0,
        17.0,
        21.0,
        15.0,
        23.0,
        88.0,
        90.0
      ]
    },
    {
      "shape": [
        2,
        276
      ],
      "dtype": "torch.int32",
      "first_values": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ]
    },
    {
      "shape": [
        2,
        1,
        276,
        276
      ],
      "dtype": "torch.bool",
      "first_values": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ]
    }
  ],
  "output": {
    "shape": [
      2,
      276,
      768
    ],
    "dtype": "torch.float32",
    "first_values": [
      -0.6138689517974854,
      0.8698011040687561,
      -1.0276541709899902,
      -1.5427495241165161,
      1.2762224674224854,
      1.0830668210983276,
      0.9424499273300171,
      1.1814709901809692
    ]
  }
}
```

**encode_transform_layer：EEG 表示与 GPT 隐藏维度对接**

```json
{
  "input": [
    {
      "shape": [
        2,
        276,
        768
      ],
      "dtype": "torch.float32",
      "first_values": [
        -0.6138689517974854,
        0.8698011040687561,
        -1.0276541709899902,
        -1.5427495241165161,
        1.2762224674224854,
        1.0830668210983276,
        0.9424499273300171,
        1.1814709901809692
      ]
    }
  ],
  "output": {
    "shape": [
      2,
      276,
      768
    ],
    "dtype": "torch.float32",
    "first_values": [
      -0.6138689517974854,
      0.8698011040687561,
      -1.0276541709899902,
      -1.5427495241165161,
      1.2762224674224854,
      1.0830668210983276,
      0.9424499273300171,
      1.1814709901809692
    ]
  }
}
```

**GPT2.transformer.h.0：GPT 第一层整合可见的 EEG / 文字上下文**

```json
{
  "input": [
    {
      "shape": [
        2,
        356,
        768
      ],
      "dtype": "torch.float32",
      "first_values": [
        -0.5539615750312805,
        0.8642162084579468,
        -1.0283362865447998,
        -1.554655909538269,
        1.2084568738937378,
        1.098949909210205,
        1.0095783472061157,
        1.206488847732544
      ]
    },
    {
      "shape": [
        2,
        1,
        356,
        356
      ],
      "dtype": "torch.bool",
      "first_values": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ]
    }
  ],
  "output": {
    "shape": [
      2,
      356,
      768
    ],
    "dtype": "torch.float32",
    "first_values": [
      0.06127279996871948,
      -0.24247324466705322,
      -0.5312659740447998,
      0.020539402961730957,
      0.05025374889373779,
      0.17121553421020508,
      0.3093830347061157,
      0.10297322273254395
    ]
  }
}
```

**GPT2.lm_head：将隐藏向量映射为合并词表上的分数**

```json
{
  "input": [
    {
      "shape": [
        2,
        356,
        768
      ],
      "dtype": "torch.float32",
      "first_values": [
        1.0610908269882202,
        -1.8356138467788696,
        -5.891976356506348,
        0.7119574546813965,
        -1.1472374200820923,
        -4.126909255981445,
        -8.7566556930542,
        -0.9348742961883545
      ]
    }
  ],
  "output": {
    "shape": [
      2,
      356,
      58496
    ],
    "dtype": "torch.bfloat16",
    "first_values": [
      22.75,
      18.625,
      32.0,
      35.25,
      26.25,
      39.0,
      21.0,
      18.5
    ]
  }
}
```

**loader：loader**

```json
{
  "eeg": {
    "shape": [
      2,
      276,
      200
    ],
    "dtype": "torch.float32",
    "first_values": [
      0.08219562470912933,
      0.05718623101711273,
      0.009778597392141819,
      -0.050105780363082886,
      -0.08476539701223373,
      -0.0890493169426918,
      -0.07969159632921219,
      -0.05026078224182129
    ]
  },
  "text": {
    "shape": [
      2,
      80
    ],
    "dtype": "torch.int64",
    "first_values": [
      50257.0,
      24361.0,
      25.0,
      1148.0,
      428.0,
      48749.0,
      10618.0,
      286.0
    ]
  },
  "answer_targets": {
    "shape": [
      2,
      80
    ],
    "dtype": "torch.int64",
    "first_values": [
      -1.0,
      -1.0,
      -1.0,
      -1.0,
      -1.0,
      -1.0,
      -1.0,
      -1.0
    ]
  },
  "channels": {
    "shape": [
      2,
      276
    ],
    "dtype": "torch.int32",
    "first_values": [
      0.0,
      2.0,
      17.0,
      21.0,
      15.0,
      23.0,
      88.0,
      90.0
    ]
  },
  "time": {
    "shape": [
      2,
      276
    ],
    "dtype": "torch.int32",
    "first_values": [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  },
  "valid": {
    "shape": [
      2,
      276
    ],
    "dtype": "torch.bool",
    "first_values": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  },
  "attention": {
    "shape": [
      2,
      1,
      356,
      356
    ],
    "dtype": "torch.bool",
    "first_values": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  }
}
```

**decoded_training_text：decoded_training_text**

```json
"Question: Is this EEG segment of high workload? Answer: No <|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"
```

**supervised_text_positions：supervised_text_positions**

```json
[
  12,
  13,
  14
]
```

**supervised_target_ids：supervised_target_ids**

```json
[
  1400,
  220,
  50256
]
```

**supervised_targets_decoded：supervised_targets_decoded**

```json
" No <|endoftext|>"
```

**logits：logits**

```json
{
  "shape": [
    2,
    356,
    58496
  ],
  "dtype": "torch.bfloat16",
  "first_values": [
    22.75,
    18.625,
    32.0,
    35.25,
    26.25,
    39.0,
    21.0,
    18.5
  ]
}
```

### 推理输出：只有问题，没有把真实答案喂给模型

Workload / eval：展示窗口准确率 0.500，无法解析 0 个。

| 文件 | 真值 | 模型原始回答 | 解析类别 |
| --- | --- | --- | --- |
| Subject26_1-000.pkl | 0 | ` No-B’` | 0 |
| Subject26_1-001.pkl | 0 | ` No-B’` | 0 |
| Subject26_2-000.pkl | 1 | ` No-B’` | 0 |
| Subject26_2-001.pkl | 1 | ` No-B’` | 0 |

第一条问题及生成 token：
```json
{
  "file": "Subject26_1-000.pkl",
  "true_label": 0,
  "prompt": "Question: Is this EEG segment of high workload? Answer:",
  "generated_token_ids": [
    1400,
    12,
    33,
    447,
    247
  ],
  "generated_answer": " No-B’",
  "predicted_label": 0
}
```

Workload / test：展示窗口准确率 0.500，无法解析 0 个。

| 文件 | 真值 | 模型原始回答 | 解析类别 |
| --- | --- | --- | --- |
| Subject31_1-000.pkl | 0 | ` No-B’` | 0 |
| Subject31_1-001.pkl | 0 | ` No.�� sidewalks` | 0 |
| Subject31_2-000.pkl | 1 | ` No-B�Thomas` | 0 |
| Subject31_2-001.pkl | 1 | ` No-B� sidewalks` | 0 |

第一条问题及生成 token：
```json
{
  "file": "Subject31_1-000.pkl",
  "true_label": 0,
  "prompt": "Question: Is this EEG segment of high workload? Answer:",
  "generated_token_ids": [
    1400,
    12,
    33,
    447,
    247
  ],
  "generated_answer": " No-B’",
  "predicted_label": 0
}
```

HMC / eval：展示窗口准确率 0.100，无法解析 2 个。

| 文件 | 真值 | 模型原始回答 | 解析类别 |
| --- | --- | --- | --- |
| SN103-0000.pkl | 0 | `B)B))` | 1 |
| SN103-0001.pkl | 0 | `C)\n\n\n` | 2 |
| SN103-0012.pkl | 1 | `C)\n))` | 2 |
| SN103-0013.pkl | 1 | ` hers)))` | -1 |
| SN103-0015.pkl | 2 | `C)\n))` | 2 |
| SN103-0016.pkl | 2 | `)\n\n))` | -1 |
| SN103-0043.pkl | 3 | `C)\n))` | 2 |
| SN103-0044.pkl | 3 | `C)\n))` | 2 |
| SN103-0119.pkl | 4 | `C)\n))` | 2 |
| SN103-0120.pkl | 4 | `C)\n))` | 2 |

第一条问题及生成 token：
```json
{
  "file": "SN103-0000.pkl",
  "true_label": 0,
  "prompt": "Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (",
  "generated_token_ids": [
    33,
    8,
    33,
    8,
    8
  ],
  "generated_answer": "B)B))",
  "predicted_label": 1
}
```

HMC / test：展示窗口准确率 0.200，无法解析 2 个。

| 文件 | 真值 | 模型原始回答 | 解析类别 |
| --- | --- | --- | --- |
| SN128-0000.pkl | 0 | `)\n\n))` | -1 |
| SN128-0001.pkl | 0 | `C)\n))` | 2 |
| SN128-0010.pkl | 1 | `C)\n))` | 2 |
| SN128-0011.pkl | 1 | `C)\n))` | 2 |
| SN128-0016.pkl | 2 | `C) ))` | 2 |
| SN128-0017.pkl | 2 | `C)\n))` | 2 |
| SN128-0048.pkl | 3 | `C) ))` | 2 |
| SN128-0049.pkl | 3 | ` hers)))` | -1 |
| SN128-0097.pkl | 4 | `C)\n))` | 2 |
| SN128-0098.pkl | 4 | `C)\n))` | 2 |

第一条问题及生成 token：
```json
{
  "file": "SN128-0000.pkl",
  "true_label": 0,
  "prompt": "Question: Which sleep type does this EEG segment belong to? Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. (E) REM. Answer: (",
  "generated_token_ids": [
    8,
    198,
    198,
    8,
    8
  ],
  "generated_answer": ")\n\n))",
  "predicted_label": -1
}
```

## 运动想象：训练 tokenizer

实际执行 16 次权重更新。逐步损失记录如下（不同训练 batch，不表示严格收敛趋势）：

```json
[
  {
    "step": 0,
    "total_loss": 3.8248674869537354,
    "EEG_domain_loss": 0.7051315307617188,
    "text_domain_loss": 0.69305419921875,
    "train/quant_loss": 1.6880307847594138e-18,
    "train/rec_freq_loss": 1.354941725730896,
    "train/rec_raw_loss": 1.0717400312423706,
    "train/total_loss": 2.4266817569732666,
    "train/domain_loss": 0.7051315307617188
  },
  {
    "step": 1,
    "total_loss": 4.046459197998047,
    "EEG_domain_loss": 0.7112503051757812,
    "text_domain_loss": 0.69305419921875,
    "train/quant_loss": 0.0014894832856953144,
    "train/rec_freq_loss": 1.5692098140716553,
    "train/rec_raw_loss": 1.0714555978775024,
    "train/total_loss": 2.6421549320220947,
    "train/domain_loss": 0.7112503051757812
  },
  {
    "step": 2,
    "total_loss": 3.996039867401123,
    "EEG_domain_loss": 0.9345283508300781,
    "text_domain_loss": 0.693206787109375,
    "train/quant_loss": 0.002140580676496029,
    "train/rec_freq_loss": 1.3017971515655518,
    "train/rec_raw_loss": 1.0643669366836548,
    "train/total_loss": 2.36830472946167,
    "train/domain_loss": 0.9345283508300781
  },
  {
    "step": 3,
    "total_loss": 4.1994404792785645,
    "EEG_domain_loss": 1.2330970764160156,
    "text_domain_loss": 0.693206787109375,
    "train/quant_loss": 0.002668794011697173,
    "train/rec_freq_loss": 1.2125195264816284,
    "train/rec_raw_loss": 1.0579484701156616,
    "train/total_loss": 2.273136615753174,
    "train/domain_loss": 1.2330970764160156
  },
  {
    "step": 4,
    "total_loss": 4.323963165283203,
    "EEG_domain_loss": 1.6003570556640625,
    "text_domain_loss": 0.693328857421875,
    "train/quant_loss": 0.0032353473361581564,
    "train/rec_freq_loss": 0.9762149453163147,
    "train/rec_raw_loss": 1.0508272647857666,
    "train/total_loss": 2.0302774906158447,
    "train/domain_loss": 1.6003570556640625
  },
  {
    "step": 5,
    "total_loss": 4.705869197845459,
    "EEG_domain_loss": 2.0053253173828125,
    "text_domain_loss": 0.693328857421875,
    "train/quant_loss": 0.0038564163260161877,
    "train/rec_freq_loss": 0.9361894130706787,
    "train/rec_raw_loss": 1.067169189453125,
    "train/total_loss": 2.0072150230407715,
    "train/domain_loss": 2.0053253173828125
  },
  {
    "step": 6,
    "total_loss": 5.169229030609131,
    "EEG_domain_loss": 2.4127044677734375,
    "text_domain_loss": 0.693328857421875,
    "train/quant_loss": 0.004535093903541565,
    "train/rec_freq_loss": 1.0063728094100952,
    "train/rec_raw_loss": 1.0522879362106323,
    "train/total_loss": 2.0631957054138184,
    "train/domain_loss": 2.4127044677734375
  },
  {
    "step": 7,
    "total_loss": 5.434338569641113,
    "EEG_domain_loss": 2.763824462890625,
    "text_domain_loss": 0.693328857421875,
    "train/quant_loss": 0.005153269041329622,
    "train/rec_freq_loss": 0.9202468395233154,
    "train/rec_raw_loss": 1.0517851114273071,
    "train/total_loss": 1.9771852493286133,
    "train/domain_loss": 2.763824462890625
  },
  {
    "step": 8,
    "total_loss": 5.673880577087402,
    "EEG_domain_loss": 3.043609619140625,
    "text_domain_loss": 0.693328857421875,
    "train/quant_loss": 0.00565909780561924,
    "train/rec_freq_loss": 0.8690575957298279,
    "train/rec_raw_loss": 1.062225341796875,
    "train/total_loss": 1.9369421005249023,
    "train/domain_loss": 3.043609619140625
  },
  {
    "step": 9,
    "total_loss": 5.801133155822754,
    "EEG_domain_loss": 3.2480010986328125,
    "text_domain_loss": 0.693328857421875,
    "train/quant_loss": 0.006022636778652668,
    "train/rec_freq_loss": 0.8219439387321472,
    "train/rec_raw_loss": 1.031836748123169,
    "train/total_loss": 1.859803318977356,
    "train/domain_loss": 3.2480010986328125
  },
  {
    "step": 10,
    "total_loss": 5.896430015563965,
    "EEG_domain_loss": 3.388519287109375,
    "text_domain_loss": 0.693756103515625,
    "train/quant_loss": 0.0062275901436805725,
    "train/rec_freq_loss": 0.7813023328781128,
    "train/rec_raw_loss": 1.0266245603561401,
    "train/total_loss": 1.8141545057296753,
    "train/domain_loss": 3.388519287109375
  },
  {
    "step": 11,
    "total_loss": 5.950345993041992,
    "EEG_domain_loss": 3.47625732421875,
    "text_domain_loss": 0.693756103515625,
    "train/quant_loss": 0.006261311937123537,
    "train/rec_freq_loss": 0.7424576878547668,
    "train/rec_raw_loss": 1.0316137075424194,
    "train/total_loss": 1.7803326845169067,
    "train/domain_loss": 3.47625732421875
  },
  {
    "step": 12,
    "total_loss": 6.14484977722168,
    "EEG_domain_loss": 3.5245208740234375,
    "text_domain_loss": 0.69390869140625,
    "train/quant_loss": 0.006138404365628958,
    "train/rec_freq_loss": 0.852716326713562,
    "train/rec_raw_loss": 1.0675653219223022,
    "train/total_loss": 1.926419973373413,
    "train/domain_loss": 3.5245208740234375
  },
  {
    "step": 13,
    "total_loss": 5.986968517303467,
    "EEG_domain_loss": 3.551605224609375,
    "text_domain_loss": 0.69390869140625,
    "train/quant_loss": 0.005837840959429741,
    "train/rec_freq_loss": 0.7527387142181396,
    "train/rec_raw_loss": 0.9828780889511108,
    "train/total_loss": 1.7414546012878418,
    "train/domain_loss": 3.551605224609375
  },
  {
    "step": 14,
    "total_loss": 6.291786193847656,
    "EEG_domain_loss": 3.555999755859375,
    "text_domain_loss": 0.694061279296875,
    "train/quant_loss": 0.005566191393882036,
    "train/rec_freq_loss": 1.0586785078048706,
    "train/rec_raw_loss": 0.9774804711341858,
    "train/total_loss": 2.0417251586914062,
    "train/domain_loss": 3.555999755859375
  },
  {
    "step": 15,
    "total_loss": 6.059067249298096,
    "EEG_domain_loss": 3.5432281494140625,
    "text_domain_loss": 0.694061279296875,
    "train/quant_loss": 0.005230001173913479,
    "train/rec_freq_loss": 0.8280602693557739,
    "train/rec_raw_loss": 0.9884876012802124,
    "train/total_loss": 1.8217778205871582,
    "train/domain_loss": 3.5432281494140625
  }
]
```

### 数据与关键层实际输出

**VQ.encoder.patch_embed：波形 patch 变成 768 维**

```json
{
  "input": [
    {
      "shape": [
        1,
        1024,
        200
      ],
      "dtype": "torch.float32",
      "first_values": [
        0.0,
        -0.35736653208732605,
        -0.41484272480010986,
        -0.38017404079437256,
        -0.30920714139938354,
        -0.20357254147529602,
        -0.2679107189178467,
        -0.2516748309135437
      ]
    }
  ],
  "output": {
    "shape": [
      1,
      1024,
      768
    ],
    "dtype": "torch.bfloat16",
    "first_values": [
      -0.134765625,
      0.056640625,
      0.003662109375,
      0.04736328125,
      0.130859375,
      0.080078125,
      -0.04638671875,
      -0.09521484375
    ]
  }
}
```

**VQ.encoder：跨 patch 的 EEG 表示**

```json
{
  "input": [
    {
      "shape": [
        1,
        1024,
        200
      ],
      "dtype": "torch.float32",
      "first_values": [
        0.0,
        -0.35736653208732605,
        -0.41484272480010986,
        -0.38017404079437256,
        -0.30920714139938354,
        -0.20357254147529602,
        -0.2679107189178467,
        -0.2516748309135437
      ]
    },
    {
      "shape": [
        1,
        1024
      ],
      "dtype": "torch.int32",
      "first_values": [
        27.0,
        28.0,
        29.0,
        30.0,
        31.0,
        32.0,
        33.0,
        38.0
      ]
    },
    {
      "shape": [
        1,
        1024
      ],
      "dtype": "torch.int32",
      "first_values": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ]
    },
    {
      "shape": [
        1,
        1,
        1024,
        1024
      ],
      "dtype": "torch.bool",
      "first_values": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ]
    }
  ],
  "output": {
    "shape": [
      1,
      1024,
      768
    ],
    "dtype": "torch.float32",
    "first_values": [
      -0.2295931875705719,
      1.040188193321228,
      -0.30615293979644775,
      0.3488064706325531,
      -0.7152860760688782,
      0.510679304599762,
      -1.0236328840255737,
      0.3147096335887909
    ]
  }
}
```

**VQ.encode_task_layer：768 维投影到 128 维，准备查码本**

```json
{
  "input": [
    {
      "shape": [
        1,
        1024,
        768
      ],
      "dtype": "torch.float32",
      "first_values": [
        -0.2295931875705719,
        1.040188193321228,
        -0.30615293979644775,
        0.3488064706325531,
        -0.7152860760688782,
        0.510679304599762,
        -1.0236328840255737,
        0.3147096335887909
      ]
    }
  ],
  "output": {
    "shape": [
      1,
      1024,
      128
    ],
    "dtype": "torch.float32",
    "first_values": [
      0.36704766750335693,
      -0.26626646518707275,
      0.25586143136024475,
      0.08302189409732819,
      0.42945343255996704,
      0.027404502034187317,
      0.28159099817276,
      -0.2819708585739136
    ]
  }
}
```

**VQ.quantize：返回量化向量、量化损失与离散编号**

```json
{
  "input": [
    {
      "shape": [
        1,
        1024,
        128
      ],
      "dtype": "torch.float32",
      "first_values": [
        0.36704766750335693,
        -0.26626646518707275,
        0.25586143136024475,
        0.08302189409732819,
        0.42945343255996704,
        0.027404502034187317,
        0.28159099817276,
        -0.2819708585739136
      ]
    }
  ],
  "output": [
    {
      "shape": [
        1,
        1024,
        128
      ],
      "dtype": "torch.float32",
      "first_values": [
        0.13617479801177979,
        -0.09878494590520859,
        0.0949246734380722,
        0.03080114908516407,
        0.15932735800743103,
        0.010167079046368599,
        0.10447034984827042,
        -0.10461127758026123
      ]
    },
    {
      "shape": [],
      "dtype": "torch.float32",
      "first_values": [
        1.6880307847594138e-18
      ]
    },
    {
      "shape": [
        1024
      ],
      "dtype": "torch.int64",
      "first_values": [
        4741.0,
        93.0,
        2171.0,
        1168.0,
        1042.0,
        757.0,
        485.0,
        915.0
      ]
    }
  ]
}
```

**VQ.decoder_freq：量化向量送入频率 decoder**

```json
{
  "input": [
    {
      "shape": [
        1,
        1024,
        128
      ],
      "dtype": "torch.float32",
      "first_values": [
        0.13617479801177979,
        -0.09878494590520859,
        0.0949246734380722,
        0.03080114908516407,
        0.15932735800743103,
        0.010167079046368599,
        0.10447034984827042,
        -0.10461127758026123
      ]
    },
    {
      "shape": [
        1,
        1024
      ],
      "dtype": "torch.int32",
      "first_values": [
        27.0,
        28.0,
        29.0,
        30.0,
        31.0,
        32.0,
        33.0,
        38.0
      ]
    },
    {
      "shape": [
        1,
        1024
      ],
      "dtype": "torch.int32",
      "first_values": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ]
    },
    {
      "shape": [
        1,
        1,
        1024,
        1024
      ],
      "dtype": "torch.bool",
      "first_values": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ]
    }
  ],
  "output": {
    "shape": [
      1,
      1024,
      768
    ],
    "dtype": "torch.float32",
    "first_values": [
      1.3905869722366333,
      0.2152865082025528,
      0.5071491599082947,
      -0.03718407452106476,
      -0.48267775774002075,
      0.747094452381134,
      0.5800036787986755,
      0.5968402028083801
    ]
  }
}
```

**VQ.decoder_raw：量化向量送入波形 decoder**

```json
{
  "input": [
    {
      "shape": [
        1,
        1024,
        128
      ],
      "dtype": "torch.float32",
      "first_values": [
        0.13617479801177979,
        -0.09878494590520859,
        0.0949246734380722,
        0.03080114908516407,
        0.15932735800743103,
        0.010167079046368599,
        0.10447034984827042,
        -0.10461127758026123
      ]
    },
    {
      "shape": [
        1,
        1024
      ],
      "dtype": "torch.int32",
      "first_values": [
        27.0,
        28.0,
        29.0,
        30.0,
        31.0,
        32.0,
        33.0,
        38.0
      ]
    },
    {
      "shape": [
        1,
        1024
      ],
      "dtype": "torch.int32",
      "first_values": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ]
    },
    {
      "shape": [
        1,
        1,
        1024,
        1024
      ],
      "dtype": "torch.bool",
      "first_values": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ]
    }
  ],
  "output": {
    "shape": [
      1,
      1024,
      768
    ],
    "dtype": "torch.float32",
    "first_values": [
      -0.7131578326225281,
      0.0431072823703289,
      1.9015123844146729,
      0.2253638058900833,
      0.19189567863941193,
      -1.3138357400894165,
      -0.7393704652786255,
      2.8959546089172363
    ]
  }
}
```

**domain_classifier：判别 EEG / 文字来自哪个域**

```json
{
  "input": [
    {
      "shape": [
        1,
        1024,
        768
      ],
      "dtype": "torch.float32",
      "first_values": [
        -0.2295931875705719,
        1.040188193321228,
        -0.30615293979644775,
        0.3488064706325531,
        -0.7152860760688782,
        0.510679304599762,
        -1.0236328840255737,
        0.3147096335887909
      ]
    }
  ],
  "output": {
    "shape": [
      1,
      1024,
      2
    ],
    "dtype": "torch.bfloat16",
    "first_values": [
      -0.00970458984375,
      0.060791015625,
      0.035400390625,
      0.07470703125,
      0.1162109375,
      0.009521484375,
      0.06640625,
      -0.0260009765625
    ]
  }
}
```

**loader：loader**

```json
[
  {
    "shape": [
      1,
      1024,
      200
    ],
    "dtype": "torch.float32",
    "first_values": [
      0.0,
      -0.35736653208732605,
      -0.41484272480010986,
      -0.38017404079437256,
      -0.30920714139938354,
      -0.20357254147529602,
      -0.2679107189178467,
      -0.2516748309135437
    ]
  },
  {
    "shape": [
      1,
      1024,
      100
    ],
    "dtype": "torch.float32",
    "first_values": [
      2.661433696746826,
      3.2857422828674316,
      1.7939071655273438,
      1.9713157415390015,
      1.859724521636963,
      0.7090606689453125,
      0.6426541209220886,
      0.6248266696929932
    ]
  },
  {
    "shape": [
      1,
      1024,
      200
    ],
    "dtype": "torch.float32",
    "first_values": [
      -0.06946524977684021,
      -0.7997724413871765,
      -0.9172296524047852,
      -0.846381425857544,
      -0.7013548612594604,
      -0.4854820966720581,
      -0.6169623136520386,
      -0.583782970905304
    ]
  },
  {
    "shape": [
      1,
      1024
    ],
    "dtype": "torch.int32",
    "first_values": [
      27.0,
      28.0,
      29.0,
      30.0,
      31.0,
      32.0,
      33.0,
      38.0
    ]
  },
  {
    "shape": [
      1,
      1024
    ],
    "dtype": "torch.int32",
    "first_values": [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  },
  {
    "shape": [
      1,
      1024
    ],
    "dtype": "torch.bool",
    "first_values": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  }
]
```

## 运动想象：多通道自回归

实际执行 16 次权重更新。逐步损失记录如下（不同训练 batch，不表示严格收敛趋势）：

```json
[
  {
    "step": 0,
    "EEG_loss": 13.695637702941895,
    "text_loss": 1.0985996723175049
  },
  {
    "step": 1,
    "EEG_loss": 10.605729103088379,
    "text_loss": 0.9577939510345459
  },
  {
    "step": 2,
    "EEG_loss": 9.831510543823242,
    "text_loss": 0.8503578901290894
  },
  {
    "step": 3,
    "EEG_loss": 9.670345306396484,
    "text_loss": 0.7438265085220337
  },
  {
    "step": 4,
    "EEG_loss": 9.294629096984863,
    "text_loss": 0.6480824947357178
  },
  {
    "step": 5,
    "EEG_loss": 9.403027534484863,
    "text_loss": 0.5544439554214478
  },
  {
    "step": 6,
    "EEG_loss": 9.383333206176758,
    "text_loss": 0.4712708294391632
  },
  {
    "step": 7,
    "EEG_loss": 9.097916603088379,
    "text_loss": 0.4051179587841034
  },
  {
    "step": 8,
    "EEG_loss": 8.934179306030273,
    "text_loss": 0.3458274006843567
  },
  {
    "step": 9,
    "EEG_loss": 8.920735359191895,
    "text_loss": 0.2812694013118744
  },
  {
    "step": 10,
    "EEG_loss": 8.823990821838379,
    "text_loss": 0.23128299415111542
  },
  {
    "step": 11,
    "EEG_loss": 8.721776962280273,
    "text_loss": 0.18292847275733948
  },
  {
    "step": 12,
    "EEG_loss": 8.699121475219727,
    "text_loss": 0.13540536165237427
  },
  {
    "step": 13,
    "EEG_loss": 8.619824409484863,
    "text_loss": 0.10548905283212662
  },
  {
    "step": 14,
    "EEG_loss": 8.535025596618652,
    "text_loss": 0.07793240994215012
  },
  {
    "step": 15,
    "EEG_loss": 8.487239837646484,
    "text_loss": 0.05235478654503822
  }
]
```

### 数据与关键层实际输出

**tokenizer：完整 EEG encoder 输出；连续向量进入 GPT**

```json
{
  "input": [
    {
      "shape": [
        1,
        1024,
        200
      ],
      "dtype": "torch.float32",
      "first_values": [
        0.0,
        -0.35736653208732605,
        -0.41484272480010986,
        -0.38017404079437256,
        -0.30920714139938354,
        -0.20357254147529602,
        -0.2679107189178467,
        -0.2516748309135437
      ]
    },
    {
      "shape": [
        1,
        1024
      ],
      "dtype": "torch.int32",
      "first_values": [
        27.0,
        28.0,
        29.0,
        30.0,
        31.0,
        32.0,
        33.0,
        38.0
      ]
    },
    {
      "shape": [
        1,
        1024
      ],
      "dtype": "torch.int32",
      "first_values": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ]
    },
    {
      "shape": [
        1,
        1,
        1024,
        1024
      ],
      "dtype": "torch.bool",
      "first_values": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ]
    }
  ],
  "output": {
    "shape": [
      1,
      1024,
      768
    ],
    "dtype": "torch.float32",
    "first_values": [
      -0.5518482327461243,
      0.15996891260147095,
      1.778295874595642,
      0.03762798756361008,
      -0.5435000061988831,
      0.2648809552192688,
      -1.6713169813156128,
      0.012007876299321651
    ]
  }
}
```

**GPT2.transformer.h.0：GPT 第一层整合可见的 EEG / 文字上下文**

```json
{
  "input": [
    {
      "shape": [
        1,
        1024,
        768
      ],
      "dtype": "torch.float32",
      "first_values": [
        -0.5383232831954956,
        0.2142423540353775,
        1.7414549589157104,
        -0.035421304404735565,
        -0.561882734298706,
        0.20134685933589935,
        -1.6408077478408813,
        0.032152168452739716
      ]
    },
    {
      "shape": [
        1,
        1,
        1024,
        1024
      ],
      "dtype": "torch.bool",
      "first_values": [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
      ]
    }
  ],
  "output": {
    "shape": [
      1,
      1024,
      768
    ],
    "dtype": "torch.float32",
    "first_values": [
      0.048712849617004395,
      0.3914884328842163,
      1.2180174589157104,
      1.0402623414993286,
      -0.512627363204956,
      -0.12482501566410065,
      -1.4767452478408813,
      0.4515857696533203
    ]
  }
}
```

**GPT2.lm_head：将隐藏向量映射为合并词表上的分数**

```json
{
  "input": [
    {
      "shape": [
        1,
        1024,
        768
      ],
      "dtype": "torch.float32",
      "first_values": [
        8.356498718261719,
        -0.62642902135849,
        10.87645435333252,
        2.262484073638916,
        0.08976514637470245,
        -1.6103581190109253,
        -9.832477569580078,
        6.62510347366333
      ]
    }
  ],
  "output": {
    "shape": [
      1,
      1024,
      58496
    ],
    "dtype": "torch.bfloat16",
    "first_values": [
      2.453125,
      2.15625,
      -2.0,
      -2.578125,
      -1.25,
      -0.8046875,
      0.1875,
      -0.828125
    ]
  }
}
```

**loader：loader**

```json
[
  {
    "shape": [
      1,
      1024,
      200
    ],
    "dtype": "torch.float32",
    "first_values": [
      0.0,
      -0.35736653208732605,
      -0.41484272480010986,
      -0.38017404079437256,
      -0.30920714139938354,
      -0.20357254147529602,
      -0.2679107189178467,
      -0.2516748309135437
    ]
  },
  {
    "shape": [
      1,
      1024
    ],
    "dtype": "torch.int32",
    "first_values": [
      27.0,
      28.0,
      29.0,
      30.0,
      31.0,
      32.0,
      33.0,
      38.0
    ]
  },
  {
    "shape": [
      1,
      1024
    ],
    "dtype": "torch.int32",
    "first_values": [
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ]
  },
  {
    "shape": [
      1,
      1024
    ],
    "dtype": "torch.bool",
    "first_values": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  },
  {
    "shape": [
      1,
      1,
      1024,
      1024
    ],
    "dtype": "torch.bool",
    "first_values": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ]
  }
]
```

**code_ids_first_second：code_ids_first_second**

```json
[
  1231,
  93,
  1173,
  1168,
  1042,
  757,
  7,
  915,
  1445,
  165,
  115,
  146,
  334,
  1043,
  275,
  76,
  2057,
  1833,
  517,
  490,
  60,
  1930,
  380,
  488,
  83,
  1315,
  659,
  792,
  770,
  382,
  178,
  1290,
  23,
  251,
  2104,
  95,
  144,
  560,
  3009,
  1537,
  207,
  73,
  1265,
  2514,
  2528,
  421,
  846,
  1228,
  1393,
  1760,
  152,
  1802,
  635,
  1616,
  1719,
  507,
  117,
  707,
  1386,
  1112,
  166,
  55,
  529,
  1052
]
```

**code_ids_next_second：code_ids_next_second**

```json
[
  796,
  4306,
  2685,
  1725,
  2197,
  1668,
  590,
  279,
  342,
  2056,
  526,
  26,
  958,
  647,
  78,
  654,
  496,
  84,
  288,
  196,
  691,
  274,
  538,
  256,
  33,
  2414,
  57,
  2564,
  1252,
  916,
  624,
  1019,
  470,
  1931,
  378,
  347,
  2525,
  354,
  557,
  482,
  3854,
  2407,
  376,
  220,
  2890,
  349,
  820,
  876,
  550,
  1476,
  136,
  418,
  1214,
  13,
  1117,
  1615,
  113,
  377,
  1283,
  298,
  1417,
  2181,
  1710,
  321
]
```

**target_first_second：target_first_second**

```json
[
  796,
  4306,
  2685,
  1725,
  2197,
  1668,
  590,
  279,
  342,
  2056,
  526,
  26,
  958,
  647,
  78,
  654,
  496,
  84,
  288,
  196,
  691,
  274,
  538,
  256,
  33,
  2414,
  57,
  2564,
  1252,
  916,
  624,
  1019,
  470,
  1931,
  378,
  347,
  2525,
  354,
  557,
  482,
  3854,
  2407,
  376,
  220,
  2890,
  349,
  820,
  876,
  550,
  1476,
  136,
  418,
  1214,
  13,
  1117,
  1615,
  113,
  377,
  1283,
  298,
  1417,
  2181,
  1710,
  321
]
```

**ignored_target_value_before_offset：ignored_target_value_before_offset**

```json
-50305
```

**supervised_patch_count：supervised_patch_count**

```json
960
```

**logits：logits**

```json
{
  "shape": [
    1,
    1024,
    58496
  ],
  "dtype": "torch.bfloat16",
  "first_values": [
    2.453125,
    2.15625,
    -2.0,
    -2.578125,
    -1.25,
    -0.8046875,
    0.1875,
    -0.828125
  ]
}
```

### S002 留出记录的下一秒预测

```json
{
  "file": "S002R01-000.pkl",
  "EEG_loss": 8.411685943603516,
  "predicted_global_ids_first_8": [
    50356,
    53341,
    50356,
    50356,
    50356,
    53341,
    53341,
    50356
  ],
  "true_global_ids_first_8": [
    51100,
    54610,
    52989,
    52029,
    52501,
    51972,
    50894,
    50583
  ]
}
```

## tokenizer 在 S002 上实际重建了什么

这次只加载保存的权重并前向一次，未更新权重。下面的 first_values 是数组展开后的前八个数；目标和重建是标准化空间的值，不直接等于原始微伏。

```json
{
  "checkpoint": "tokenizer.pt",
  "file": "S002R01-000.pkl",
  "slurm_job": "4350049",
  "updates": 0,
  "input": {
    "shape": [
      1,
      1024,
      200
    ],
    "dtype": "torch.float32",
    "first_values": [
      1.3552527156068805e-16,
      0.27022406458854675,
      0.1455792933702469,
      0.2765379250049591,
      0.2204429656267166,
      0.08724703639745712,
      0.2380007952451706,
      0.1869584619998932
    ]
  },
  "encoder_features": {
    "shape": [
      1,
      1024,
      768
    ],
    "dtype": "torch.float32",
    "first_values": [
      -0.520768404006958,
      0.14915941655635834,
      1.8063929080963135,
      0.030724642798304558,
      -0.5426980257034302,
      0.23276139795780182,
      -1.6813749074935913,
      0.014052514918148518
    ]
  },
  "quantized_vectors": {
    "shape": [
      1,
      1024,
      128
    ],
    "dtype": "torch.float32",
    "first_values": [
      0.14565777778625488,
      -0.08777745813131332,
      0.0912557914853096,
      0.02908233553171158,
      0.1580079346895218,
      0.005046024918556213,
      0.10624250024557114,
      -0.09236916899681091
    ]
  },
  "code_ids_first_8": [
    1231,
    93,
    1173,
    1168,
    1042,
    757,
    7,
    915
  ],
  "frequency_target": {
    "shape": [
      1,
      1024,
      100
    ],
    "dtype": "torch.float32",
    "first_values": [
      5.968497276306152,
      4.310696601867676,
      2.360982894897461,
      2.8155369758605957,
      1.890008568763733,
      1.2677690982818604,
      1.3624781370162964,
      1.1408394575119019
    ]
  },
  "frequency_reconstruction": {
    "shape": [
      1,
      1024,
      100
    ],
    "dtype": "torch.bfloat16",
    "first_values": [
      3.40625,
      2.6875,
      1.5390625,
      1.1953125,
      0.79296875,
      0.5625,
      1.0625,
      1.1328125
    ]
  },
  "waveform_target": {
    "shape": [
      1,
      1024,
      200
    ],
    "dtype": "torch.float32",
    "first_values": [
      -0.03875432908535004,
      0.7141860723495483,
      0.3668813705444336,
      0.7317787408828735,
      0.575478196144104,
      0.2043469399213791,
      0.6244005560874939,
      0.4821784496307373
    ]
  },
  "waveform_reconstruction": {
    "shape": [
      1,
      1024,
      200
    ],
    "dtype": "torch.bfloat16",
    "first_values": [
      -0.205078125,
      0.062255859375,
      -0.2001953125,
      -0.1298828125,
      0.0703125,
      0.12353515625,
      0.134765625,
      -0.02490234375
    ]
  },
  "frequency_mse": 0.8577395677566528,
  "waveform_mse": 1.056148886680603,
  "quantization_loss": 0.004878392443060875
}
```

## 如何解释这次结果

四阶段都有有限损失、反向更新、保存的 checkpoint 及留出样本输出，才能说教学流程已跑通。少量更新的损失与少数窗口的准确率不能证明模型已经学会任务，也不能证明多任务或预训练优于原始模型。
文字辅助数据是显式保存的简短英文，不是 OpenWebText；EEG tokenizer 随机初始化后只训练少量步。自回归阶段的 GPT 复用作者权重，不能称为从零预训练。
