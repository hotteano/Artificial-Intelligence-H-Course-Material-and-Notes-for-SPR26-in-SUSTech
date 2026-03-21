# Project 1 命令行格式汇总

本文档列出所有符合项目要求的命令行格式。

---

## 1. Evaluator 评估器

### 命令格式
```bash
python Evaluator.py -n <social_network> -i <initial_seed_set> -b <balanced_seed_set> -k <budget> -o <object_value_output_path>
```

### 参数说明
| 参数 | 说明 |
|-----|------|
| `-n` | 社交网络文件路径 |
| `-i` | 初始种子集文件路径 |
| `-b` | 平衡种子集文件路径 (Evaluator 读取) |
| `-k` | 预算 (正整数) |
| `-o` | 目标函数值输出文件路径 |

### 示例
```powershell
Set-Location "Project1\Project1\Project1\Testcase\Evaluator"
python Evaluator.py `
    -n "..\Evolutionary\map1\dataset1" `
    -i "..\Evolutionary\map1\seed" `
    -b "..\Evolutionary\map1\seed_balanced_evol" `
    -k 10 `
    -o ".\eval_result.txt"
```

---

## 2. 启发式算法 IEMP_Heur.py

### 命令格式
```bash
python IEMP_Heur.py -n <social_network> -i <initial_seed_set> -b <balanced_seed_set> -k <budget>
```

### 参数说明
| 参数 | 说明 |
|-----|------|
| `-n` | 社交网络文件路径 |
| `-i` | 初始种子集文件路径 |
| `-b` | 平衡种子集文件路径 (算法输出) |
| `-k` | 预算 (正整数) |

### 示例
```powershell
Set-Location "Project1\Project1\Project1\Testcase\Heuristic"
python IEMP_Heur.py `
    -n ".\map1\dataset1" `
    -i ".\map1\seed" `
    -b ".\map1\seed_balanced_heur" `
    -k 10
```

---

## 3. 进化算法 IEMP_Evol.py

### 命令格式
```bash
python IEMP_Evol.py -n <social_network> -i <initial_seed_set> -b <balanced_seed_set> -k <budget>
```

### 参数说明
| 参数 | 说明 |
|-----|------|
| `-n` | 社交网络文件路径 |
| `-i` | 初始种子集文件路径 |
| `-b` | 平衡种子集文件路径 (算法输出) |
| `-k` | 预算 (正整数) |

### 可选参数
| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--pop-size` | 40 | 种群大小 |
| `--generations` | 80 | 最大进化代数 |
| `--seed` | None | 随机种子 |

### 示例
```powershell
Set-Location "Project1\Project1\Project1\Testcase\Evolutionary"
python IEMP_Evol.py `
    -n ".\map1\dataset1" `
    -i ".\map1\seed" `
    -b ".\map1\seed_balanced_evol" `
    -k 10
```

---

## 4. 文件结构

```
Testcase/
├── Evaluator/
│   ├── Evaluator.py          # 评估器 (已符合要求)
│   ├── map1/
│   └── map2/
├── Heuristic/
│   ├── IEMP_Heur.py          # 启发式算法 (已重命名)
│   ├── map1/
│   ├── map2/
│   └── map3/
├── Evolutionary/
│   ├── IEMP_Evol.py          # 进化算法 (已重命名)
│   ├── README.md             # 详细文档
│   ├── map1/
│   ├── map2/
│   └── map3/
└── COMMANDS.md               # 本文件
```

---

## 5. 测试流程

### 完整测试流程

```powershell
# 1. 运行启发式算法
Set-Location "Project1\Project1\Project1\Testcase\Heuristic"
python IEMP_Heur.py -n ".\map1\dataset1" -i ".\map1\seed" -b ".\map1\seed_balanced_heur" -k 10

# 2. 运行进化算法
Set-Location "Project1\Project1\Project1\Testcase\Evolutionary"
python IEMP_Evol.py -n ".\map1\dataset1" -i ".\map1\seed" -b ".\map1\seed_balanced_evol" -k 10

# 3. 评估启发式算法结果
Set-Location "Project1\Project1\Project1\Testcase\Evaluator"
python Evaluator.py `
    -n "..\Heuristic\map1\dataset1" `
    -i "..\Heuristic\map1\seed" `
    -b "..\Heuristic\map1\seed_balanced_heur" `
    -k 10 `
    -o ".\eval_heur.txt"

# 4. 评估进化算法结果
python Evaluator.py `
    -n "..\Evolutionary\map1\dataset1" `
    -i "..\Evolutionary\map1\seed" `
    -b "..\Evolutionary\map1\seed_balanced_evol" `
    -k 10 `
    -o ".\eval_evol.txt"
```

---

## 6. 注意事项

1. **文件命名**：确保文件名完全匹配
   - `IEMP_Heur.py` (H大写)
   - `IEMP_Evol.py` (E大写)
   - `Evaluator.py` (E大写)

2. **参数格式**：使用短格式 `-n`, `-i`, `-b`, `-k`

3. **路径格式**：Windows 使用反斜杠 `\`，Linux/Mac 使用正斜杠 `/`

4. **输出格式**：平衡种子集文件格式
   ```
   |S1| |S2|
   (S1 nodes, one per line)
   (S2 nodes, one per line)
   ```
