
# colab报错解决方法

## 命令
~~~
!pip install -q tflite-model-maker
~~~
报错：
~~~
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.7/10.7 MB 50.4 MB/s eta 0:00:00
  Installing build dependencies ... done
  error: subprocess-exited-with-error
  
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> See above for output.
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  Getting requirements to build wheel ... error
error: subprocess-exited-with-error

× Getting requirements to build wheel did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.
~~~

解决方法：https://github.com/abetlen/llama-cpp-python/issues/1366
~~~
!pip install scikit-build-core==0.9.0
!CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=61" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.62 --force-reinstall --upgrade --no-cache-dir --verbose --no-build-isolation
~~~
也可尝试
~~~
!sudo apt update
!sudo apt install -y build-essential python3-dev
~~~

看到这个信息，说明方案一没有成功。问题在于：**在 Colab 中，`source` 命令无法在 notebook 单元格中永久改变环境**。当我们执行 `!source py310_env/bin/activate` 时，它只在那一个单元格的命令行会话中生效，下一个单元格又回到了默认的 Python 3.12 环境。

让我们用**更适合 Colab 的方法**来切换 Python 版本。

### 在 Colab 中正确降级 Python 版本的方案

#### 方案一：使用 `update-alternatives` 切换系统默认 Python（推荐）

这个方法会临时改变整个运行时环境的 Python 版本。

```python
# 安装 Python 3.10
!sudo apt update
!sudo apt install -y python3.10 python3.10-dev python3.10-venv python3.10-distutils

# 设置 Python 3.10 为默认版本
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
!sudo update-alternatives --set python3 /usr/bin/python3.10

# 检查当前 Python 版本
!python3 --version

# 重新链接 pip 到 Python 3.10
!curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
!python3 -m pip install --upgrade pip

# 现在安装 tflite-model-maker
!python3 -m pip install -q tflite-model-maker

print("安装完成！")
```

#### 方案二：使用 `virtualenv` 并显式指定 Python 解释器

```python
# 安装 Python 3.10
!sudo apt update
!sudo apt install -y python3.10 python3.10-dev python3.10-venv

# 创建虚拟环境
!python3.10 -m venv py310_env

# 直接使用虚拟环境中的 Python 和 pip 来安装
!/content/py310_env/bin/pip install -q tflite-model-maker

print("安装完成！使用以下命令测试：")
print("! /content/py310_env/bin/python -c 'import tflite_model_maker; print(\"导入成功\")'")
```

#### 方案三：使用 `conda` 创建 Python 3.10 环境

```python
# 初始化 conda（如果尚未初始化）
!conda init bash --quiet

# 创建并激活 Python 3.10 环境
!conda create -n py310_env python=3.10 -y --quiet
!source /usr/local/etc/profile.d/conda.sh && conda activate py310_env

# 在 conda 环境中安装
!conda install -c conda-forge tflite-model-maker -y --quiet

print("安装完成！")
```

### 如果上述方法都太复杂，试试这个最简单的方法：

#### 方案四：直接安装预编译的依赖（快速尝试）

```python
# 先尝试安装一个可能兼容的 numpy 版本
!pip install -q "numpy>=1.24.0"

# 然后使用 --no-deps 跳过依赖检查，直接安装 tflite-model-maker
!pip install -q --no-deps tflite-model-maker

# 手动安装它可能需要的其他依赖
!pip install -q "tf-models-official>=2.3.0"
!pip install -q "tensorflow>=2.13.0"

print("尝试性安装完成，可能需要额外调试")
```

### 验证安装是否成功

无论使用哪种方法，最后请运行以下代码来验证：

```python
try:
    import tflite_model_maker
    print("✅ tflite_model_maker 导入成功！")
    print(f"版本: {tflite_model_maker.__version__}")
except Exception as e:
    print(f"❌ 导入失败: {e}")
```

**请先尝试方案一**，它是最直接的方法。如果还有问题，Colab 重启运行时后会自动恢复默认设置，不会造成永久影响。


ERROR: Cannot install tflite-model-maker==0.1.2, tflite-model-maker==0.2.0, tflite-model-maker==0.2.1, tflite-model-maker==0.2.2, tflite-model-maker==0.2.3, tflite-model-maker==0.3.3, tflite-model-maker==0.3.4, tflite-model-maker==0.4.0, tflite-model-maker==0.4.1, tflite-model-maker==0.4.2 and tflite-model-maker==0.4.3 because these package versions have conflicting dependencies.
ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts
安装完成！


这个错误信息非常明确！问题不再是 Python 版本或编译工具，而是 **`tflite-model-maker` 及其依赖包之间存在版本冲突**。

错误显示所有版本的 `tflite-model-maker` (从 0.1.2 到 0.4.3) 都有冲突的依赖关系。这是因为 Colab 预装的环境中有一些较新版本的包（如 TensorFlow、NumPy 等），与 `tflite-model-maker` 要求的旧版本不兼容。

### 解决方案

我们需要**创建一个隔离的虚拟环境**，并在其中安装一个兼容的 TensorFlow 版本，然后再安装 `tflite-model-maker`。

#### 最佳方案：使用虚拟环境安装兼容版本

在 Colab 单元格中执行以下代码：

```python
# 1. 安装 Python 3.9（与 tflite-model-maker 兼容性更好）
!sudo apt update
!sudo apt install -y python3.9 python3.9-venv python3.9-dev

# 2. 创建并切换到虚拟环境
!python3.9 -m venv tflite_env
# 在 Colab 中需要使用完整的路径来调用虚拟环境中的 pip
!/content/tflite_env/bin/pip install --upgrade pip

# 3. 先安装一个兼容的 TensorFlow 版本
!/content/tflite_env/bin/pip install "tensorflow==2.9.0"

# 4. 然后安装 tflite-model-maker
!/content/tflite_env/bin/pip install "tflite-model-maker==0.4.3"

print("安装完成！")
```

#### 验证安装

创建一个新的单元格来测试安装是否成功：

```python
# 使用虚拟环境中的 Python 来测试
!/content/tflite_env/bin/python -c "
import tflite_model_maker
print('✅ tflite_model_maker 导入成功')
print('版本:', tflite_model_maker.__version__)

import tensorflow as tf
print('TensorFlow 版本:', tf.__version__)

import numpy as np
print('NumPy 版本:', np.__version__)
"
```

### 备选方案：如果上述方法仍然失败

如果还是有问题，可以尝试使用 `conda` 来管理环境：

```python
# 使用 conda 创建隔离环境
!conda create -n tflite_env python=3.9 -y
# 初始化 conda
import sys
!conda run -n tflite_env pip install "tensorflow==2.9.0" "tflite-model-maker==0.4.3"

print("使用以下命令测试: !conda run -n tflite_env python -c 'import tflite_model_maker'")
```

### 临时解决方案：使用 `--use-deprecated=legacy-resolver`

如果不想创建虚拟环境，可以尝试使用旧的依赖解析器：

```python
# 先安装编译工具
!sudo apt update
!sudo apt install -y build-essential python3-dev

# 使用旧版解析器强制安装
!pip install --use-deprecated=legacy-resolver "tflite-model-maker==0.4.3"

print("尝试完成，可能仍需要额外依赖")
```

### 总结

1. **主要问题**：依赖版本冲突，不是编译问题
2. **最佳解决方案**：创建虚拟环境，安装兼容的 TensorFlow 2.9.0，再安装 tflite-model-maker
3. **推荐使用**：上面的虚拟环境方案，这是最可靠的方法

请先尝试**虚拟环境方案**，这应该能解决您的依赖冲突问题。