# 代码风格

[English](../en/code-style.md)

这些约定用于保持 Python 与 TileLang 源码简洁且一致。

## 导入

使用依赖原本的名称。TileLang DSL 统一使用 `import tilelang.language as T`。
从定义它们的模块导入类和类型，通过所属模块调用函数。包初始化文件保持轻量，
并在接近使用位置时加载依赖。

## 命名与类型

可复用操作使用数学名称，模型专属名称只描述模型语义。Kernel 符号由算法名称、
workload 和 launch 参数组成。模块和函数使用 snake_case，类使用 PascalCase。

区分 workload 维度（`m`、`n`、`k`）与 tile 维度（`tile_m`、`tile_n`、
`tile_k`）。配置和结果使用带类型的记录，元数据使用不可变记录。精确标注公开
接口。必须实现的扩展点使用 `ABC` 和 `abstractmethod`。

## 命令行接口

命令行参数、选项和子命令统一使用 Typer 表达。在带类型的命令函数中完成解析与
验证，不手动读取 `sys.argv`。

## 断言

内部不变量直接写成 `assert condition, "message"`，不要展开成
`if not condition: raise ...`。由外部输入、文件、设备或其他运行时条件造成的
失败仍然使用异常。

## TileLang

每个 kernel 实现文件只保留一个 JIT 定义。执行变体使用编译期参数表达。用简洁
注释解释数值选择和硬件布局。直接表达数据流，并在实测有收益时复用已加载的操作数。
