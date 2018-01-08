# 如何使用Loo.py

## 安装 Loo.py 

为便于测试使用，安装基于 pyopencl 的 Loo.py。安装过程包括创建虚拟环境和安装相应的包两部分。采用Anaconda或者Miniconda完成安装较为简便：

```
conda install git pip pocl islpy pyopencl
```

为便于运行以及观察测试结果，建议在jyputer notebook中运行。

注意需要在jupyter notebook中配置虚拟环境：

```
conda create -n loopydev
source activate loopydev
```

使用loo.py只需：

```python
import loopy as lp
```

## 基本操作

Loo.py的操作与OpenCL存在较多相似之处，因此本节会使用一些OpenCL中的概念进行介绍，以便于理解loo.py的操作实现。

### 声明

声明内核至少需要包含两部分：循环域和指令，分别为第一个和第二个参数

```python
knl = lp.make_kernel(
     "{ [i]: 0<=i<n }",
     "out[i] = a[i]*a[i]")
```

循环域指明了循环变量的取值范围，在此例中仅有`i`一个循环变量。事实上，loo.py的优化很大一部分基于对循环变量进行操作，因此循环变量可附加不同的标识以进行循环展开、依赖声明以及分解为子变量等操作。具体的用法将在[相关标识](#循环变量相关标识)部分介绍。

指令部分包含循环体内部执行的操作，一般为标量赋值的形式。其中指令之间的依赖是loo.py优化的重点。指令之间的依赖会降低循环并行度，但缺少依赖则会影响计算结果的正确性。

### 内核信息

通过打印内核可以查看其具体信息：

```
---------------------------------------------------------------------------
KERNEL: loopy_kernel
---------------------------------------------------------------------------
ARGUMENTS:
a: GlobalArg, type: <runtime>, shape: (n), dim_tags: (N0:stride:1)
n: ValueArg, type: <runtime>
out: GlobalArg, type: <runtime>, shape: (n), dim_tags: (N0:stride:1)
---------------------------------------------------------------------------
DOMAINS:
[n] -> { [i] : 0 <= i < n }
---------------------------------------------------------------------------
INAME IMPLEMENTATION TAGS:
i: None
---------------------------------------------------------------------------
INSTRUCTIONS:
for i
  out[i] = a[i]*a[i]  {id=insn}
end i
---------------------------------------------------------------------------
```

其中`Arguments`为参数，包括输入和输出的变量，以引用形式传递。注意此时的变量是`GlobalArg`，即OpenCL设备的全局变量，这将会在后续的性能优化中详细说明。`Domains`为循环域，当额外声明了标识时也会一并显示。`iname`即为循环变量，参见[循环变量相关标识](#循环变量相关标识)部分。`Instructions`包含声明的指令，每条指令均有`id`，用于计算依赖关系。

### 运行内核与代码生成

在loo.py内可直接运行声明的内核，并与numpy的结果对比：

```python
x_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
evt, (out,) = knl(queue, a=x_vec_dev)
assert (out.get() == (x_vec_dev*x_vec_dev).get()).all()
```

设置`write_cl`参数则可以打印出生成的OpenCL代码：

```c
#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(__global float const *__restrict__ a, int const n, __global float *__restrict__ out)
{
  for (int i = 0; i <= -1 + n; ++i)
    out[i] = a[i] * a[i];
}
```

上述代码中可以看出，参数`x_vec_dev`的属性被用于生成目标代码。比如`float`类型是由参数的类型`float32`推断得到，且`n`是由参数的shape推断得到。并且，对于每种参数类型均需要重新生成一个内核对应的目标代码，与C++中的模板类似，相较于Haskell不够优雅。

而设置`write_wrapper`参数可以打印调用内核的Python宿主代码：
```python
def invoke_loopy_kernel_loopy_kernel(_lpy_cl_kernels, queue, allocator=None, wait_for=None, out_host=None, a=None, n=None, out=None):
    if allocator is None:
        allocator = _lpy_cl_tools.DeferredAllocator(queue.context)

    # {{{ find integer arguments from shapes

    if n is None:
        if a is not None:
            n = a.shape[0]
        elif out is not None:
            n = out.shape[0]
    ···
    _lpy_evt = _lpy_host_loopy_kernel(_lpy_cl_kernels, queue, a.base_data, n, out.base_data, wait_for=wait_for)
    if out_host is None and (_lpy_encountered_numpy and not _lpy_encountered_dev):
        out_host = True
    if out_host:
        pass
        out = out.get(queue=queue)

    return _lpy_evt, (out,)

```

可通过检查上述生成的代码是否与预期相符进行调试分析。

## 循环依赖

### 指令依赖关系

Loo.py的编程模型默认为完全乱序，因此若采用串行的编程思维会导致loo.py生成错误的代码。例如在以下错误示范中：

```python
knl = lp.make_kernel(
     "{ [i,j]: 0<=i,j<n }",
     """
     out[j,i] = a[i,j] {id=transpose}
     out[i,j] = 2*out[i,j]  {dep=transpose}
     """)
```

其中第二条指令（乘2）依赖第一条指令（转置）的结果，因此显然需要声明二者的依赖关系，否则按照loo.py的编程模型第二条指令可以在第一条指令前执行。loo.py会在上述存在数据冲突（写后读）时自动生成依赖，但并不足以覆盖所有情况。由于第二条指令需要在第一条指令全部执行完成（两层循环均结束）后执行，因此第二条指令不应与第一条使用相同的循环变量，而应声明使用不同的循环变量：

```python
knl = lp.make_kernel(
     "{ [i,j,ii,jj]: 0<=i,j,ii,jj<n }",
     """
     out[j,i] = a[i,j] {id=transpose}
     out[ii,jj] = 2*out[ii,jj]  {dep=transpose}
     """)
```

从而得到正确的循环依赖关系。

### 多重循环

由于loo.py的编程模型默认为完全乱序，多重循环内下标不存在依赖关系时可以有多种不同的循环序。而生成目标代码时可以指定loo.py按照特定的下标顺序执行循环：

```python
knl = lp.prioritize_loops(knl, "j,i")
```

对应的目标代码：

```c
for (int j = 0; j <= -1 + n; ++j)
    for (int i = 0; i <= -1 + n; ++i)
        ···
```


## 内核变换

Loo.py的声明方式常用于基于多面体模型产生内核，然后进行内核变换增加约束条件，最后得到正确的目标代码，或者提高代码性能。因此，内核变换的参数是loo.py的主要应用之一。例如上述例子中采用了指定循环下标的内核变换，得到的新内核生成的目标代码就反映了内核变换的结果。

### 循环变量相关标识

常用的内核变换还包括下标分解和参数假设：

```python
knl = lp.make_kernel(
     "{ [i]: 0<=i<n }",
     "a[i] = 0", assumptions="n>=1")
knl = lp.split_iname(knl, "i", 16)
knl = lp.prioritize_loops(knl, "i_outer,i_inner")
knl = lp.set_options(knl, "write_cl")
evt, (out,) = knl(queue, a=x_vec_dev)
```

通过将循环下标分解为`_outer, _inner`，可以将原始的循环细分，从而提高并行度；对于参数的假设则有助于减少不必要的判断，比如此例中就可节约边界情况的判断。

```c
#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) loopy_kernel(__global float *__restrict__ a, int const n)
{
  for (int i_outer = 0; i_outer <= -1 + ((15 + n) / 16); ++i_outer)
    for (int i_inner = 0; i_inner <= (-16 + n + -16 * i_outer >= 0 ? 15 : -1 + n + -16 * i_outer); ++i_inner)
      a[16 * i_outer + i_inner] = 0.0f;
}
```

当然，上述内核变换都不会影响结果，可以分析得知分解下标后二层循环结果与原先一致。

### 循环展开

在上述参数假设中，可以假设参数为4的倍数，从而将循环展开4次：
```c
  for (int i_outer = 0; i_outer <= int_floor_div_pos_b(-4 + n, 4); ++i_outer)
  {
    a[4 * i_outer] = 0.0f;
    a[4 * i_outer + 1] = 0.0f;
    a[4 * i_outer + 2] = 0.0f;
    a[4 * i_outer + 3] = 0.0f;
  }
```
基于OpenCL的异构平台，可以充分利用指令多发射或者设备的共享内存，提升执行效率。

### 循环并行

```python
knl = lp.make_kernel(
    "{ [i]: 0<=i<n }",
    "a[i] = 0", assumptions="n>=0")
knl = lp.split_iname(knl, "i", 128,
        outer_tag="g.0", inner_tag="l.0")
knl = lp.set_options(knl, "write_cl")
evt, (out,) = knl(queue, a=x_vec_dev)
```

上述代码片段不仅进行了下标分解，还将局部标签分配给了内层循环，将工作组标签分配给了外层循环下标，工作组的大小为参数128，进行OpenCL的并行执行。

## 存储

### 临时变量

此前提到的的内核均为赋值指令，而部分中间结果实际上并不需要作为最终结果输出。因此，如果能够将中间结果存储在临时位置（例如寄存器）将可以提高性能。

```python
knl = lp.make_kernel(
    "{ [i]: 0<=i<n }",
    """
    <float32> a_temp = sin(a[i])
    out1[i] = a_temp {id=out1}
    out2[i] = sqrt(1-a_temp*a_temp) {dep=out1}
    """)
```

其中`a_temp`即被标记为临时变量，其shape以及类型均可由上下文推断得到，因此`<>`中类型可以省略。

### 预取

与临时变量相关，若部分数据需要访问，则提前预取可减少访问时间。而预取的内容也可能用于临时变量。

```python
knl = lp.make_kernel(
    "{ [i_outer,i_inner, k]:  "
         "0<= 16*i_outer + i_inner <n and 0<= i_inner,k <16}",
    """
    out[16*i_outer + i_inner] = sum(k, a[16*i_outer + i_inner])
    """)
knl = lp.tag_inames(knl, dict(i_outer="g.0", i_inner="l.0"))
knl = lp.set_options(knl, "write_cl")
knl_pf = lp.add_prefetch(knl, "a", ["i_inner"])
evt, (out,) = knl_pf(queue, a=x_vec_dev)
```

例如上述例子加入预取后则会对下标细分后的`a`进行预取，并且加入[屏障](#屏障)：

```c
  if (-1 + -16 * gid(0) + -1 * lid(0) + n >= 0)
    acc_k = 0.0f;
  if (-1 + -16 * gid(0) + -1 * lid(0) + n >= 0)
    a_fetch[lid(0)] = a[16 * gid(0) + lid(0)];
  barrier(CLK_LOCAL_MEM_FENCE) /* for a_fetch (insn_k_update depends on a_fetch_rule) */;
  if (-1 + -16 * gid(0) + -1 * lid(0) + n >= 0)
  {
    for (int k = 0; k <= 15; ++k)
      acc_k = acc_k + a_fetch[lid(0)];
    out[16 * gid(0) + lid(0)] = acc_k;
  }
```

## 同步

### 屏障

在代码生成前，loo.py会检查是否存在内存访问冲突（写后写，写后读，读后写），如果存在则会自动生成插入屏障指令。

在局部内存之间的访问冲突主要通过屏障解决，而全局内存如果存在访问冲突且存在的依赖不能避免冲突，loo.py会通过报错的方式要求插入屏障。

### 原子操作

Loo.py支持原子操作，通过声明变量以及指令的原子标签即可生成相应效果的目标代码。

## 性能分析

## 相关链接

[GitHub repository](https://github.com/01-Loopy/loo.py-intro)

[项目简介、分工以及进展记录](https://github.com/01-Loopy/2017fall-student-teamworks/blob/master/01-loopy.md)