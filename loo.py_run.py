#!/usr/bin/env python3

import numpy as np
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
# loopy currently requires on pyopencl 
import loopy as lp
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)
# Set up pyopencl.Context & CommandQueue

n = 16*16
x_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32) # device side
y_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
z_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
a_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
b_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
x_vec_host = np.random.randn(n).astype(np.float32) # host side
y_vec_host = np.random.randn(n).astype(np.float32)

knl = lp.make_kernel(
     "{ [i,j,ii,jj]: 0<=i,j,ii,jj<n }",
     """
     out[j,i] = a[i,j] {id=transpose}
     out[ii,jj] = 2*out[ii,jj]  {dep=transpose}
     """)
knl = lp.prioritize_loops(knl, "i,j,ii,jj")
knl = lp.set_options(knl, "write_cl")
print(knl) # Kernel info, including loop domain, instructions and arguments

evt, (out,) = knl(queue, a=a_mat_dev) # run the kernel
assert (out.get() == a_mat_dev.get().T*2).all() # always write tests!

knl = lp.set_options(knl, write_wrapper=True, write_cl=False)
# peek at generated code
evt, (out,) = knl(queue, a=x_vec_host)

knl = lp.make_kernel(
    "{ [i]: 0<=i<n }",
    "a[i] = 0", assumptions="n>=1")
knl = lp.split_iname(knl, "i", 16) # split loop variable
knl = lp.prioritize_loops(knl, "i_outer,i_inner")
knl = lp.set_options(knl, "write_cl")
evt, (out,) = knl(queue, a=x_vec_dev)

knl = lp.make_kernel(
    "{ [i]: 0<=i<n }",
    "a[i] = a[i] * b[i] + c[i]", assumptions="n>=0 and n mod 4 = 0")
orig_knl = knl # copy kernel, test assumptions, and unrolling
knl = lp.split_iname(knl, "i", 4)
knl = lp.tag_inames(knl, dict(i_inner="unr"))
knl = lp.prioritize_loops(knl, "i_outer,i_inner")
knl = lp.set_options(knl, "write_cl")
evt, (out,) = knl(queue, a=x_vec_dev, b=y_vec_dev, c=z_vec_dev)

from warnings import resetwarnings, filterwarnings
resetwarnings() # surpress some warnings during stats
filterwarnings('ignore', category=Warning)

knl = lp.add_and_infer_dtypes(knl,
    dict(a=np.float32, b=np.float32, c=np.float32))
op_map = lp.get_op_map(knl) # get operations counting
print(lp.stringify_stats_mapping(op_map))

mem_map = lp.get_mem_access_map(knl) # get memory access(load, store) counting
print(lp.stringify_stats_mapping(mem_map))

