import tilelang
import tilelang.language as T


@tilelang.jit
def elementwise_add(A, B, block_M, block_N, in_dtype, out_dtype, threads):
    M, N = T.const("M, N")
    A: T.Tensor((M, N), in_dtype)
    B: T.Tensor((M, N), in_dtype)
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_N), in_dtype)
        B_shared = T.alloc_shared((block_M, block_N), in_dtype)
        C_local = T.alloc_fragment((block_M, block_N), out_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)

        T.copy(A[by * block_M, bx * block_N], A_shared)
        T.copy(B[by * block_M, bx * block_N], B_shared)
        for y, x in T.Parallel(block_M, block_N):
            C_local[y, x] = A_shared[y, x] + B_shared[y, x]
        T.copy(C_local, C_shared)
        T.copy(C_shared, C[by * block_M, bx * block_N])

    return C
