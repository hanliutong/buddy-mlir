func.func @rvk_aes64esm() -> i64 {
  %sew = arith.constant 32 : i64
  %lmul = arith.constant 2 : i64
  // %vl = "rvk.intr.aes64esm" %avl, %sew, %lmul : index
  %vl = "rvk.intr.aes64esm"(%sew, %lmul) : (i64, i64)-> i64
  return %vl : i64
}
