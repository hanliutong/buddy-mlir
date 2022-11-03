// This file is from upstream MLIR integration test.
func.func @rol(%data : i32, %bits : i32) -> i32{
  %c32 = arith.constant 32 : i32
  %offset = arith.subi %c32, %bits :i32
  %l = arith.shli %data, %bits : i32
  %r = arith.shrui %data, %offset :i32
  %res = arith.ori %l, %r : i32
  return %res :i32
}

func.func @f0(%x : i32, %y : i32, %z : i32) -> i32{
  %t1 = arith.xori %y, %z : i32
  %t2 = arith.andi %t1, %x : i32
  %res = arith.xori %t2, %z : i32
  return %res :i32
}

func.func @sum4(%a : i32, %b : i32, %c : i32, %d : i32) -> i32{
  %t1 = arith.addi %a, %b : i32
  %t2 = arith.addi %c, %d : i32
  %res = arith.addi %t1, %t2 : i32
  return %res :i32
}

func.func @sha1c(%a : i32, %b : i32, %c : i32, %d : i32, %e : i32, %t : vector<4xi32>) -> (i32, i32, i32, i32){
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %i2 = arith.constant 2 : index
  %i3 = arith.constant 3 : index
  %i5 = arith.constant 5 : i32
  %i30 = arith.constant 30 : i32

  %t0 = vector.extractelement %t[%i0:index] : vector<4xi32>
  %t1 = vector.extractelement %t[%i1:index] : vector<4xi32>
  %t2 = vector.extractelement %t[%i2:index] : vector<4xi32>
  %t3 = vector.extractelement %t[%i3:index] : vector<4xi32>

  %a30 = arith.shli %a, %i30 : i32
  %a5 = arith.shli %a, %i5 : i32
  %b30 = arith.shli %b, %i30 : i32

  %Tf0 = func.call @f0(%b, %c, %d) : (i32, i32, i32) -> i32
  %T0 = func.call @sum4(%a5, %Tf0, %e, %t0)  :(i32, i32, i32, i32) -> i32
  %T0R5 = func.call @rol(%T0, %i5)  :(i32, i32) -> i32
  %T0R30 = func.call @rol(%T0, %i30)  :(i32, i32) -> i32

  %Tf1 = func.call @f0(%a, %b30, %c) : (i32, i32, i32) -> i32
  %T1 = func.call @sum4(%T0R5, %Tf1, %d, %t1)  :(i32, i32, i32, i32) -> i32
  %T1R5 = func.call @rol(%T1, %i5)  :(i32, i32) -> i32
  %T1R30 = func.call @rol(%T1, %i30)  :(i32, i32) -> i32

  %Tf2 = func.call @f0(%T0, %a30, %b30) : (i32, i32, i32) -> i32
  %T2 = func.call @sum4(%T1R5, %Tf2, %c, %t2)  :(i32, i32, i32, i32) -> i32
  %T2R5 = func.call @rol(%T2, %i5)  :(i32, i32) -> i32

  %Tf3 = func.call @f0(%T1, %T0R30, %a30) : (i32, i32, i32) -> i32
  %T3 = func.call @sum4(%T2R5, %Tf3, %b30, %t3)  :(i32, i32, i32, i32) -> i32

  return %T3, %T2, %T1R30, %T0R30 : i32, i32, i32, i32
}

func.func @main() {
  %data = memref.alloc() : memref<1x4xi32>
  %cpu_output = memref.alloc() : memref<1x4xi32>
  %gpu_output = memref.alloc() : memref<1x4xi32>
  %cst0 = arith.constant 0 : i32
  %cst1 = arith.constant 1 : i32
  %cst2 = arith.constant 2 : i32
  %cst3 = arith.constant 3 : i32
  %cst4 = arith.constant 4 : i32
  %cst5 = arith.constant 5 : i32

  
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index

  %cast_data = memref.cast %data : memref<1x4xi32> to memref<*xi32>
  gpu.host_register %cast_data : memref<*xi32>
  %cpu_output_cast = memref.cast %cpu_output : memref<1x4xi32> to memref<*xi32>
  %gpu_output_cast = memref.cast %gpu_output : memref<1x4xi32> to memref<*xi32>
  gpu.host_register %gpu_output_cast : memref<*xi32>

  memref.store %cst1, %data[%c0, %c0] : memref<1x4xi32>
  memref.store %cst2, %data[%c0, %c1] : memref<1x4xi32>
  memref.store %cst3, %data[%c0, %c2] : memref<1x4xi32>
  memref.store %cst4, %data[%c0, %c3] : memref<1x4xi32>

  %vec = vector.load %data[%c0, %c0] : memref<1x4xi32>, vector<4xi32>

  %a_cpu, %b_cpu, %c_cpu, %d_cpu = func.call @sha1c(%cst1, %cst2, %cst3, %cst4, %cst5, %vec) :(i32, i32, i32, i32, i32, vector<4xi32>) -> (i32, i32, i32, i32)
  memref.store %a_cpu, %cpu_output[%c0, %c0] : memref<1x4xi32>
  memref.store %b_cpu, %cpu_output[%c0, %c1] : memref<1x4xi32>
  memref.store %c_cpu, %cpu_output[%c0, %c2] : memref<1x4xi32>
  memref.store %d_cpu, %cpu_output[%c0, %c3] : memref<1x4xi32>
  call @printMemrefI32(%cpu_output_cast) : (memref<*xi32>) -> ()

  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
             threads(%tx, %ty, %tz) in (%block_x = %c1, %block_y = %c1, %block_z = %c1) {
    %a, %b, %c, %d = func.call @sha1c(%cst1, %cst2, %cst3, %cst4, %cst5, %vec) :(i32, i32, i32, i32, i32, vector<4xi32>) -> (i32, i32, i32, i32)
    memref.store %a, %gpu_output[%c0, %c0] : memref<1x4xi32>
    memref.store %b, %gpu_output[%c0, %c1] : memref<1x4xi32>
    memref.store %c, %gpu_output[%c0, %c2] : memref<1x4xi32>
    memref.store %d, %gpu_output[%c0, %c3] : memref<1x4xi32>
    gpu.terminator
  }
  call @printMemrefI32(%gpu_output_cast) : (memref<*xi32>) -> ()

  return
}

func.func private @printMemrefI32(memref<*xi32>)

// func.func @sha1c_vec(%abcd: vector<4xi32>, %e : i32, %t : vector<4xi32>) -> vector<4xi32>{
//   %i0 = arith.constant 0 : index
//   %i1 = arith.constant 1 : index
//   %i2 = arith.constant 2 : index
//   %i3 = arith.constant 3 : index
//   %i5 = arith.constant 5 : i32
//   %i30 = arith.constant 30 : i32
//   %a = vector.extractelement %abcd[%i0:index] : vector<4xi32>
//   %b = vector.extractelement %abcd[%i1:index] : vector<4xi32>
//   %c = vector.extractelement %abcd[%i2:index] : vector<4xi32>
//   %d = vector.extractelement %abcd[%i3:index] : vector<4xi32>
//   %t0 = vector.extractelement %t[%i0:index] : vector<4xi32>
//   %t1 = vector.extractelement %t[%i1:index] : vector<4xi32>
//   %t2 = vector.extractelement %t[%i2:index] : vector<4xi32>
//   %t3 = vector.extractelement %t[%i3:index] : vector<4xi32>

//   %a30 = arith.shli %a, %i30 : i32
//   %a5 = arith.shli %a, %i5 : i32
//   %b30 = arith.shli %b, %i30 : i32

//   %Tf0 = func.call @f0(%b, %c, %d) : (i32, i32, i32) -> i32
//   %T0 = func.call @sum4(%a5, %Tf0, %e, %t0)  :(i32, i32, i32, i32) -> i32
//   %T0R5 = func.call @rol(%T0, %i5)  :(i32, i32) -> i32
//   %T0R30 = func.call @rol(%T0, %i30)  :(i32, i32) -> i32

//   %Tf1 = func.call @f0(%a, %b30, %c) : (i32, i32, i32) -> i32
//   %T1 = func.call @sum4(%T0R5, %Tf1, %d, %t1)  :(i32, i32, i32, i32) -> i32
//   %T1R5 = func.call @rol(%T1, %i5)  :(i32, i32) -> i32

//   %Tf2 = func.call @f0(%T0, %a30, %b30) : (i32, i32, i32) -> i32
//   %T2 = func.call @sum4(%T1R5, %Tf2, %c, %t2)  :(i32, i32, i32, i32) -> i32
//   %T2R5 = func.call @rol(%T2, %i5)  :(i32, i32) -> i32

//   %Tf3 = func.call @f0(%T1, %T0R30, %a30) : (i32, i32, i32) -> i32
//   %T3 = func.call @sum4(%T2R5, %Tf3, %b30, %t3)  :(i32, i32, i32, i32) -> i32

//   %s = arith.constant 10.1 : f32
//   %t = vector.splat %s : vector<8x16xi32>
//   %r0 = vector.insertelement %T3, %offset :i32
//   %res = arith.ori %l, %r : vector<4xi32>
//   return %res : vector<4xi32>
// }