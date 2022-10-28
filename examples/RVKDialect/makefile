#!/bin/bash
BUDDY_OPT := ../../build/bin/buddy-opt
BUDDY_TRANSLATE := ../../build/bin/buddy-translate
LLC := ../../llvm/build/bin/llc
OPT_FLAG := -O0

rvk-setvl-lower:
	@${BUDDY_OPT} ./rvk.mlir \
		--lower-rvk --convert-func-to-llvm \
		--reconcile-unrealized-casts \
		-o ./log.mlir

rvk-setvl-translate:
	@${BUDDY_OPT} ./rvk.mlir \
		--lower-rvk --convert-func-to-llvm \
		--reconcile-unrealized-casts | \
	${BUDDY_TRANSLATE} --buddy-to-llvmir -o log.ll
