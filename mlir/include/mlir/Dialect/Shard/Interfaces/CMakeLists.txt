set(LLVM_TARGET_DEFINITIONS ShardingInterface.td)
mlir_tablegen(ShardingInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ShardingInterface.cpp.inc -gen-op-interface-defs)
add_mlir_dialect_tablegen_target(MLIRShardingInterfaceIncGen)

# ShardingInterface.h includes the generated Shard operation declarations.
target_link_libraries(MLIRShardingInterfaceIncGen INTERFACE
  MLIRShardIncGen)
