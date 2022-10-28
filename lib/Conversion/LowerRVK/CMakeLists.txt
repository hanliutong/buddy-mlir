add_mlir_library(LowerRVKPass
  LowerRVKPass.cpp

  LINK_LIBS PUBLIC
  BuddyRVKTransforms
  )
