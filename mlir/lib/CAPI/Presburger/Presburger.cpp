#include "mlir/CAPI/Presburger.h"
#include "mlir-c/Presburger.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::presburger;

//===----------------------------------------------------------------------===//
// IntegerRelation creation/destruction and basic metadata operations
//===----------------------------------------------------------------------===//

/// Constructs a relation reserving memory for the specified number
/// of constraints and variables.
MlirPresburgerIntegerRelation
mlirPresburgerIntegerRelationCreate(unsigned numReservedInequalities,
                                    unsigned numReservedEqualities,
                                    unsigned numReservedCols) {
  auto space = PresburgerSpace::getRelationSpace();
  IntegerRelation *relation = new IntegerRelation(
      numReservedInequalities, numReservedEqualities, numReservedCols, space);
  return wrap(relation);
}

/// Constructs an IntegerRelation from a packed 2D matrix of tableau
/// coefficients in row-major order. The first `numDomainVars` columns are
/// considered domain and the remaining `numRangeVars` columns are domain
/// variables.
MlirPresburgerIntegerRelation
mlirPresburgerIntegerRelationCreateFromCoefficients(
    const int64_t *inequalityCoefficients, unsigned numInequalities,
    const int64_t *equalityCoefficients, unsigned numEqualities,
    unsigned numDomainVars, unsigned numRangeVars,
    unsigned numExtraReservedInequalities, unsigned numExtraReservedEqualities,
    unsigned numExtraReservedCols) {
  auto space = PresburgerSpace::getRelationSpace(numDomainVars, numRangeVars);

  IntegerRelation *relation =
      new IntegerRelation(numInequalities + numExtraReservedInequalities,
                          numEqualities + numExtraReservedInequalities,
                          numDomainVars + numRangeVars + 1, space);
  unsigned numCols = numRangeVars + numDomainVars + 1;
  for (const int64_t *rowPtr = inequalityCoefficients;
       rowPtr < inequalityCoefficients + numCols * numInequalities;
       rowPtr += numCols) {
    llvm::ArrayRef<int64_t> coef(rowPtr, rowPtr + numCols);
    relation->addInequality(coef);
  }

  for (const int64_t *rowPtr = equalityCoefficients;
       rowPtr < equalityCoefficients + numCols * numEqualities;
       rowPtr += numCols) {
    llvm::ArrayRef<int64_t> coef(rowPtr, rowPtr + numCols);
    relation->addEquality(coef);
  }
  llvm::dbgs() << "created relation @ " << relation << "\n";
  return wrap(relation);
}

/// Destroys an IntegerRelation.
void mlirPresburgerIntegerRelationDestroy(
    MlirPresburgerIntegerRelation relation) {
  if (relation.ptr)
    delete reinterpret_cast<IntegerRelation *>(relation.ptr);
}

/// Returns the number of inequalities and equalities.
unsigned mlirPresburgerIntegerRelationNumConstraints(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of inequality constraints.
unsigned mlirPresburgerIntegerRelationNumInequalities(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of equality constraints.
unsigned mlirPresburgerIntegerRelationNumEqualities(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of columns classified as domain variables.
unsigned mlirPresburgerIntegerRelationNumDomainVars(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of columns classified as range variables.
unsigned mlirPresburgerIntegerRelationNumRangeVars(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of columns classified as symbol variables.
unsigned mlirPresburgerIntegerRelationNumSymbolVars(
    MlirPresburgerIntegerRelation relation);

/// Returns the number of columns classified as local variables.
unsigned mlirPresburgerIntegerRelationNumLocalVars(
    MlirPresburgerIntegerRelation relation);

/// Returns the total number of columns in the tableau.
unsigned
mlirPresburgerIntegerRelationNumCols(MlirPresburgerIntegerRelation relation) {
  return unwrap(relation)->getNumCols();
}

/// Return the VarKind of the var at the specified position.
MlirPresburgerVariableKind
mlirPresburgerIntegerRelationGetVarKindAt(unsigned pos);

void mlirPresburgerIntegerRelationDump(MlirPresburgerIntegerRelation relation) {
  unwrap(relation)->dump();
}

//===----------------------------------------------------------------------===//
// IntegerRelation binary operations
//===----------------------------------------------------------------------===//

/// Return whether `lhs` and `rhs` are equal. This is integer-exact
/// and somewhat expensive, since it uses the integer emptiness check
/// (see IntegerRelation::findIntegerSample()).
bool mlirPresburgerIntegerRelationIsEqual(MlirPresburgerIntegerRelation lhs,
                                          MlirPresburgerIntegerRelation rhs);

/// Return the intersection of the two relations.
/// If there are locals, they will be merged.
MlirPresburgerIntegerRelation
mlirPresburgerIntegerRelationIntersect(MlirPresburgerIntegerRelation lhs,
                                       MlirPresburgerIntegerRelation rhs);

//===----------------------------------------------------------------------===//
// IntegerRelation Tableau Inspection and Manipulation
//===----------------------------------------------------------------------===//

/// Returns the value at the specified equality row and column.
MlirPresburgerDynamicAPInt mlirPresburgerIntegerRelationAtEq(unsigned i,
                                                             unsigned j);

/// The same, but casts to int64_t. This is unsafe and will assert-fail if the
/// value does not fit in an int64_t.
int64_t
mlirPresburgerIntegerRelationAtEq64(MlirPresburgerIntegerRelation relation,
                                    unsigned row, unsigned col) {
  return unwrap(relation)->atEq64(row, col);
}

/// Returns the value at the specified inequality row and column.
MlirPresburgerDynamicAPInt
mlirPresburgerIntegerRelationAtIneq(MlirPresburgerIntegerRelation relation,
                                    unsigned row, unsigned col);

/// Returns the value at the specified inequality row and column.
int64_t
mlirPresburgerIntegerRelationAtIneq64(MlirPresburgerIntegerRelation relation,
                                      unsigned row, unsigned col) {
  return unwrap(relation)->atIneq64(row, col);
}

/// Adds an equality with the given coefficients.
void mlirPresburgerIntegerRelationAddEquality(const int64_t *coefficients,
                                              size_t coefficientsSize);

/// Adds an inequality with the given coefficients.
void mlirPresburgerIntegerRelationAddInequality(const int64_t *coefficients,
                                                size_t coefficientsSize);