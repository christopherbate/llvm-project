#include "mlir-c/Presburger.h"
#include "PybindUtils.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <pybind11/attr.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

static bool isSignedIntegerFormat(std::string_view format) {
  if (format.empty())
    return false;
  char code = format[0];
  return code == 'i' || code == 'b' || code == 'h' || code == 'l' ||
         code == 'q';
}

namespace {
struct PyPresburgerIntegerRelation {
  PyPresburgerIntegerRelation(MlirPresburgerIntegerRelation relation)
      : relation(relation) {}

  PyPresburgerIntegerRelation(PyPresburgerIntegerRelation &&other) noexcept
      : relation(other.relation) {
    other.relation.ptr = nullptr;
  }

  ~PyPresburgerIntegerRelation() {
    if (relation.ptr) {
      mlirPresburgerIntegerRelationDestroy(relation);
      relation.ptr = {nullptr};
    }
  }

  static PyPresburgerIntegerRelation
  getFromBuffers(py::buffer inequalitiesCoefficients,
                 py::buffer equalityCoefficients, unsigned numDomainVars,
                 unsigned numRangeVars);

  py::object getCapsule();

  MlirPresburgerIntegerRelation relation{nullptr};
};

/// A utility that enables accessing/modifying the underlying coefficients
/// easier.
struct PyPresburgerTableau {
  enum class Kind { Equalities, Inequalities };

  PyPresburgerTableau(MlirPresburgerIntegerRelation relation, Kind kind)
      : relation(relation), kind(kind) {}

  static void bind(py::module &module);

  int64_t at64(int64_t row, int64_t col) const {
    if (kind == Kind::Equalities)
      return mlirPresburgerIntegerRelationAtEq64(relation, row, col);
    return mlirPresburgerIntegerRelationAtIneq64(relation, row, col);
  }

  MlirPresburgerIntegerRelation relation;
  Kind kind;
};
} // namespace

PyPresburgerIntegerRelation PyPresburgerIntegerRelation::getFromBuffers(
    py::buffer inequalitiesCoefficients, py::buffer equalityCoefficients,
    unsigned numDomainVars, unsigned numRangeVars) {
  // Request a contiguous view. In exotic cases, this will cause a copy.
  int flags = PyBUF_ND;
  flags |= PyBUF_FORMAT;

  // Get the view of the inequality coefficients.
  std::unique_ptr<Py_buffer> ineqView = std::make_unique<Py_buffer>();
  if (PyObject_GetBuffer(inequalitiesCoefficients.ptr(), ineqView.get(),
                         flags) != 0)
    throw py::error_already_set();

  auto freeIneqBuffer = llvm::make_scope_exit([&]() {
    if (ineqView)
      PyBuffer_Release(ineqView.get());
  });
  if (!PyBuffer_IsContiguous(ineqView.get(), 'A'))
    throw std::invalid_argument("Contiguous buffer is required.");

  if (!isSignedIntegerFormat(ineqView->format) || ineqView->itemsize != 8)
    throw std::invalid_argument(
        std::string("IntegerRelation can only be created from a buffer of "
                    "i64 values but got buffer with format: ") +
        std::string(ineqView->format));

  if (ineqView->ndim != 2)
    throw std::invalid_argument(
        std::string("expected 2d inequality coefficients but got rank ") +
        std::to_string(ineqView->ndim));

  unsigned numInequalities = ineqView->shape[0];

  // Get the view of the eequality coefficients.
  std::unique_ptr<Py_buffer> eqView = std::make_unique<Py_buffer>();
  if (PyObject_GetBuffer(equalityCoefficients.ptr(), eqView.get(), flags) != 0)
    throw py::error_already_set();
  auto freeEqBuffer = llvm::make_scope_exit([&]() {
    if (eqView)
      PyBuffer_Release(eqView.get());
  });
  if (!PyBuffer_IsContiguous(eqView.get(), 'A'))
    throw std::invalid_argument("Contiguous buffer is required.");
  if (!isSignedIntegerFormat(eqView->format) || eqView->itemsize != 8)
    throw std::invalid_argument(
        std::string("IntegerRelation can only be created from a buffer of "
                    "i64 values but got buffer with format: ") +
        std::string(eqView->format));
  if (eqView->ndim != 2)
    throw std::invalid_argument(
        std::string("expected 2d equality coefficients but got rank ") +
        std::to_string(eqView->ndim));

  unsigned numEqualities = eqView->shape[0];

  if (eqView->shape[1] != numDomainVars + numRangeVars + 1 ||
      eqView->shape[1] != ineqView->shape[1])
    throw std::invalid_argument(
        "expected number of columns of inequality and equality coefficient "
        "matrices to equal numRangeVars + numDomainVars + 1");

  MlirPresburgerIntegerRelation relation =
      mlirPresburgerIntegerRelationCreateFromCoefficients(
          reinterpret_cast<const int64_t *>(ineqView->buf), numInequalities,
          reinterpret_cast<const int64_t *>(eqView->buf), numEqualities,
          numDomainVars, numRangeVars);

  return PyPresburgerIntegerRelation(relation);
}

py::object PyPresburgerIntegerRelation::getCapsule() {
  throw std::invalid_argument("unimplemented");
}

void PyPresburgerTableau::bind(py::module &m) {
  py::class_<PyPresburgerTableau>(m, "IntegerRelationTableau",
                                  py::module_local())
      .def("__getitem__", [](PyPresburgerTableau &tableau, int64_t row,
                             int64_t col) { return tableau.at64(row, col); });
}

static void populatePresburgerModule(py::module &m) {
  PyPresburgerTableau::bind(m);
  py::class_<PyPresburgerIntegerRelation>(m, "IntegerRelation",
                                          py::module_local())
      .def(py::init<>(&PyPresburgerIntegerRelation::getFromBuffers))
      .def_property_readonly(MLIR_PYTHON_CAPI_PTR_ATTR,
                             &PyPresburgerIntegerRelation::getCapsule)
      .def("dump",
           [](PyPresburgerIntegerRelation &relation) {
             mlirPresburgerIntegerRelationDump(relation.relation);
           })
      .def(
          "inequalities",
          [](PyPresburgerIntegerRelation &relation) {
            PyPresburgerTableau tableau(
                relation.relation, PyPresburgerTableau::Kind::Inequalities);
            return tableau;
          },
          py::keep_alive<0, 1>())
      .def("get_equality",
           [](PyPresburgerIntegerRelation &relation, int64_t row) {
             unsigned numCol =
                 mlirPresburgerIntegerRelationNumCols(relation.relation);
             std::vector<int64_t> result(numCol);
             for (unsigned i = 0; i < numCol; i++)
               result[i] = mlirPresburgerIntegerRelationAtEq64(
                   relation.relation, row, i);
             return result;
           })
      .def("get_inequality", [](PyPresburgerIntegerRelation &relation,
                                int64_t row) {
        unsigned numCol =
            mlirPresburgerIntegerRelationNumCols(relation.relation);
        std::vector<int64_t> result(numCol);
        for (unsigned i = 0; i < numCol; i++)
          result[i] =
              mlirPresburgerIntegerRelationAtIneq64(relation.relation, row, i);
        return result;
      });
}

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_mlirPresburger, m) {
  m.doc() = "MLIR Presburger utilities";
  populatePresburgerModule(m);
}
