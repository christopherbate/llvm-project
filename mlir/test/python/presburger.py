from mlir import presburger
import numpy as np

eqs = np.zeros((2, 3), dtype=np.int64)
eqs[1, 0] = 1
eqs[1, 1] = 2
eqs[1, 2] = 3
print(eqs)
ineqs_data = np.zeros((2, 3), dtype=np.int64)
ineqs_data[0, 0] = 99
ineqs_data[0, 2] = 91324

relation = presburger.IntegerRelation(ineqs_data, eqs, 1, 1)
print(relation.get_equality(0)[0])
print(relation.get_equality(0)[1])
print(relation.get_equality(0)[2])
print(relation.get_equality(1)[0])
print(relation.get_equality(1)[1])
print(relation.get_equality(1)[2])

print("dumping relation")
relation.dump()

ineqs = relation.inequalities()
for i in range(ineqs_data.shape[0]):
    for j in range(ineqs_data.shape[1]):
        print(ineqs_data[i, j])
