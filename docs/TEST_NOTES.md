# Test notes

## TC35 - the expected value in the specification looks wrong
Inputs: A {2,5} = 1..10, B {5,4} = 1..20 (row-major).
Specification expects `[[190,200,210,220],[440,475,510,545]]`.
The product of those inputs is `[[175,190,205,220],[400,440,480,520]]` (checked by hand and with NumPy):
row 0, column 0 = 1*1 + 2*5 + 3*9 + 4*13 + 5*17 = 175.
The library returns the mathematically correct matrix. It is recorded as FAIL against the specification as
written, with a note explaining why; ask the evaluator to confirm. `common_test` ignores this one case in its
exit code only when the output equals the NumPy-verified product.

## Semantics
- reshape(): deep copy. reshape_(): in-place, no data movement.
- Copy construction / assignment: deep copy (TC47, TC48).
- Errors: exceptions (`std::out_of_range`, `std::invalid_argument`, `std::length_error`, `std::bad_alloc`).
- Rank-0 scalars are supported (`Tensor(std::vector<size_t>{}, {5})`).
