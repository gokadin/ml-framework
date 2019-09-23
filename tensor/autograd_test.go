package tensor

import (
    "github.com/stretchr/testify/assert"
    "testing"
)

/** GENERAL **/

func Test_Autograd_operationName(t *testing.T) {
    a := NewTensor([][]float64{{1, 1}, {1, 1}})
    b := NewTensor([][]float64{{1, 1}, {1, 1}})

    c := Add(a, b)

    assert.Equal(t, operationAdd, c.operation.name)
}

func Test_Autograd_operationChildren(t *testing.T) {
    a := NewTensor([][]float64{{1, 1}, {1, 1}})
    b := NewTensor([][]float64{{1, 1}, {1, 1}})

    c := Add(a, b)

    assert.Equal(t, 2, len(c.operation.children))
}

func Test_Autograd_operationChildrenAreLeaf(t *testing.T) {
    a := NewTensor([][]float64{{1, 1}, {1, 1}})
    b := NewTensor([][]float64{{1, 1}, {1, 1}})

    c := Add(a, b)

    assert.True(t, c.operation.children[0].isLeaf())
    assert.True(t, c.operation.children[1].isLeaf())
}

func Test_Autograd_Gradient_prunesGraphCorrectly(t *testing.T) {
    a := NewTensor([][]float64{{1, 1}, {1, 1}})
    b := NewTensor([][]float64{{2, 1}, {1, 1}})
    c := Add(a, b)
    d := NewTensor([][]float64{{3, 1}, {1, 1}})
    e := Add(c, d)

    graph := createDerivativeGraph(b.operation, e.operation)

    assert.Equal(t, graph[0].tensor.id, e.id)
    assert.Equal(t, graph[1].tensor.id, c.id)
    assert.Equal(t, graph[2].tensor.id, b.id)
}

/* DERIVATIVES */

func Test_Autograd_Gradient_Derivative(t *testing.T) {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    c := Add(a, b)
    d := NewTensor([][]float64{{3}})
    e := Add(c, d)
    graph := createDerivativeGraph(b.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{1}}, grad)
}

func Test_Autograd_Differentiate_Add(t *testing.T) {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    e := Add(a, b)
    graph := createDerivativeGraph(a.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{1}}, grad)
}

func Test_Autograd_Differentiate_Add_chain(t *testing.T) {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    c := Pow(a, 2)
    e := Add(c, b)
    graph := createDerivativeGraph(a.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{2}}, grad)
}

func Test_Autograd_Differentiate_Sub(t *testing.T) {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    e := Sub(a, b)
    graph := createDerivativeGraph(a.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{1}}, grad)
}

func Test_Autograd_Differentiate_SubAndDeriveB(t *testing.T) {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    e := Sub(a, b)
    graph := createDerivativeGraph(b.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{-1}}, grad)
}

func Test_Autograd_Differentiate_Sub_chain(t *testing.T) {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    c := Pow(b, 2)
    e := Sub(a, c)
    graph := createDerivativeGraph(b.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{-4}}, grad)
}

func Test_Autograd_Differentiate_power(t *testing.T) {
    a := NewTensor([][]float64{{2}})
    e := Pow(a, 2)
    graph := createDerivativeGraph(a.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{4}}, grad)
}

func Test_Autograd_Differentiate_powerHigherThanTwo(t *testing.T) {
    a := NewTensor([][]float64{{2}})
    e := Pow(a, 4)
    graph := createDerivativeGraph(a.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{32}}, grad)
}

func Test_Autograd_Differentiate_power_chain(t *testing.T) {
    a := NewTensor([][]float64{{2}})
    b := Pow(a, 2)
    e := Pow(b, 2)
    graph := createDerivativeGraph(a.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{32}}, grad)
}

func Test_Autograd_Differentiate_divScalar(t *testing.T) {
    a := NewTensor([][]float64{{2}})
    e := DivScalar(a, 2)
    graph := createDerivativeGraph(a.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{0.5}}, grad)
}

func Test_Autograd_Differentiate_divScalar_chain(t *testing.T) {
    a := NewTensor([][]float64{{2}})
    b := Pow(a, 2)
    e := DivScalar(b, 2)
    graph := createDerivativeGraph(a.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{2}}, grad)
}

func Test_Autograd_Differentiate_sum_0(t *testing.T) {
    a := NewTensor([][]float64{{1, 1}, {1, 1}, {1, 1}})
    e := Sum(a, 0)
    graph := createDerivativeGraph(a.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{3, 3}, {3, 3}, {3, 3}}, grad)
}

func Test_Autograd_Differentiate_sum_0_chain(t *testing.T) {
    a := NewTensor([][]float64{{1, 1}, {1, 1}, {1, 1}})
    b := Pow(a, 2)
    e := Sum(b, 0)
    graph := createDerivativeGraph(a.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{6, 6}, {6, 6}, {6, 6}}, grad)
}

func Test_Autograd_Differentiate_dotForA(t *testing.T) {
    a := NewTensor([][]float64{{1, 2}, {2, 1}})
    b := NewTensor([][]float64{{0, 3}, {1, 1}})
    e := Dot(a, b)
    graph := createDerivativeGraph(a.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{15, 7}, {21, 8}}, grad)
}

func Test_Autograd_Differentiate_dotForB(t *testing.T) {
    a := NewTensor([][]float64{{1, 2}, {2, 1}})
    b := NewTensor([][]float64{{0, 3}, {1, 1}})
    e := Dot(a, b)
    graph := createDerivativeGraph(b.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{4, 19}, {5, 17}}, grad)
}

/* DERIVATIVE COMBINATIONS */

func Test_Autograd_Gradient_DerivativePow(t *testing.T) {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    c := Add(a, b)
    e := Pow(c, 2)
    graph := createDerivativeGraph(b.operation, e.operation)

	grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{6}}, grad)
}

func Test_Autograd_Gradient_DerivativePowComplicated(t *testing.T) {
    a := NewTensor([][]float64{{1}})
    b := NewTensor([][]float64{{2}})
    d := NewTensor([][]float64{{3}})
    c := Sub(b, d)
    f := Pow(c, 3)
    e := Sub(a, f)
    graph := createDerivativeGraph(d.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{3}}, grad)
}

func Test_Autograd_Gradient_DerivativeErrorSimple(t *testing.T) {
    y := NewTensor([][]float64{{1}})
    yHat := NewTensor([][]float64{{2}})
    e := DivScalar(Sum(Pow(Sub(y, yHat), 2), 0), 2)
    graph := createDerivativeGraph(y.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{-1}}, grad)
}

func Test_Autograd_Gradient_DerivativeErrorSimpleMultipleValues(t *testing.T) {
    y := NewTensor([][]float64{{1, 1}})
    yHat := NewTensor([][]float64{{2, 2}})
    e := DivScalar(Sum(Pow(Sub(y, yHat), 2), 0), 2)
    graph := createDerivativeGraph(y.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{-1, -1}}, grad)
}

func Test_Autograd_Gradient_DerivativeErrorFullForWWithSingleValue(t *testing.T) {
    x := NewTensor([][]float64{{1}})
    w := NewTensor([][]float64{{1}})
    yHat := NewTensor([][]float64{{2}})
    e := DivScalar(Sum(Pow(Sub(Dot(x, w), yHat), 2), 0), 2)
    graph := createDerivativeGraph(w.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{-1}}, grad)
}

func Test_Autograd_Gradient_DerivativeErrorFullForW(t *testing.T) {
    x := NewTensor([][]float64{{0, 1}})
    w := NewTensor([][]float64{{1, 1}, {1, 1}})
    yHat := NewTensor([][]float64{{2, 2}})
    y := Dot(x, w)
    diff := Sub(y, yHat)
    power := Pow(diff, 2)
    //summation := Sum(power, 0)
    e := DivScalar(power, 2)
    graph := createDerivativeGraph(w.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{0, -1}, {0, -1}}, grad)
}

func Test_Autograd_Gradient_DerivativeOfDotAndSub(t *testing.T) {
    x := NewTensor([][]float64{{0, 1}})
    w := NewTensor([][]float64{{1, 1}, {1, 1}})
    yHat := NewTensor([][]float64{{2, 2}})
    y := Dot(x, w)
    e := Sub(y, yHat)
    graph := createDerivativeGraph(w.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{0, 0}, {1, 1}}, grad)
}

func Test_Autograd_Gradient_DerivativeOfDotSubAndPow(t *testing.T) {
    x := NewTensor([][]float64{{0, 1}})
    w := NewTensor([][]float64{{1, 1}, {1, 1}})
    yHat := NewTensor([][]float64{{2, 2}})
    y := Dot(x, w)
    s := Sub(y, yHat)
    e := Pow(s, 2)
    graph := createDerivativeGraph(w.operation, e.operation)

    grad := graph[0].differentiate()

    assert.Equal(t, [][]float64{{0, 0}, {-2, -2}}, grad)
}
