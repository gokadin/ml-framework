package tensor

import (
    "github.com/stretchr/testify/assert"
    "testing"
)

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
