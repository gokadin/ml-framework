package tensor

import "testing"

func Test_Autograd_operationName(t *testing.T) {
    a := NewTensor([][]float64{{1, 1}, {1, 1}})
    b := NewTensor([][]float64{{1, 1}, {1, 1}})

    c := Add(a, b)

    if c.operation.name != operationAdd {
        t.Fatalf("expected %s, got %s", operationAdd, c.operation.name)
    }
}

func Test_Autograd_operationChildren(t *testing.T) {
    a := NewTensor([][]float64{{1, 1}, {1, 1}})
    b := NewTensor([][]float64{{1, 1}, {1, 1}})

    c := Add(a, b)

    if len(c.operation.children) != 2 {
        t.Fatalf("expected %d, got %d", 2, len(c.operation.children))
    }
}

func Test_Autograd_operationChildrenAreLeaf(t *testing.T) {
    a := NewTensor([][]float64{{1, 1}, {1, 1}})
    b := NewTensor([][]float64{{1, 1}, {1, 1}})

    c := Add(a, b)

    if !c.operation.children[0].isLeaf() || !c.operation.children[1].isLeaf() {
        t.Fatal("expected true, got false")
    }
}

func Test_Autograd_Gradient_prunesGraphCorrectly(t *testing.T) {
    autograd := NewAutograd()
    a := NewTensor([][]float64{{1, 1}, {1, 1}})
    b := NewTensor([][]float64{{1, 1}, {1, 1}})
    c := Add(a, b)

    graph := autograd.pruneGraph(a, c)

    if graph == nil {
        t.Fatal("expected true, got false")
    }
}
