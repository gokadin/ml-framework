package tensor

import (
    "fmt"
    "testing"
)

func Test_autograd_mul_simplest_example(t *testing.T) {
    a := NewTensor([][]float64{{2}})
    b := NewTensor([][]float64{{3}})
    c := Mul(a, b)

    c.Backward()

    if a.grad[0][0] != 3 {
        t.Fatalf("expected %f, got %f", 3.0, a.grad[0][0])
    }
    if b.grad[0][0] != 2 {
        t.Fatalf("expected %f, got %f", 2.0, a.grad[0][0])
    }
}

func Test_autograd_mul_withTwoMul(t *testing.T) {
    a := NewTensor([][]float64{{2}})
    b := NewTensor([][]float64{{3}})
    c := Mul(a, b)
    d := NewTensor([][]float64{{4}})
    e := Mul(c, d)

    e.Backward()

    if a.grad[0][0] != 12 {
        t.Fatalf("expected %f, got %f", 12.0, a.grad[0][0])
    }
    if b.grad[0][0] != 8 {
        t.Fatalf("expected %f, got %f", 8.0, a.grad[0][0])
    }
}

func Test_autograd_mul_withTwoMulAndVector(t *testing.T) {
    a := NewTensor([][]float64{{2, 4}})
    b := NewTensor([][]float64{{3, 5}})
    c := Mul(a, b)
    d := NewTensor([][]float64{{4, 4}})
    e := Mul(c, d)

    e.Backward()

    expectedAGrad := [][]float64{{12, 20}}
    expectedBGrad := [][]float64{{8, 16}}
    if !equals(a.grad, expectedAGrad) {
        t.Fatal("gradients are not equals for tensor a:", a.grad, "!=", expectedAGrad)
    }
    if !equals(b.grad, expectedBGrad) {
        t.Fatal("gradients are not equals for tensor b:", b.grad, "!=", expectedBGrad)
    }
}

func Test_autograd_add_simplest_example(t *testing.T) {
    a := NewTensor([][]float64{{2}})
    b := NewTensor([][]float64{{3}})
    c := Add(a, b)

    c.Backward()

    if a.grad[0][0] != 1 {
        t.Fatalf("expected %f, got %f", 1.0, a.grad[0][0])
    }
    if b.grad[0][0] != 1 {
        t.Fatalf("expected %f, got %f", 1.0, a.grad[0][0])
    }
}

func Test_autograd_mul_and_add_withMultipleVectors(t *testing.T) {
    a := NewTensor([][]float64{{2, 4}})
    b := NewTensor([][]float64{{3, 5}})
    c := Mul(a, b)
    d := NewTensor([][]float64{{4, 4}})
    e := Add(c, d)

    e.Backward()

    expectedAGrad := [][]float64{{3, 5}}
    expectedBGrad := [][]float64{{2, 4}}
    if !equals(a.grad, expectedAGrad) {
        t.Fatal("gradients are not equals for tensor a:", a.grad, "!=", expectedAGrad)
    }
    if !equals(b.grad, expectedBGrad) {
        t.Fatal("gradients are not equals for tensor b:", b.grad, "!=", expectedBGrad)
    }
}

func Test_autograd_sub_withVectors(t *testing.T) {
    a := NewTensor([][]float64{{2, 4}})
    b := NewTensor([][]float64{{3, 5}})
    c := Sub(a, b)

    c.Backward()

    expectedAGrad := [][]float64{{1, 1}}
    expectedBGrad := [][]float64{{-1, -1}}
    if !equals(a.grad, expectedAGrad) {
        t.Fatal("gradients are not equals for tensor a:", a.grad, "!=", expectedAGrad)
    }
    if !equals(b.grad, expectedBGrad) {
        t.Fatal("gradients are not equals for tensor b:", b.grad, "!=", expectedBGrad)
    }
}

func Test_autograd_mul_and_sub_withMultipleVectors(t *testing.T) {
    a := NewTensor([][]float64{{2, 4}})
    b := NewTensor([][]float64{{3, 5}})
    c := Mul(a, b)
    d := NewTensor([][]float64{{4, 4}})
    e := Sub(d, c)

    e.Backward()

    expectedAGrad := [][]float64{{-3, -5}}
    expectedBGrad := [][]float64{{-2, -4}}
    if !equals(a.grad, expectedAGrad) {
        t.Fatal("gradients are not equals for tensor a:", a.grad, "!=", expectedAGrad)
    }
    if !equals(b.grad, expectedBGrad) {
        t.Fatal("gradients are not equals for tensor b:", b.grad, "!=", expectedBGrad)
    }
}

func Test_autograd_pow_of2WithVector(t *testing.T) {
    a := NewTensor([][]float64{{2, 4}})
    b := Pow(a, 2)

    b.Backward()

    expectedGrad := [][]float64{{4, 8}}
    if !equals(a.grad, expectedGrad) {
        t.Fatal("gradients are not equals for tensor a:", a.grad, "!=", expectedGrad)
    }
}

func Test_autograd_pow_of3WithVector(t *testing.T) {
    a := NewTensor([][]float64{{2, 4}})
    b := Pow(a, 3)

    b.Backward()

    expectedGrad := [][]float64{{12, 48}}
    if !equals(a.grad, expectedGrad) {
        t.Fatal("gradients are not equals for tensor a:", a.grad, "!=", expectedGrad)
    }
}

func Test_autograd_mul_and_pow_withMultipleVectors(t *testing.T) {
    a := NewTensor([][]float64{{2, 4}})
    b := Pow(a, 3)
    c := NewTensor([][]float64{{4, 4}})
    e := Mul(b, c)

    e.Backward()

    expectedAGrad := [][]float64{{48, 192}}
    expectedCGrad := [][]float64{{8, 64}}
    if !equals(a.grad, expectedAGrad) {
        t.Fatal("gradients are not equals for tensor a:", a.grad, "!=", expectedAGrad)
    }
    if !equals(c.grad, expectedCGrad) {
        t.Fatal("gradients are not equals for tensor c:", c.grad, "!=", expectedCGrad)
    }
}

func Test_autograd_dot_withMultipleVectors(t *testing.T) {
    data := NewTensor([][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}})
    target := NewTensor([][]float64{{0}, {1}, {0}, {1}})
    w := make([]*Tensor, 2)
    w[0] = NewTensor([][]float64{{0.3, 0.5}, {0.6, 0.4}})
    w[1] = NewTensor([][]float64{{0.8}, {0.3}})

    for i := 0; i < 10; i++ {
        pred := Dot(Dot(data, w[0]), w[1])
        loss := SumX(Pow(Sub(pred, target), 2))

        loss.Backward()

        for _, weights := range w {
            weights.mat = sub(weights.mat, mulScalar(weights.grad, 0.1))
            weights.grad = mulScalar(weights.grad, 0)
        }

        fmt.Println(loss.mat)
    }
}

func Test_autograd_BackwardFullWithOneAssociationAndAutograd(t *testing.T) {
    data := NewTensor([][]float64{{1, 1}})
    target := NewTensor([][]float64{{0.5}})

    w := make([]*Tensor, 2)
    w[0] = NewTensor([][]float64{{0.5, 0.5}, {0.5, 0.5}})
    w[1] = NewTensor([][]float64{{0.5}, {0.5}})

    pred := Dot(Dot(data, w[0]), w[1])
    loss := SumX(Pow(Sub(pred, target), 2))

    loss.Backward()

    expectedW0Grad := [][]float64{{0.125, 0.125}, {0.125, 0.125}}
    expectedW1Grad := [][]float64{{0.25}, {0.25}}
    if !equals(w[0].grad, expectedW0Grad) {
        t.Fatalf("gradients are not equal")
    }
    if !equals(w[1].grad, expectedW1Grad) {
        t.Fatalf("gradients are not equal")
    }
}

func Test_autograd_sigmoid(t *testing.T) {
    a := NewTensor([][]float64{{1}, {2}})
    b := a.Sigmoid()
    y1 := b.mat[0][0]
    y2 := b.mat[1][0]

    b.Backward()

    expected := [][]float64{{y1 * (1 - y1)}, {y2 * (1 - y2)}}
    if !equals(a.grad, expected) {
        t.Fatal("gradients are not equals for tensor a:", a.grad, "!=", expected)
    }
}
