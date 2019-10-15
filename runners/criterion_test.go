package runners

import (
    "github.com/gokadin/ml-framework/tensor"
    "testing"
)

func Test_Criterion_meanSquared_oneAssociation(t *testing.T) {
    target := tensor.NewTensor([][]float64{{0.5}})
    pred := tensor.NewTensor([][]float64{{1}})
    c := NewCriterion(lossFunctionMeanSquared)

    loss := c.Forward(pred, target)

    expected := tensor.NewTensor([][]float64{{0.125}})
    if !expected.Equals(loss) {
        t.Fatal("tensors don't match", expected.Data(), loss.Data())
    }
}

func Test_Criterion_meanSquared_multipleOutputs(t *testing.T) {
    target := tensor.NewTensor([][]float64{{0.5, 0.5, 0.5}})
    pred := tensor.NewTensor([][]float64{{1, 1, 1}})
    c := NewCriterion(lossFunctionMeanSquared)

    loss := c.Forward(pred, target)

    expected := tensor.NewTensor([][]float64{{0.125, 0.125, 0.125}})
    if !expected.Equals(loss) {
        t.Fatal("tensors don't match", expected.Data(), loss.Data())
    }
}

func Test_Criterion_meanSquared_multipleAssociations(t *testing.T) {
    target := tensor.NewTensor([][]float64{{0.5, 0.5}, {0.5, 0.5}})
    pred := tensor.NewTensor([][]float64{{1, 1}, {1, 1}})
    c := NewCriterion(lossFunctionMeanSquared)

    loss := c.Forward(pred, target)

    expected := tensor.NewTensor([][]float64{{0.25, 0.25}})
    if !expected.Equals(loss) {
        t.Fatal("tensors don't match", expected.Data(), loss.Data())
    }
}
