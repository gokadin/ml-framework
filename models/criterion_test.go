package models

import (
	"github.com/gokadin/ml-framework/tensor"
    "github.com/stretchr/testify/assert"
    "testing"
)

func Test_Criterion_meanSquared_oneAssociation(t *testing.T) {
    target := tensor.NewTensor([][]float64{{0.5}})
    pred := tensor.NewTensor([][]float64{{1}})
    c := newCriterion(LossMeanSquared)

    loss := c.forward(pred, target)

    expected := tensor.NewTensor([][]float64{{0.125}})
    assert.Equal(t, expected.Data(), loss.Data())
}

func Test_Criterion_meanSquared_multipleOutputs(t *testing.T) {
    target := tensor.NewTensor([][]float64{{0.5, 0.5, 0.5}})
    pred := tensor.NewTensor([][]float64{{1, 1, 1}})
    c := newCriterion(LossMeanSquared)

    loss := c.forward(pred, target)

    expected := tensor.NewTensor([][]float64{{0.125, 0.125, 0.125}})
    assert.Equal(t, expected.Data(), loss.Data())
}

func Test_Criterion_meanSquared_multipleAssociations(t *testing.T) {
    target := tensor.NewTensor([][]float64{{0.5, 0.5}, {0.5, 0.5}})
    pred := tensor.NewTensor([][]float64{{1, 1}, {1, 1}})
    c := newCriterion(LossMeanSquared)

    loss := c.forward(pred, target)

    expected := tensor.NewTensor([][]float64{{0.25, 0.25}})
    assert.Equal(t, expected.Data(), loss.Data())
}
