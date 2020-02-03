package models

import (
    "github.com/gokadin/ml-framework/tensor"
    "github.com/stretchr/testify/assert"
    "math"
    "testing"
)

func Test_Criterion_meanSquared_oneAssociation(t *testing.T) {
    target := tensor.Constant([][]float64{{0.5}})
    pred := tensor.Constant([][]float64{{1}})
    c := newCriterion(LossMeanSquared)
    graph := tensor.NewGraph()

    loss := c.forward(pred, target)
    graph.Forward(loss)

    expected := tensor.Constant([][]float64{{0.25}})
    assert.Equal(t, expected.Data(), loss.Data())
}

func Test_Criterion_meanSquared_multipleOutputs(t *testing.T) {
    target := tensor.Constant([][]float64{{0.5, 0.5, 0.5}})
    pred := tensor.Constant([][]float64{{1, 1, 1}})
    c := newCriterion(LossMeanSquared)
    graph := tensor.NewGraph()

    loss := c.forward(pred, target)
    graph.Forward(loss)

    expected := tensor.Constant([][]float64{{0.25, 0.25, 0.25}})
    assert.Equal(t, expected.Data(), loss.Data())
}

func Test_Criterion_meanSquared_multipleAssociations(t *testing.T) {
    target := tensor.Constant([][]float64{{0.5, 0.5}, {0.5, 0.5}})
    pred := tensor.Constant([][]float64{{1, 1}, {1, 1}})
    c := newCriterion(LossMeanSquared)
    graph := tensor.NewGraph()

    loss := c.forward(pred, target)
    graph.Forward(loss)

    expected := tensor.Constant([][]float64{{0.25, 0.25}})
    assert.Equal(t, expected.Data(), loss.Data())
}

func Test_Criterion_crossEntropy_onePositiveAssociation(t *testing.T) {
    target := tensor.Constant([][]float64{{1}})
    pred := tensor.Constant([][]float64{{0.5}})
    c := newCriterion(LossCrossEntropy)
    graph := tensor.NewGraph()

    loss := c.forward(pred, target)
    graph.Forward(loss)

    expected := tensor.Constant([][]float64{{-math.Log(0.5)}})
    assert.Equal(t, expected.Data(), loss.Data())
}

func Test_Criterion_crossEntropy_oneNegativeAssociation(t *testing.T) {
    target := tensor.Constant([][]float64{{0}})
    pred := tensor.Constant([][]float64{{0.5}})
    c := newCriterion(LossCrossEntropy)
    graph := tensor.NewGraph()

    loss := c.forward(pred, target)
    graph.Forward(loss)

    expected := tensor.Constant([][]float64{{0}})
    assert.Equal(t, expected.Data(), loss.Data())
}

func Test_Criterion_crossEntropy_multipleClassesOneAssociation(t *testing.T) {
    target := tensor.Constant([][]float64{{1, 0, 0}})
    pred := tensor.Constant([][]float64{{0.5, 0.4, 0.1}})
    c := newCriterion(LossCrossEntropy)
    graph := tensor.NewGraph()

    loss := c.forward(pred, target)
    graph.Forward(loss)

    expected := tensor.Constant([][]float64{{-math.Log(0.5)}})
    assert.Equal(t, expected.Data(), loss.Data())
}

func Test_Criterion_crossEntropy_multipleClassesMultipleAssociations(t *testing.T) {
    target := tensor.Constant([][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})
    pred := tensor.Constant([][]float64{{0.5, 0.4, 0.1}, {0.5, 0.1, 0.4}, {0.5, 0.3, 0.2}})
    c := newCriterion(LossCrossEntropy)
    graph := tensor.NewGraph()

    loss := c.forward(pred, target)
    graph.Forward(loss)

    expected := tensor.Constant([][]float64{{((-math.Log(0.5)) + (-math.Log(0.1)) + (-math.Log(0.2))) / 3}})
    assert.Equal(t, expected.Data(), loss.Data())
}
