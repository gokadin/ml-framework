package models

import (
    "github.com/gokadin/ml-framework/mat"
    "github.com/gokadin/ml-framework/tensor"
    "github.com/stretchr/testify/assert"
    "math"
    "testing"
)

func Test_Criterion_meanSquared_oneAssociation(t *testing.T) {
    target := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{0.5}))
    pred := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
    c := newCriterion(LossMeanSquared)
    graph := tensor.NewGraph()

    loss := c.forward(pred, target)
    graph.Forward(loss)

    expected := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{0.25}))
    assert.Equal(t, expected.Data(), loss.Data())
}

func Test_Criterion_meanSquared_multipleOutputs(t *testing.T) {
    target := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 3), []float32{0.5, 0.5, 0.5}))
    pred := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 3), []float32{1, 1, 1}))
    c := newCriterion(LossMeanSquared)
    graph := tensor.NewGraph()

    loss := c.forward(pred, target)
    graph.Forward(loss)

    expected := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 3), []float32{0.25, 0.25, 0.25}))
    assert.Equal(t, expected.Data(), loss.Data())
}

func Test_Criterion_meanSquared_multipleAssociations(t *testing.T) {
    target := tensor.Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{0.5, 0.5, 0.5, 0.5}))
    pred := tensor.Constant(mat.NewMat32f(mat.WithShape(2, 2), []float32{1, 1, 1, 1}))
    c := newCriterion(LossMeanSquared)
    graph := tensor.NewGraph()

    loss := c.forward(pred, target)
    graph.Forward(loss)

    expected := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 2), []float32{0.25, 0.25}))
    assert.Equal(t, expected.Data(), loss.Data())
}

func Test_Criterion_crossEntropy_onePositiveAssociation(t *testing.T) {
    target := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{1}))
    pred := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{0.5}))
    c := newCriterion(LossCrossEntropy)
    graph := tensor.NewGraph()

    loss := c.forward(pred, target)
    graph.Forward(loss)

    expected := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{float32(-math.Log(0.5))}))
    assert.True(t, expected.Data().Equals32f(loss.Data()))
}

func Test_Criterion_crossEntropy_multipleClassesOneAssociation(t *testing.T) {
    target := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 3), []float32{1, 0, 0}))
    pred := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 3), []float32{0.5, 0.4, 0.1}))
    c := newCriterion(LossCrossEntropy)
    graph := tensor.NewGraph()

    loss := c.forward(pred, target)
    graph.Forward(loss)

    expected := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{float32(-math.Log(0.5))}))
    assert.True(t, expected.Data().Equals32f(loss.Data()))
}

func Test_Criterion_crossEntropy_multipleClassesMultipleAssociations(t *testing.T) {
    target := tensor.Constant(mat.NewMat32f(mat.WithShape(3, 3), []float32{1, 0, 0, 0, 1, 0, 0, 0, 1}))
    pred := tensor.Constant(mat.NewMat32f(mat.WithShape(3, 3), []float32{0.5, 0.4, 0.1, 0.5, 0.1, 0.4, 0.5, 0.3, 0.2}))
    c := newCriterion(LossCrossEntropy)
    graph := tensor.NewGraph()

    loss := c.forward(pred, target)
    graph.Forward(loss)

    expected := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 1), []float32{float32((-(math.Log(0.5)) + (-math.Log(0.1)) + (-math.Log(0.2))) / 3)}))
    assert.True(t, expected.Data().Equals32f(loss.Data()))
}

func Test_Criterion_crossEntropy_backward(t *testing.T) {
    target := tensor.Constant(mat.NewMat32f(mat.WithShape(1, 4), []float32{0, 1, 0, 0}))
    pred := tensor.Softmax(tensor.Constant(mat.NewMat32f(mat.WithShape(1, 4), []float32{0.3, 0.5, 0.1, 0.1})))
    c := newCriterion(LossCrossEntropy)
    graph := tensor.NewGraph()
    loss := c.forward(pred, target)
    graph.Forward(loss)

    graph.Backward(loss, pred)

    expected := mat.Sub(pred.Data(), target.Data())
    assert.True(t, expected.Equals32f(pred.Gradient()))
}

func Test_Criterion_crossEntropy_backward_multiDimension(t *testing.T) {
    target := tensor.Constant(mat.NewMat32f(mat.WithShape(2, 4), []float32{0, 1, 0, 0, 0, 1, 0, 0}))
    pred := tensor.Constant(mat.NewMat32f(mat.WithShape(2, 4), []float32{0.2, 0.5, 0.1, 0.1, 0.2, 0.5, 0.1, 0.1}))
    c := newCriterion(LossCrossEntropy)
    graph := tensor.NewGraph()
    loss := c.forward(pred, target)
    graph.Forward(loss)

    graph.Backward(loss, pred)

    expected := mat.Sub(pred.Data(), target.Data())
    assert.True(t, expected.Equals32f(pred.Gradient()))
}
