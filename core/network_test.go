package core

import (
    "github.com/gokadin/ml-framework/tensor"
    "github.com/stretchr/testify/assert"
    "math"
    "testing"
)

func TestNetwork_Forward_oneLayer(t *testing.T) {
    n := NewNetwork().
        AddInputLayer(2).
        AddOutputLayer(1, tensor.ActivationFunctionIdentity)
    data := tensor.NewTensor([][]float64{{1, 1}})

    pred := n.Forward(data)

    expected := tensor.NewTensor([][]float64{{n.InputLayer().GetParameters()[0].Data()[0][0] + n.InputLayer().GetParameters()[0].Data()[1][0]}})
    assert.Equal(t, expected.Data(), pred.Data())
}

func TestNetwork_Forward_twoLayers(t *testing.T) {
    n := NewNetwork().
        AddInputLayer(2).
        AddHiddenLayer(2, tensor.ActivationFunctionIdentity).
        AddOutputLayer(1, tensor.ActivationFunctionIdentity)
    data := tensor.NewTensor([][]float64{{1, 1}})

    pred := n.Forward(data)

    l1n1 := n.InputLayer().GetParameters()[0].Data()[0][0] + n.InputLayer().GetParameters()[0].Data()[1][0]
    l1n2 := n.InputLayer().GetParameters()[0].Data()[0][1] + n.InputLayer().GetParameters()[0].Data()[1][1]
    expected := tensor.NewTensor([][]float64{{l1n1 * n.GetLayer(1).GetParameters()[0].Data()[0][0] + l1n2 * n.GetLayer(1).GetParameters()[0].Data()[1][0]}})
    assert.Equal(t, expected.Data(), pred.Data())
}

func TestNetwork_Forward_twoLayersWithSigmoid(t *testing.T) {
    n := NewNetwork().
        AddInputLayer(2).
        AddHiddenLayer(2, tensor.ActivationFunctionSigmoid).
        AddOutputLayer(1, tensor.ActivationFunctionIdentity)
    data := tensor.NewTensor([][]float64{{1, 1}})

    pred := n.Forward(data)

    l1n1 := 1 / (1 + math.Pow(math.E, -(n.InputLayer().GetParameters()[0].Data()[0][0] + n.InputLayer().GetParameters()[0].Data()[1][0])))
    l1n2 := 1 / (1 + math.Pow(math.E, -(n.InputLayer().GetParameters()[0].Data()[0][1] + n.InputLayer().GetParameters()[0].Data()[1][1])))
    expected := tensor.NewTensor([][]float64{{l1n1 * n.GetLayer(1).GetParameters()[0].Data()[0][0] + l1n2 * n.GetLayer(1).GetParameters()[0].Data()[1][0]}})
    assert.Equal(t, expected.Data(), pred.Data())
}
