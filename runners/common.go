package runners

import (
	"github.com/gokadin/ann-core/core"
	"github.com/gokadin/ann-core/layer"
)

func accumulateError(network *core.Network, expectedOutput []float64) float64 {
	err := 0.0
	for i, n := range network.OutputLayer().Nodes() {
		err += n.Output() - expectedOutput[i]
	}
	return err
}

func calculateDeltas(network *core.Network, expectedOutput []float64) {
	calculateOutputDeltas(network.OutputLayer(), expectedOutput)
    calculateHiddenDeltas(network)
}

func calculateHiddenDeltas(network *core.Network) {
	// going backwards from the last hidden layer to the first hidden layer
	for i := network.LayerCount() - 2; i > 0; i-- {
		for _, n := range network.GetLayer(i).Nodes() {
			sumPreviousDeltasAndWeights := 0.0
			for _, c := range n.Connections() {
				sumPreviousDeltasAndWeights += c.NextNode().Delta() * c.Weight()
			}
			n.SetDelta(sumPreviousDeltasAndWeights * network.GetLayer(i).ActivationDerivative()(n.Input()))
		}
	}
}

func calculateOutputDeltas(outputLayer *layer.Layer, expectedOutput []float64) {
    for i, n := range outputLayer.Nodes() {
        n.SetDelta(n.Output() - expectedOutput[i])
	}
}

func accumulateGradients(network *core.Network) {
	// going backwards from the last hidden layer to the input layer
	for i := len(network.GetLayers()) - 2; i >= 0; i-- {
		for _, node := range network.GetLayer(i).Nodes() {
			for _, connection := range node.Connections() {
				connection.AddGradient(connection.NextNode().Delta() * node.Output())
			}
		}
	}
}

func updateWeights(network *core.Network, learningRate float64) {
	for i := 0; i < len(network.GetLayers()) - 1; i++ {
		for _, node := range network.GetLayer(i).Nodes() {
			for _, connection := range node.Connections() {
				connection.UpdateWeight(learningRate)
				connection.ResetGradient()
			}
		}
	}
}
