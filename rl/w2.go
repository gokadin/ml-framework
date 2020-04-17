package rl

import (
	"fmt"
	"github.com/gokadin/ml-framework/datasets"
	"github.com/gokadin/ml-framework/mat"
	"github.com/gokadin/ml-framework/models"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/tensor"
	"math/rand"
	"runtime"
	"time"
)

type W2 struct {
	state *state
	agent *agent
	model *models.Model
}

func NewW2() *W2 {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UTC().UnixNano())

	return &W2{
		state: newState(),
		agent: newAgent(),
		model: buildModel(),
	}
}

func (w *W2) Run() {
	dataset := datasets.NewDataset()
	dataset.AddData(datasets.TrainingSetX, mat.NewMat32f(mat.WithShape(1, 1), []float32{float32(w.state.currentState())})).OneHot(10)
	x := tensor.Variable(1, 10)

	var rewardSum float32

	for i := 0; i < w.model.Configuration().Epochs; i++ {
		x.SetData(dataset.Get(datasets.TrainingSetX).Data().Data())
		pred := w.model.Predict(x)

		w.model.Forward(pred)

		aveSoftmax := softmax(pred.ToFloat32())
		action := w.agent.choose(aveSoftmax)
		currentReward := w.state.takeAction(action)
		rewardSum += currentReward
		y := tensor.Variable(1, 10)
		predMat := pred.ToFloat32()
		predMat[action] = currentReward
		y.SetData(predMat)

		loss := w.model.Loss(pred, y)
		w.model.Forward(loss)
		w.model.Backward(loss, w.model.TrainableVariables()...)

		for _, parameter := range w.model.TrainableVariables() {
			w.model.Optimizer().Update(parameter, dataset.BatchSize(), (i + 1) * dataset.BatchSize() + dataset.BatchCounter())
		}

		dataset.AddData(datasets.TrainingSetX, mat.NewMat32f(mat.WithShape(1, 1), []float32{float32(w.state.currentState())})).OneHot(10)

		if i != 0 && i % 500 == 0 {
			fmt.Println(fmt.Sprintf("loss %f   reward %f", averageLoss(loss), rewardSum / float32(i)))
		}
	}
}

func buildModel() *models.Model {
	model := models.Build(
		modules.Dense(100, modules.ActivationRelu),
		modules.Dense(10, modules.ActivationRelu))

	model.Configure(models.ModelConfig{
		Epochs: 5000,
		Loss: models.LossMeanSquared,
	})

	return model
}

func averageLoss(loss *tensor.Tensor) float32 {
	return mat.Sum(loss.ToMat32f(), 1).At(0) / float32(loss.Shape().Y)
}
