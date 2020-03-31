package rl

import (
	"fmt"
	"github.com/gokadin/ml-framework/graphics2D"
	"github.com/gokadin/ml-framework/mat"
	"github.com/gokadin/ml-framework/models"
	"github.com/gokadin/ml-framework/modules"
	"github.com/gokadin/ml-framework/tensor"
	"math/rand"
	"runtime"
	"time"
)

type W4 struct {
	gridWorld *graphics2D.GridWorld
	model *models.Model
	metric *metric
	epsilon float64
	gamma float32
}

func NewW4() *W4 {
	runtime.GOMAXPROCS(runtime.NumCPU())
	rand.Seed(time.Now().UTC().UnixNano())

	return &W4{
		epsilon: 1.0,
		gamma: 0.9,
		metric: newMetric(),
	}
}

func (w *W4) Run() {
	w.metric.start()
	w.model = w.buildModel()
	state := tensor.Variable(mat.WithShape(1, 64))
	nextState := tensor.Variable(mat.WithShape(1, 64))
	y := tensor.Variable(mat.WithShape(1, 4))
	qval := w.model.Predict(state)
	newQ := w.model.PredictNoGrad(nextState)
	loss := w.model.Loss(qval, y)
	w.metric.events.trainingStarted <- true
	for i := 0; i < w.model.Configuration().Epochs; i++ {
		w.metric.events.epochStarted <- true

		w.createGame()
		stateMat := w.gridWorld.GetState()
		w.addNoise(stateMat)
		state.SetData(stateMat)
		gameInProgress := true
		for gameInProgress {
			w.model.Graph().Forward(qval)

			action := w.selectAction(qval)
			w.gridWorld.MakeMove(action)
			w.metric.events.gameActionTaken <- true

			reward := w.gridWorld.GetReward()

			nextStateMat := w.gridWorld.GetState()
			w.addNoise(nextStateMat)
			nextState.SetData(nextStateMat)

			w.model.Graph().Forward(newQ)
			maxQValue := maxValue(newQ.Data().Data())

			yValue := w.calculateYValue(maxQValue, reward)
			y.SetData(mat.NewMat32f(mat.WithShape(1, 4), qval.Data().Copy()))
			y.Data().Set(action, yValue)

			w.model.Graph().Forward(loss)
			w.metric.events.loss <- loss.Data().At(action)
			w.model.Graph().Backward(loss, w.model.TrainableVariables()...)

			for _, parameter := range w.model.TrainableVariables() {
				w.model.Optimizer().Update(parameter, 1, i + 2)
			}

			state.SetData(nextStateMat)

			if reward != -1 {
				gameInProgress = false
				w.metric.events.gameFinished <- true
				if reward == 10 {
					w.metric.events.gameWon <- true
				}
			}
		}

		if i != 0 && i % 100 == 0 {
			w.metric.events.statusUpdate <- true
		}

		if w.epsilon > 0.1 {
			w.epsilon -= 1.0 / float64(w.model.Configuration().Epochs)
		}

		w.metric.events.epochFinished <- true
	}

	w.metric.events.trainingFinished <- true
	w.model.Save("experimental-rl-batch")
}

func (w *W4) selectAction(qval *tensor.Tensor) int {
	if rand.Float64() < w.epsilon {
		return rand.Intn(4)
	}

	return maxIndex(qval.Data().Data())
}

func (w *W4) calculateYValue(maxQValue float32, reward int) float32 {
	if reward == -1 {
		return float32(reward) + w.gamma * maxQValue
	}

	return float32(reward)
}

func (w *W4) RunSaved() {
	w.model = models.Restore("experimental-rl-batch")

	w.test()
}

func (w *W4) test() {
	w.createGame()
	w.gridWorld.Print()
	state := tensor.Variable(mat.WithShape(1, 64))
	qval := w.model.Predict(state)
	counter := 0
	isGameRunning := true
	for isGameRunning {
		stateMat := w.gridWorld.GetState()
		w.addNoise(stateMat)
		state.SetData(stateMat)

		w.model.Graph().Forward(qval)
		action := maxIndex(qval.Data().Data())
		w.gridWorld.MakeMove(action)
		fmt.Println(fmt.Sprintf("taking action %d", action))

		reward := w.gridWorld.GetReward()
		if reward != -1 {
			isGameRunning = false
			if reward > 0 {
				fmt.Println("game won")
			} else {
				fmt.Println("game lost")
			}
		}

		counter++
		if counter > 15 {
			fmt.Println("game lost... too many moves")
			isGameRunning = false
		}

		w.gridWorld.Print()
		time.Sleep(500 * time.Millisecond)
	}
}

func (w *W4) buildModel() *models.Model {
	model := models.Build(
		modules.Dense(164, modules.ActivationRelu),
		modules.Dense(150, modules.ActivationRelu),
		modules.Dense(4, modules.ActivationIdentity))

	model.Configure(models.ModelConfig{
		Epochs: 5000,
		Loss: models.LossMeanSquared,
		LearningRate: 0.0001,
	})

	model.Initialize(64)

	return model
}

func (w *W4) createGame() {
	w.gridWorld = graphics2D.NewGridWorld(1042, 768, 4)
	w.gridWorld.PlaceAgent(0, 3)
	w.gridWorld.PlaceWall(1, 2)
	w.gridWorld.PlaceTarget(2, 0)
	w.gridWorld.PlaceDanger(0, 1)
}

func (w *W4) addNoise(mat *mat.Mat32f) {
	mat.Apply(func(f float32) float32 {
		return f + rand.Float32() / 10
	})
}
