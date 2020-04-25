package rl

//import (
//	"fmt"
//	"github.com/encryptio/alias"
//	"github.com/gokadin/ml-framework/graphics2D"
//	"github.com/gokadin/ml-framework/models"
//	"github.com/gokadin/ml-framework/modules"
//	"github.com/gokadin/ml-framework/runners"
//	"github.com/gokadin/ml-framework/tensor"
//	gym "github.com/openai/gym-http-api/binding-go"
//	"math"
//	"math/rand"
//	"runtime"
//	"time"
//)
//
///*
//	Implements the REINFORCE algorithm
// */
//type Reinforce struct {
//	gridWorld *graphics2D.GridWorld
//	model *models.Model
//	targetModel *models.Model
//	metric *metric
//	gamma float32
//	maxEpisodes int
//	maxDur int
//	batchSize int
//	memSize int
//	stateSize int
//	numActions int
//}
//
//func NewReinforce() *Reinforce {
//	runtime.GOMAXPROCS(runtime.NumCPU())
//	rand.Seed(time.Now().UTC().UnixNano())
//
//	return &Reinforce{
//		gamma: 0.99,
//		memSize: 1000,
//		batchSize: 200,
//		stateSize: 64,
//		numActions: 4,
//		maxEpisodes: 500,
//		maxDur: 200,
//		metric: newMetric(),
//	}
//}
//
//const BaseURL = "http://localhost:5000"
//
//func (w *Reinforce) setupGymClient() (*gym.Client, gym.InstanceID) {
//	env, err := gym.NewClient(BaseURL)
//	must(err)
//
//	insts, err := env.ListAll()
//	must(err)
//	fmt.Println("Started with instances:", insts)
//
//	id, err := env.Create("CartPole-v0")
//	must(err)
//
//	actSpace, err := env.ActionSpace(id)
//	must(err)
//	fmt.Printf("Action space: %+v\n", actSpace)
//
//	//must(env.StartMonitor(id, "/tmp/cartpole-monitor", false, false, false))
//
//	return env, id
//}
//
//func (w *Reinforce) Run() {
//	w.metric.start()
//	w.model = w.buildModel()
//	graph := tensor.NewGraph()
//	env, id := w.setupGymClient()
//
//	previousState := tensor.Variable(1, 4)
//	currentState := tensor.Variable(1, 4)
//	actionProbabilities := w.model.Predict(currentState)
//	batchStates := tensor.Variable(1, 1)
//	predBatch := w.model.Predict(batchStates)
//	discountedRewards := tensor.Variable(1, 1)
//	loss := tensor.Neg(tensor.Sum(tensor.Mul(discountedRewards, tensor.Log(predBatch)), 0))
//	//loss := tensor.CrossEntropy(predBatch, discountedRewards)
//	for episode := 0; episode < w.maxEpisodes; episode++ {
//		w.metric.events.epochStarted <- true
//		replayStates := make([]float32, 0)
//		replayActions := make([]int, 0)
//		replayRewards := make([]float32, 0)
//		currentStateMat, err := env.Reset(id)
//		currentState.SetData(w.obsToState(currentStateMat.([]float64)))
//		must(err)
//
//		for t := 0; t < w.maxDur; t++ {
//			graph.Forward(actionProbabilities)
//			action := w.selectAction(actionProbabilities.ToFloat64())
//			previousState.SetData(currentState.ToFloat32())
//			currentStateMat, _, done, _, err := env.Step(id, action, false)
//			w.metric.events.gameActionTaken <- true
//			currentState.SetData(w.obsToState(currentStateMat.([]float64)))
//			must(err)
//
//			replayStates = append(replayStates, previousState.ToFloat32()...)
//			replayActions = append(replayActions, action)
//			replayRewards = append(replayRewards, float32(t + 1))
//
//			if done {
//				break
//			}
//		}
//
//		discountedRewardsMat := w.discountedRewards(replayRewards)
//		discountedRewards.Reshape(len(replayActions), 1).SetData(discountedRewardsMat)
//
//		batchStates.Reshape(len(replayActions), 4).SetData(replayStates)
//		graph.Forward(predBatch)
//		predBatch.SetData(w.updatePredBatch(replayActions, discountedRewardsMat, predBatch.ToFloat32()))
//
//		graph.Forward(loss)
//		w.metric.events.loss <- loss.ToFloat32()[0]
//		graph.Backward(loss, w.model.TrainableVariables()...)
//		w.model.Optimizer().Update(w.model.TrainableVariables()...)
//
//		w.metric.events.epochFinished <- true
//		w.metric.events.statusUpdate <- true
//	}
//
//	env.Close(id)
//	//must(env.CloseMonitor(id))
//}
//
//func (w *Reinforce) getProbBatchMat(actionBatch []int, predBatch []float32) []float32 {
//	numActions := 2
//	proBatchMat := make([]float32, len(actionBatch))
//	for i := 0; i < len(actionBatch); i++ {
//		proBatchMat[i] = predBatch[i * numActions + actionBatch[i]]
//	}
//	return proBatchMat
//}
//
//func (w *Reinforce) updatePredBatch(actionBatch []int, discountedRewards, predBatchMat []float32) []float32 {
//	numActions := 2
//	updated := make([]float32, len(actionBatch) * numActions)
//	for i := 0; i < len(actionBatch); i++ {
//		if actionBatch[i] == 0 {
//			updated[i * numActions + 1] = discountedRewards[i]
//			updated[i * numActions] = predBatchMat[i * numActions]
//		} else {
//			updated[i * numActions] = discountedRewards[i]
//			updated[i * numActions + 1] = predBatchMat[i * numActions + 1]
//		}
//	}
//	return updated
//}
//
//func (w *Reinforce) discountedRewards(rewards []float32) []float32 {
//	discounted := make([]float32, len(rewards))
//	var max float32
//	for i := 0; i < len(rewards); i++ {
//		discounted[i] = float32(math.Pow(float64(w.gamma), float64(i))) * rewards[i]
//
//		if discounted[i] > max {
//			max = discounted[i]
//		}
//	}
//
//	// normalize
//	for i := 0; i < len(discounted); i++ {
//		discounted[i] = discounted[i] / max
//	}
//	return discounted
//}
//
//func (w *Reinforce) selectAction(probabilities []float64) int {
//	x, _ := alias.New(probabilities)
//	index := x.Gen(rand.New(rand.NewSource(time.Now().UTC().UnixNano())))
//	return int(index)
//}
//
//func (w *Reinforce) obsToState(obs []float64) []float32 {
//	state := make([]float32, len(obs))
//	for i := 0; i < len(obs); i++ {
//		state[i] = float32(obs[i])
//	}
//	return state
//}
//
//func (w *Reinforce) buildModel() *models.Model {
//	model := models.Build(
//		modules.Dense(150, modules.ActivationLeakyRelu),
//		modules.Dense(2, modules.ActivationSoftmax))
//
//	model.Configure(runners.ModelConfig{
//		LearningRate: 0.0009,
//	})
//
//	model.Initialize(4)
//
//	return model
//}
//
//func must(err error) {
//	if err != nil {
//		panic(err)
//	}
//}
