package rl

//import (
//	"fmt"
//	"github.com/gokadin/ml-framework/graphics2D"
//	"github.com/gokadin/ml-framework/mat"
//	"github.com/gokadin/ml-framework/models"
//	"github.com/gokadin/ml-framework/modules"
//	"github.com/gokadin/ml-framework/runners"
//	"github.com/gokadin/ml-framework/tensor"
//	"math/rand"
//	"runtime"
//	"sync"
//	"time"
//)
//
///*
//	Implements experience learning using batches
// */
//type W4 struct {
//	gridWorld *graphics2D.GridWorld
//	model *models.Model
//	metric *metric
//	epsilon float64
//	gamma float32
//	maxMoves int
//	batchSize int
//	memSize int
//	stateSize int
//	numActions int
//}
//
//func NewW4() *W4 {
//	runtime.GOMAXPROCS(runtime.NumCPU())
//	rand.Seed(time.Now().UTC().UnixNano())
//
//	return &W4{
//		epsilon: 1.0,
//		gamma: 0.9,
//		maxMoves: 50,
//		memSize: 1000,
//		batchSize: 200,
//		stateSize: 64,
//		numActions: 4,
//		metric: newMetric(),
//	}
//}
//
//func (w *W4) Run() {
//	w.metric.start()
//	w.model = w.buildModel()
//	//w.model = models.Restore("experimental-rl-batch")
//
//	replayBuffer := NewReplayBuffer(w.memSize, w.batchSize)
//
//	oldState := tensor.Variable(1, w.stateSize)
//	newState := tensor.Variable(1, w.stateSize)
//	oldQVal := w.model.PredictNoGrad(oldState)
//
//	batchOldState := tensor.Variable(w.batchSize, w.stateSize)
//	batchOldQVal := w.model.Predict(batchOldState)
//	batchNewState := tensor.Variable(w.batchSize, w.stateSize)
//	batchNewQVal := w.model.PredictNoGrad(batchNewState)
//
//	batchY := tensor.Variable(w.batchSize, w.numActions)
//	loss := w.model.Loss(batchOldQVal, batchY)
//
//	w.metric.events.trainingStarted <- true
//
//	for i := 0; i < w.model.Configuration().Epochs; i++ {
//		graph := tensor.NewGraph()
//		w.metric.events.epochStarted <- true
//		w.createRandomGame()
//		//w.createGame()
//
//		oldState.SetData(w.addNoise(w.gridWorld.GetState()).Data())
//
//		moveCounter := 0
//		gameInProgress := true
//		for gameInProgress {
//			moveCounter++
//
//			graph.Forward(oldQVal)
//			action := w.selectAction(oldQVal)
//			w.gridWorld.MakeMove(action)
//			w.metric.events.gameActionTaken <- true
//			reward := w.gridWorld.GetReward()
//
//			newState.SetData(w.addNoise(w.gridWorld.GetState()).Data())
//
//			replayBuffer.Append(oldState.ToMat32f(), newState.ToMat32f(), action, reward)
//			if replayBuffer.IsFull() {
//				batchOldStateSlice := make([]float32, w.batchSize * w.stateSize)
//				batchNewStateSlice := make([]float32, w.batchSize * w.stateSize)
//				experienceBatch := replayBuffer.NextBatch()
//				for batchIndex, experience := range experienceBatch {
//					for stateIndex := 0; stateIndex < w.stateSize; stateIndex++ {
//						batchOldStateSlice[batchIndex * w.stateSize + stateIndex] = experience.oldState.At(stateIndex)
//						batchNewStateSlice[batchIndex * w.stateSize + stateIndex] = experience.newState.At(stateIndex)
//					}
//				}
//				batchOldState.SetData(batchOldStateSlice)
//				batchNewState.SetData(batchNewStateSlice)
//
//				graph.Forward(batchOldQVal)
//				graph.Forward(batchNewQVal)
//
//				batchYSlice := make([]float32, w.batchSize * w.numActions)
//				for batchIndex, experience := range experienceBatch {
//					maxQValue := maxValue(batchNewQVal.ToFloat32()[batchIndex * w.numActions:batchIndex * w.numActions + w.numActions])
//					for actionIndex := 0; actionIndex < w.numActions; actionIndex++ {
//						if actionIndex == experience.action {
//							batchYSlice[batchIndex * w.numActions + actionIndex] = w.calculateYValue(maxQValue, experience.reward)
//							continue
//						}
//						batchYSlice[batchIndex * w.numActions + actionIndex] = batchOldQVal.ToFloat32()[batchIndex * w.numActions + actionIndex]
//					}
//				}
//				batchY.SetData(batchYSlice)
//
//				graph.Forward(loss)
//				w.metric.events.loss <- loss.ToFloat32()[action]
//				graph.Backward(loss, w.model.TrainableVariables()...)
//				w.model.Optimizer().Update(w.model.TrainableVariables()...)
//			}
//
//			if reward != -1 || moveCounter > w.maxMoves {
//				gameInProgress = false
//				moveCounter = 0
//				w.metric.events.gameFinished <- true
//				if reward == 10 {
//					w.metric.events.gameWon <- true
//				}
//			} else {
//				oldState.SetData(newState.ToFloat32())
//			}
//		}
//
//		if i != 0 && i % 100 == 0 {
//			for i, module := range w.model.Modules() {
//				w.metric.events.moduleWeightAverage <- struct{int; float32}{i, module.GetParameters()[0].ToMat32f().Average()}
//				w.metric.events.moduleBiasAverage <- struct{int; float32}{i, module.GetParameters()[1].ToMat32f().Average()}
//				w.metric.events.moduleWeightGradientAverage <- struct{int; float32}{i, module.GetParameters()[0].GradientToMat32().Average()}
//				w.metric.events.moduleBiasGradientAverage <- struct{int; float32}{i, module.GetParameters()[1].GradientToMat32().Average()}
//			}
//
//			w.metric.events.statusUpdate <- true
//		}
//
//		if w.epsilon > 0.1 {
//			w.epsilon -= 1.0 / float64(w.model.Configuration().Epochs)
//		}
//
//		w.metric.events.epochFinished <- true
//	}
//
//	w.metric.events.trainingFinished <- true
//	wg := &sync.WaitGroup{}
//	wg.Add(1)
//	w.metric.finalize(wg)
//	wg.Wait()
//	w.model.Save("experimental-rl-batch")
//}
//
//func (w *W4) selectAction(qval *tensor.Tensor) int {
//	if rand.Float64() < w.epsilon {
//		return rand.Intn(4)
//	}
//
//	return maxIndex(qval.ToFloat32())
//}
//
//func (w *W4) calculateYValue(maxQValue float32, reward int) float32 {
//	if reward == -1 {
//		return float32(reward) + w.gamma * maxQValue
//	}
//
//	return float32(reward)
//}
//
//func (w *W4) TestPercentage() {
//	w.model = models.Restore("experimental-rl-batch")
//
//	runs := 1000
//	numWins := 0
//	for i := 0; i < runs; i++ {
//		if w.test() {
//			numWins++
//		}
//	}
//
//	fmt.Println(fmt.Sprintf("gridworld performance %2.f%%", float32(numWins) * 100 / float32(runs)))
//}
//
//func (w *W4) TestSingle() {
//	w.model = models.Restore("experimental-rl-batch")
//	w.test()
//}
//
//func (w *W4) test() bool {
//	graph := tensor.NewGraph()
//	w.createRandomGame()
//	//w.createGame()
//	w.gridWorld.Print()
//	state := tensor.Variable(1, 64)
//	qval := w.model.Predict(state)
//	counter := 0
//	isGameRunning := true
//	for isGameRunning {
//		stateMat := w.gridWorld.GetState()
//		w.addNoise(stateMat)
//		state.SetData(stateMat.Data())
//
//		graph.Forward(qval)
//		action := maxIndex(qval.ToFloat32())
//		w.gridWorld.MakeMove(action)
//		reward := w.gridWorld.GetReward()
//		fmt.Println(fmt.Sprintf("action %d reward %d", action, reward))
//
//		if reward != -1 {
//			isGameRunning = false
//			if reward > 0 {
//				fmt.Println("game won")
//				return true
//			} else {
//				fmt.Println("game lost")
//			}
//		}
//
//		counter++
//		if counter > 15 {
//			fmt.Println("game lost... too many moves")
//			isGameRunning = false
//		}
//
//		w.gridWorld.Print()
//		time.Sleep(500 * time.Millisecond)
//	}
//
//	return false
//}
//
//func (w *W4) buildModel() *models.Model {
//	model := models.Build(
//		//modules.Dense(200, modules.ActivationRelu),
//		//modules.Dense(164, modules.ActivationRelu),
//		modules.Dense(150, modules.ActivationRelu),
//		modules.Dense(100, modules.ActivationRelu),
//		modules.Dense(4, modules.ActivationIdentity))
//
//	model.Configure(runners.ModelConfig{
//		Epochs: 1500,
//		Loss: models.LossMeanSquared,
//		LearningRate: 0.0001,
//	})
//
//	model.Initialize(64)
//
//	return model
//}
//
//func (w *W4) createGame() {
//	w.gridWorld = graphics2D.NewGridWorld(1042, 768, 4)
//	w.gridWorld.PlaceAgent(0, 3)
//	w.gridWorld.PlaceWall(1, 2)
//	w.gridWorld.PlaceTarget(2, 0)
//	w.gridWorld.PlaceDanger(0, 1)
//}
//
//func (w *W4) createRandomGame() {
//	w.gridWorld = graphics2D.NewGridWorld(1042, 768, 4)
//	usedPositions := make([]struct{i int; j int}, 0)
//	agentI := rand.Intn(4)
//	agentJ := rand.Intn(4)
//	usedPositions = append(usedPositions, struct{i int; j int}{agentI, agentJ})
//	w.gridWorld.PlaceAgent(agentI, agentJ)
//	for {
//		i := rand.Intn(4)
//		j := rand.Intn(4)
//		if w.positionsContains(usedPositions, i, j) {
//			continue
//		}
//		w.gridWorld.PlaceWall(i, j)
//		usedPositions = append(usedPositions, struct{i int; j int}{i, j})
//		break
//	}
//	for {
//		i := rand.Intn(4)
//		j := rand.Intn(4)
//		if w.positionsContains(usedPositions, i, j) {
//			continue
//		}
//		w.gridWorld.PlaceTarget(i, j)
//		usedPositions = append(usedPositions, struct{i int; j int}{i, j})
//		break
//	}
//	for {
//		i := rand.Intn(4)
//		j := rand.Intn(4)
//		if w.positionsContains(usedPositions, i, j) {
//			continue
//		}
//		w.gridWorld.PlaceDanger(i, j)
//		usedPositions = append(usedPositions, struct{i int; j int}{i, j})
//		break
//	}
//}
//
//func (w *W4) positionsContains(arr []struct{i int; j int}, i, j int) bool {
//	for k := 0; k < len(arr); k++ {
//		if arr[k].i == i && arr[k].j == j {
//			return true
//		}
//	}
//	return false
//}
//
//func (w *W4) addNoise(mat *mat.Mat32f) *mat.Mat32f {
//	mat.Apply(func(f float32) float32 {
//		return f + rand.Float32() / 10
//	})
//	return mat
//}
