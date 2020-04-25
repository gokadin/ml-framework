package rl

//import (
//	"fmt"
//	"github.com/encryptio/alias"
//	"math"
//	"math/rand"
//	"runtime"
//	"time"
//)
//
//type Workspace struct {
//	rewardHistory map[action][]float32
//	arms []float32
//	bestArm int
//	eps float32
//	averageReward float32
//	softValues []float64
//}
//
//func NewWorkspace() *Workspace {
//	runtime.GOMAXPROCS(runtime.NumCPU())
//	rand.Seed(time.Now().UTC().UnixNano())
//
//	ws := &Workspace{
//		rewardHistory: make(map[action][]float32),
//		arms: make([]float32, 10),
//		eps: 0.1,
//		softValues: make([]float64, 0),
//	}
//
//	var bestProbability float32
//	for i := range ws.arms {
//		ws.arms[i] = rand.Float32()
//		if ws.arms[i] > bestProbability {
//			bestProbability = ws.arms[i]
//			ws.bestArm = i
//		}
//	}
//
//	for i := 0; i < len(ws.arms); i++ {
//		ws.rewardHistory[action(i)] = make([]float32, 0)
//		ws.softValues = append(ws.softValues, 1.0 / float64(len(ws.arms)))
//	}
//
//	return ws
//}
//
//func (w *Workspace) Run() {
//	for i := 0; i < 500; i++ {
//		x, _ := alias.New(w.softValues)
//		index  := x.Gen(rand.New(rand.NewSource(time.Now().UTC().UnixNano())))
//		choice := action(index)
//
//		w.rewardHistory[choice] = append(w.rewardHistory[choice], reward(w.arms[choice]))
//
//		percentCorrect := 100 * float32(len(w.rewardHistory[action(w.bestArm)])) / float32(i + 1)
//
//		averagesOfActions := make([]float32, len(w.arms))
//		for j := 0; j < len(w.arms); j++ {
//			averagesOfActions[j] = w.averageRewardOfAction(action(j))
//		}
//		w.softValues = softmax(averagesOfActions)
//
//		fmt.Println(fmt.Sprintf("Average %f   correct %f%%", w.rewardHistoryAverage(), percentCorrect))
//	}
//}
//
//func (w *Workspace) rewardHistoryAverage() float32 {
//	var average float32
//	var count int
//	for _, rewards := range w.rewardHistory {
//		for _, reward := range rewards {
//			average += reward
//			count++
//		}
//	}
//	return average / float32(count)
//}
//
//func (w *Workspace) averageRewardOfAction(action action) float32 {
//	rewards := w.rewardHistory[action]
//	var sum float32
//	for _, reward := range rewards {
//		sum += reward
//	}
//	return sum / float32(len(rewards))
//}
//
//func (w *Workspace) getBestAction(actions []action) action {
//	var bestAverageReward float32
//	var bestAction action
//	for _, action := range actions {
//		averageReward := w.averageRewardOfAction(action)
//		if averageReward > bestAverageReward {
//			bestAverageReward = averageReward
//			bestAction = action
//		}
//	}
//	return bestAction
//}
//
//func (w *Workspace) getBestArm() int {
//	var bestArm int
//	var bestMean float32
//	for i := range w.arms {
//		var average float32
//		for _, reward := range w.rewardHistory[action(i)] {
//			average += reward
//		}
//		average /= float32(len(w.rewardHistory[action(i)]))
//		if average > bestMean {
//			bestMean = average
//			bestArm = i
//		}
//	}
//
//	return bestArm
//}
//
//func reward(probability float32) float32 {
//	var reward float32
//	for i := 0; i < 10; i++ {
//		if rand.Float32() < probability {
//			reward++
//		}
//	}
//	return reward
//}
//
//func softmax(a []float32) []float64 {
//	result := make([]float64, len(a))
//	sum := 0.0
//	for i := 0; i < len(a); i++ {
//		sum += math.Exp(float64(a[i]))
//	}
//	for i := 0; i < len(a); i++ {
//		result[i] = math.Exp(float64(a[i])) / sum
//	}
//	return result
//}