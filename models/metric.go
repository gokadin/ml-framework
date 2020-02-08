package models

import (
	"fmt"
	"time"
)

type metricEvents struct {
	trainingStarted chan bool
	trainingFinished chan bool
	epochStarted chan int
	epochFinished chan float32
	batchStarted chan int
	batchFinished chan bool
	forwardStarted chan bool
	forwardFinished chan bool
	backwardStarted chan bool
	backwardFinished chan bool
}

func makeMetricEvents() metricEvents {
	return metricEvents{
		trainingStarted: make(chan bool),
		trainingFinished: make(chan bool),
		epochStarted: make(chan int),
		epochFinished: make(chan float32),
		batchStarted: make(chan int),
		batchFinished: make(chan bool),
		forwardStarted: make(chan bool),
		forwardFinished: make(chan bool),
		backwardStarted: make(chan bool),
		backwardFinished: make(chan bool),
	}
}

type metricTiming struct {
	timeMs int64
	timeAveMs int64
	iteration int
}

type metric struct {
	modelConfig *ModelConfig
	timings map[string]*metricTiming
	events metricEvents
}

func newMetric(modelConfig *ModelConfig) *metric {
	m := &metric{
		modelConfig: modelConfig,
		events: makeMetricEvents(),
		timings: make(map[string]*metricTiming),
	}
	m.timings["epoch"] = &metricTiming{}
	m.timings["batch"] = &metricTiming{}
	m.timings["forward"] = &metricTiming{}
	m.timings["backward"] = &metricTiming{}
	return m
}

func (m *metric) start() {
	go m.receiveEvents()
}

func (m *metric) receiveEvents() {
	for {
		select {
		case <- m.events.trainingStarted:
			fmt.Println("beginning training")
		case <- m.events.trainingFinished:
			fmt.Println("training finished")
		case epoch := <- m.events.epochStarted:
			m.timings["epoch"].iteration = epoch
			m.timings["epoch"].timeMs = time.Now().UnixNano()
		case epochLoss := <- m.events.epochFinished:
			epochTimeMs := (time.Now().UnixNano() - m.timings["epoch"].timeMs) / int64(time.Millisecond)
			batchTimeAveMs := m.timings["batch"].timeAveMs / (int64(m.timings["batch"].iteration) + 1)
			m.timings["batch"].timeAveMs = 0
			forwardTimeAveMs := m.timings["forward"].timeAveMs / (int64(m.timings["forward"].iteration) + 1)
			m.timings["forward"].timeAveMs = 0
			m.timings["forward"].iteration = 0
			backwardTimeAveMs := m.timings["backward"].timeAveMs / (int64(m.timings["backward"].iteration) + 1)
			m.timings["backward"].timeAveMs = 0
			m.timings["backward"].iteration = 0
			fmt.Printf("epoch %d finished in %dms with loss %f\n", m.timings["epoch"].iteration, epochTimeMs, epochLoss)
			fmt.Printf("ave batch: %dms ave forward: %dms ave backward: %dms\n", batchTimeAveMs, forwardTimeAveMs, backwardTimeAveMs)
		case batch := <- m.events.batchStarted:
			m.timings["batch"].iteration = batch
			m.timings["batch"].timeMs = time.Now().UnixNano()
		case <- m.events.batchFinished:
			batchTimeMs := (time.Now().UnixNano() - m.timings["batch"].timeMs) / int64(time.Millisecond)
			m.timings["batch"].timeAveMs += batchTimeMs
		case <- m.events.forwardStarted:
			m.timings["forward"].iteration++
			m.timings["forward"].timeMs = time.Now().UnixNano()
		case <- m.events.forwardFinished:
			forwardTimeMs := (time.Now().UnixNano() - m.timings["forward"].timeMs) / int64(time.Millisecond)
			m.timings["forward"].timeAveMs += forwardTimeMs
		case <- m.events.backwardStarted:
			m.timings["backward"].iteration++
			m.timings["backward"].timeMs = time.Now().UnixNano()
		case <- m.events.backwardFinished:
			backwardTimeMs := (time.Now().UnixNano() - m.timings["backward"].timeMs) / int64(time.Millisecond)
			m.timings["backward"].timeAveMs += backwardTimeMs
		}
	}
}
