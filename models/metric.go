package models

import (
	"fmt"
	"time"
)

type metricEvents struct {
	trainingStarted chan bool
	trainingFinished chan bool
	epochStarted chan int
	epochFinished chan float64
	batchStarted chan int
	batchFinished chan bool
}

func makeMetricEvents() metricEvents {
	return metricEvents{
		trainingStarted: make(chan bool),
		trainingFinished: make(chan bool),
		epochStarted: make(chan int),
		epochFinished: make(chan float64),
		batchStarted: make(chan int),
		batchFinished: make(chan bool),
	}
}

type metric struct {
	modelConfig *ModelConfig
	events metricEvents
	epochTime int64
	epochTimeAve int64
	batchTime int64
	batchTimeAve int64
	batchCounter int
	epochCounter int
}

func newMetric(modelConfig *ModelConfig) *metric {
	return &metric{
		modelConfig: modelConfig,
		events: makeMetricEvents(),
	}
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
			m.epochCounter = epoch
			m.epochTime = time.Now().UnixNano()
		case epochLoss := <- m.events.epochFinished:
			epochTimeMs := (time.Now().UnixNano() - m.epochTime) / int64(time.Millisecond)
			fmt.Printf("epoch %d finished in %dms with loss %f\n", m.epochCounter, epochTimeMs, epochLoss)
			fmt.Printf("ave batch time: %dms\n", m.batchTimeAve / int64(m.batchCounter))
			m.batchTimeAve = 0
		case batch := <- m.events.batchStarted:
			m.batchCounter = batch
			m.batchTime = time.Now().UnixNano()
		case <- m.events.batchFinished:
			batchTimeMs := (time.Now().UnixNano() - m.batchTime) / int64(time.Millisecond)
			m.batchTimeAve += batchTimeMs
		}
	}
}
