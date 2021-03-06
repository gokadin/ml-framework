package runners

import (
	"fmt"
	"github.com/gokadin/ml-framework/telemetry"
	"strconv"
	"time"
)

type metricEvents struct {
	trainingStarted   chan bool
	trainingFinished  chan bool
	epochStarted      chan int
	epochFinished     chan float32
	batchStarted      chan int
	batchFinished     chan bool
	batchLoss         chan float32
	forwardStarted    chan bool
	forwardFinished   chan bool
	backwardStarted   chan bool
	backwardFinished  chan bool
	optimizerStarted  chan bool
	optimizerFinished chan bool
	gameWon           chan bool
}

func makeMetricEvents() metricEvents {
	return metricEvents{
		trainingStarted:   make(chan bool),
		trainingFinished:  make(chan bool),
		epochStarted:      make(chan int),
		epochFinished:     make(chan float32),
		batchStarted:      make(chan int),
		batchFinished:     make(chan bool),
		batchLoss:         make(chan float32),
		forwardStarted:    make(chan bool),
		forwardFinished:   make(chan bool),
		backwardStarted:   make(chan bool),
		backwardFinished:  make(chan bool),
		optimizerStarted:  make(chan bool),
		optimizerFinished: make(chan bool),
	}
}

type metricTiming struct {
	timeMs    int64
	timeAveMs int64
	iteration int
}

type metric struct {
	modelConfig *ModelConfig
	timings     map[string]*metricTiming
	events      metricEvents
	logger      *telemetry.Logger
}

func newMetric(modelConfig *ModelConfig, logger *telemetry.Logger) *metric {
	m := &metric{
		modelConfig: modelConfig,
		events:      makeMetricEvents(),
		timings:     make(map[string]*metricTiming),
		logger:      logger,
	}
	m.timings["epoch"] = &metricTiming{}
	m.timings["batch"] = &metricTiming{}
	m.timings["forward"] = &metricTiming{}
	m.timings["backward"] = &metricTiming{}
	m.timings["optimizer"] = &metricTiming{}
	return m
}

func (m *metric) start() {
	go m.receiveEvents()
}

func (m *metric) receiveEvents() {
	for {
		select {
		case <-m.events.trainingStarted:
			m.logger.Info("training started")
		case <-m.events.trainingFinished:
			m.logger.Info("training finished")
		case epoch := <-m.events.epochStarted:
			m.timings["epoch"].iteration = epoch
			m.timings["epoch"].timeMs = time.Now().UnixNano()
		case epochLoss := <-m.events.epochFinished:
			epochTimeMs := (time.Now().UnixNano() - m.timings["epoch"].timeMs) / int64(time.Millisecond)
			batchTimeAveMs := m.timings["batch"].timeAveMs / (int64(m.timings["batch"].iteration) + 1)
			m.timings["batch"].timeAveMs = 0
			forwardTimeAveMs := m.timings["forward"].timeAveMs / (int64(m.timings["forward"].iteration) + 1)
			m.timings["forward"].timeAveMs = 0
			m.timings["forward"].iteration = 0
			backwardTimeAveMs := m.timings["backward"].timeAveMs / (int64(m.timings["backward"].iteration) + 1)
			m.timings["backward"].timeAveMs = 0
			m.timings["backward"].iteration = 0
			optimizerTimeAveMs := m.timings["optimizer"].timeAveMs / (int64(m.timings["optimizer"].iteration) + 1)
			m.timings["optimizer"].timeAveMs = 0
			m.timings["optimizer"].iteration = 0
			m.logger.Info(fmt.Sprintf("epoch %d finished in %dms with loss %f", m.timings["epoch"].iteration, epochTimeMs, epochLoss))
			m.logger.Info(fmt.Sprintf("ave batch: %dms ave forward: %dms ave backward: %dms ave optim %dms", batchTimeAveMs, forwardTimeAveMs, backwardTimeAveMs, optimizerTimeAveMs))
		case batch := <-m.events.batchStarted:
			m.timings["batch"].iteration = batch
			m.timings["batch"].timeMs = time.Now().UnixNano()
		case <-m.events.batchFinished:
			batchTimeMs := (time.Now().UnixNano() - m.timings["batch"].timeMs) / int64(time.Millisecond)
			m.timings["batch"].timeAveMs += batchTimeMs
		case batchLoss := <-m.events.batchLoss:
			m.logger.Event("loss", strconv.FormatFloat(float64(batchLoss), 'f', -1, 32))
		case <-m.events.forwardStarted:
			m.timings["forward"].iteration++
			m.timings["forward"].timeMs = time.Now().UnixNano()
		case <-m.events.forwardFinished:
			forwardTimeMs := (time.Now().UnixNano() - m.timings["forward"].timeMs) / int64(time.Millisecond)
			m.timings["forward"].timeAveMs += forwardTimeMs
		case <-m.events.backwardStarted:
			m.timings["backward"].iteration++
			m.timings["backward"].timeMs = time.Now().UnixNano()
		case <-m.events.backwardFinished:
			backwardTimeMs := (time.Now().UnixNano() - m.timings["backward"].timeMs) / int64(time.Millisecond)
			m.timings["backward"].timeAveMs += backwardTimeMs
		case <-m.events.optimizerStarted:
			m.timings["optimizer"].iteration++
			m.timings["optimizer"].timeMs = time.Now().UnixNano()
		case <-m.events.optimizerFinished:
			forwardTimeMs := (time.Now().UnixNano() - m.timings["optimizer"].timeMs) / int64(time.Millisecond)
			m.timings["optimizer"].timeAveMs += forwardTimeMs
		}
	}
}
