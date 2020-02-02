package datasets

import (
	"log"
	"math/rand"
)

const TrainingSetX = "trainingSetX"
const TrainingSetY = "trainingSetY"
const ValidationSetX = "validationSetX"
const ValidationSetY = "validationSetY"

const defaultBatchSize = 1

type Dataset struct {
	name string
	sets map[string]*set
	batchSize int
	batchCounter int
	shouldShuffle bool
}

func NewDataset() *Dataset {
	return &Dataset{
		sets: make(map[string]*set),
		batchSize: defaultBatchSize,
	}
}

func (d *Dataset) SetName(name string) {
	d.name = name
}

func (d *Dataset) AddData(name string, data [][]float64) {
	d.sets[name] = &set{data}
}

func (d *Dataset) Get(name string) *set {
	if set, ok := d.sets[name]; ok {
		return set
	}

	log.Fatalf("set %s does not exist", name)
	return nil
}

func (d *Dataset) InputSize() int {
	if set, ok := d.sets[TrainingSetX]; ok {
		return len(set.data)
	}

	return 0
}

func (d *Dataset) Shuffle() *Dataset {
	d.shouldShuffle = true
	return d
}

func (d *Dataset) DisableShuffle() *Dataset {
	d.shouldShuffle = false
	return d
}

func (d *Dataset) BatchSize() int {
	return d.batchSize
}

func (d *Dataset) NumBatches() int {
	return d.sets[TrainingSetX].ShapeX() / d.BatchSize()
}

func (d *Dataset) SetBatchSize(batchSize int) *Dataset {
	d.batchSize = batchSize
	return d
}

func (d *Dataset) HasNextBatch() bool {
	return d.batchCounter < d.NumBatches()
}

func (d *Dataset) ResetBatchCounter() {
	d.batchCounter = 0
}

func (d *Dataset) NextBatch() ([][]float64, [][]float64) {
	if d.batchCounter == 0 && d.shouldShuffle {
		d.shuffleData()
	}

	fromIndex := d.batchCounter * d.batchSize
	toIndex := (d.batchCounter + 1) * d.batchSize
	if toIndex > d.sets[TrainingSetX].ShapeX() {
		toIndex = d.sets[TrainingSetX].ShapeX()
	}

	d.batchCounter++

	return d.sets[TrainingSetX].data[fromIndex:toIndex], d.sets[TrainingSetY].data[fromIndex:toIndex]
}

func (d *Dataset) BatchCounter() int {
	return d.batchCounter
}

func (d *Dataset) shuffleData() {
	rand.Shuffle(len(d.sets[TrainingSetX].data), func(i, j int) {
		d.sets[TrainingSetX].data[i], d.sets[TrainingSetX].data[j] = d.sets[TrainingSetX].data[j], d.sets[TrainingSetX].data[i]
		d.sets[TrainingSetY].data[i], d.sets[TrainingSetY].data[j] = d.sets[TrainingSetY].data[j], d.sets[TrainingSetY].data[i]
	})
}
