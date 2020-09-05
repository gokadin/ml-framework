package telemetry

import (
	"fmt"
	"log"
	"os"
	"time"
)

const (
	defaultLogFolder = "logs"
)

type Logger struct {
	LogToFile bool
	LogFolder string
}

func NewLogger() *Logger {
	return &Logger{
		LogFolder: defaultLogFolder,
	}
}

func (l *Logger) Initialize() {
	if l.LogToFile {
		if _, err := os.Stat(l.LogFolder); os.IsNotExist(err) {
			err := os.Mkdir(l.LogFolder, 0755)
			if err != nil {
				log.Fatal(err)
			}
		}
	}
}

func (l *Logger) Event(category string, value string) {
	if l.LogToFile {
		l.writeToFile(category + ".log", value)
	}
}

func (l *Logger) Trace(value string) {
	l.writeLog("TRACE", value)
}

func (l *Logger) Debug(value string) {
	l.writeLog("DEBUG", value)
}

func (l *Logger) Info(value string) {
	l.writeLog("INFO", value)
}

func (l *Logger) Warn(value string) {
	l.writeLog("WARN", value)
}

func (l *Logger) Error(value string) {
	l.writeLog("ERROR", value)
}

func (l *Logger) Fatal(value string) {
	l.writeLog("FATAL", value)
}

func (l *Logger) writeLog(level, value string) {
	logLine := formatLogLine(level, value)
	fmt.Print(logLine)
	if l.LogToFile {
		l.writeToFile("log.log", logLine)
	}
}

func (l *Logger) writeToFile(filename string, value string) {
	f, err := os.OpenFile(l.LogFolder + "/" + filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0755)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	if _, err := f.WriteString(value); err != nil {
		log.Fatal(err)
	}
}

func formatLogLine(level, value string) string {
	return fmt.Sprintf("%s  %s  %s", time.Now().Format("2006/01/02 15:04:05"), level, value)
}
