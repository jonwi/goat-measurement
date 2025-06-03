package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"
)

type Payload struct {
	Angle          float64
	Distance       float64
	BodyLength     float64
	ShoulderHeight float64
	RumpHeight     float64
	BodyHeight     float64
	Weight         float64
	Tag            string
}

type Counter struct {
	mu    sync.Mutex
	value int
}

func (c *Counter) Get() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.value++
	return c.value
}

func goatDataHandler(path string, counter *Counter) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		log.Println("start handling")
		var number = counter.Get()
		err := r.ParseMultipartForm(100 << 20) // 10 MB of buffer else file
		if err != nil {
			log.Printf("Unable to parse form: %v\n", err)
			http.Error(w, "Unable to parse form", http.StatusBadRequest)
			return
		}

		file, _, err := r.FormFile("originalImage")
		if err != nil {
			log.Printf("Error retrieveing originalImage: %v\n", err)
			http.Error(w, "Error retrieving originalImage", http.StatusBadRequest)
			return
		}
		defer file.Close()
		var originalName = fmt.Sprintf("%d_original.png", number)
		dst, err := os.Create(filepath.Join(path, originalName))
		if err != nil {
			log.Printf("Error creating outputFile %s: %v\n", originalName, err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		defer dst.Close()

		if _, err := io.Copy(dst, file); err != nil {
			log.Printf("Error copying file to destination: %v\n", err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}

		file, _, err = r.FormFile("maskedImage")
		if err != nil {
			log.Printf("Error retrieving maskedImage: %v\n", err)
			http.Error(w, "Error retrieving maskedImage", http.StatusBadRequest)
			return
		}
		defer file.Close()

		var maskedName = fmt.Sprintf("%d_masked.png", number)
		dst, err = os.Create(filepath.Join(path, maskedName))
		if err != nil {
			log.Printf("Error creating maskedImage: %v\n", err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		defer dst.Close()

		if _, err := io.Copy(dst, file); err != nil {
			log.Printf("Error copying file to destination: %v\n", err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}

		angle, err := strconv.ParseFloat(r.FormValue("angle"), 32)
		if err != nil {
			log.Printf("Error parsing Data: %v\n", err)
			http.Error(w, "Error parsing Data", http.StatusBadRequest)
			return
		}
		distance, err := strconv.ParseFloat(r.FormValue("distance"), 32)
		if err != nil {
			log.Printf("Error parsing Data: %v\n", err)
			http.Error(w, "Error parsing Data", http.StatusBadRequest)
			return
		}
		bodyLength, err := strconv.ParseFloat(r.FormValue("bodyLength"), 32)
		if err != nil {
			log.Printf("Error parsing Data: %v\n", err)
			http.Error(w, "Error parsing Data", http.StatusBadRequest)
			return
		}
		shoulderHeight, err := strconv.ParseFloat(r.FormValue("shoulderHeight"), 32)
		if err != nil {
			log.Printf("Error parsing Data: %v\n", err)
			http.Error(w, "Error parsing Data", http.StatusBadRequest)
			return
		}
		rumpHeight, err := strconv.ParseFloat(r.FormValue("rumpHeight"), 32)
		if err != nil {
			log.Printf("Error parsing Data: %v\n", err)
			http.Error(w, "Error parsing Data", http.StatusBadRequest)
			return
		}
		bodyHeight, err := strconv.ParseFloat(r.FormValue("bodyHeight"), 32)
		if err != nil {
			log.Printf("Error parsing Data: %v\n", err)
			http.Error(w, "Error parsing Data", http.StatusBadRequest)
			return
		}
		weight, err := strconv.ParseFloat(r.FormValue("weight"), 32)
		if err != nil {
			log.Printf("Error parsing Data: %v\n", err)
			http.Error(w, "Error parsing Data", http.StatusBadRequest)
			return
		}
		tag := r.FormValue("tag")

		var p = Payload{
			Angle:          angle,
			Distance:       distance,
			BodyLength:     bodyLength,
			ShoulderHeight: shoulderHeight,
			RumpHeight:     rumpHeight,
			BodyHeight:     bodyHeight,
			Weight:         weight,
			Tag:            tag,
		}

		var dataName = fmt.Sprintf("%d_data.json", number)
		f, err := os.OpenFile(filepath.Join(path, dataName), os.O_CREATE|os.O_WRONLY, 0777)
		if err != nil {
			log.Printf("Error writing data: %v\n", err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		defer f.Close()

		data, err := json.Marshal(p)
		if err != nil {
			log.Printf("Error parsing data: %v\n", err)
			http.Error(w, "Error parsing data", http.StatusBadRequest)
			return
		}
		_, err = f.Write(data)
		if err != nil {
			log.Printf("Error writing data: %v\n", err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}

		w.WriteHeader(http.StatusOK)
		return
	}
}

func enableCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// TODO: Production
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		log.Printf("Started %s %s", r.Method, r.URL.Path)
		next.ServeHTTP(w, r)
		log.Printf("Completed %s %s in %v", r.Method, r.URL.Path, time.Since(start))
	})
}

func createTimeStampedDir() (string, error) {
	var timestamp = time.Now().Format("2006-01-02_15-04-05")
	var err = os.MkdirAll(timestamp, 0755)
	if err != nil {
		return "", fmt.Errorf("failed to create directory: %v", err)
	}

	return timestamp, nil
}

func main() {
	dir, err := createTimeStampedDir()
	if err != nil {
		log.Printf("Error while creating timestamped dir: %v\n", err)
		return
	}

	var counter = Counter{}

	http.Handle("/", loggingMiddleware(enableCORS(http.HandlerFunc(goatDataHandler(dir, &counter)))))
	log.Fatal(http.ListenAndServe(":8080", nil))
}
