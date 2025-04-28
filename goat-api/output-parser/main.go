package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"image"
	_ "image/png"
	"log"
	"os"
	"path"
	"strings"
)

type Payload struct {
	Angle          float32
	Distance       float32
	Image          string
	BodyLength     float32
	ShoulderHeight float32
	RumpHeight     float32
	Weight         float32
}

type P struct {
	Angle          float32
	Distance       float32
	BodyLength     float32
	ShoulderHeight float32
	RumpHeight     float32
	Weight         float32
}

func main() {
	infileName := flag.String("infile", "output.json", "path to file with json data")
	outputDir := flag.String("outdir", "out/", "output directory where the images are created")
	flag.Parse()

	file, err := os.Open(*infileName)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	buf := make([]byte, 1024*1024)
	scanner.Buffer(buf, 2048*2048)
	counter := 0
	log.Printf("reading %s, writing to %s\n", *infileName, *outputDir)

	for scanner.Scan() {
		log.Printf("parsing row number %d", counter)
		counter++
		line := scanner.Text()
		var p Payload
		err = json.Unmarshal([]byte(line), &p)
		if err != nil {
			log.Printf("cannot read image number %d\n", counter)
			continue
		}

		_, err := saveImage(p.Image, *outputDir, fmt.Sprintf("%d_image", counter))
		if err != nil {
			log.Printf("error while writing file %d", counter)
			log.Print(err)
			continue
		}

		err = saveData(p, *outputDir, fmt.Sprintf("%d_data", counter))
		if err != nil {
			log.Printf("error while writing file %d", counter)
			log.Print(err)
			continue
		}

	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

}

func saveData(data Payload, outpath string, fileNameBase string) error {
	imageless := P{
		Angle:          data.Angle,
		Distance:       data.Distance,
		BodyLength:     data.BodyLength,
		ShoulderHeight: data.ShoulderHeight,
		RumpHeight:     data.RumpHeight,
		Weight:         data.Weight,
	}
	jsonString, err := json.Marshal(imageless)
	if err != nil {
		return err
	}
	fileName := path.Join(outpath, fileNameBase+".json")
	return os.WriteFile(fileName, jsonString, 0644)
}

func saveImage(img string, outpath string, fileNameBase string) (string, error) {
	idx := strings.Index(img, ";base64,")
	if idx < 0 {
		return "", errors.New("invalid string")
	}
	reader := base64.NewDecoder(base64.StdEncoding, strings.NewReader(img[idx+8:]))
	buff := bytes.Buffer{}
	_, err := buff.ReadFrom(reader)
	if err != nil {
		return "", err
	}

	_, fm, err := image.DecodeConfig(bytes.NewReader(buff.Bytes()))
	if err != nil {
		return "", err
	}

	fileName := path.Join(outpath, fileNameBase+"."+fm)
	err = os.WriteFile(fileName, buff.Bytes(), 0644)
	return fileName, err
}
