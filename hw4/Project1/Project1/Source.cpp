#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

#ifdef _WIN32
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

using namespace cv;
using namespace cv::dnn;
using namespace std;

const string MODEL_PATH = "C:/Дурачок/yolo/yolov4.weights";
const string CONFIG_PATH = "C:/Дурачок/yolo/yolov4.cfg";
const string CLASSES_PATH = "C:/Дурачок/yolo/coco.names";
const string DB_PATH = "C:/Дурачок/frame_database";

const string INPUT_VIDEO = "C:/Дурачок/vid.mp4";   
const string OUTPUT_VIDEO = "C:/Дурачок/result.avi";    

const int    FRAME_STEP = 1;      
const float  CONFIDENCE = 0.5f;   


struct BBox {
	int x, y, w, h, classId;
	float conf;
};

string boxToStr(const BBox& b) {
	ostringstream oss;
	oss << b.x << " " << b.y << " " << b.w << " " << b.h << " " << b.classId << " " << b.conf;
	return oss.str();
}

BBox strToBox(const string& s) {
	BBox b;
	istringstream iss(s);
	iss >> b.x >> b.y >> b.w >> b.h >> b.classId >> b.conf;
	return b;
}

string frameFilename(int frameNum, const string& subfolder) {
	ostringstream oss;
	oss << DB_PATH << "/" << subfolder << "/frame_" << setfill('0') << setw(6) << frameNum;
	return oss.str();
}

void saveFrameData(int frameNum, const vector<BBox>& boxes) {
	ofstream f(frameFilename(frameNum, "metadata") + ".txt");
	if (!f.is_open()) return;
	for (const auto& b : boxes) f << boxToStr(b) << endl;
	f.close();
}

vector<BBox> loadFrameData(int frameNum) {
	vector<BBox> boxes;
	ifstream f(frameFilename(frameNum, "metadata") + ".txt");
	if (!f.is_open()) return boxes;
	string line;
	while (getline(f, line)) {
		if (!line.empty()) boxes.push_back(strToBox(line));
	}
	f.close();
	return boxes;
}

bool frameInDB(int frameNum) {
	return fs::exists(frameFilename(frameNum, "metadata") + ".txt");
}


class Detector {
	Net net;
	vector<string> classes;
	float confThresh;
	float nmsThresh = 0.4f;

public:
	Detector(float conf = 0.5f) : confThresh(conf) {
		ifstream f(CLASSES_PATH);
		if (!f.is_open()) {
			cerr << "Ошибка: " << CLASSES_PATH << endl;
			return;
		}
		string line;
		while (getline(f, line)) {
			if (!line.empty()) classes.push_back(line);
		}
		f.close();

		net = readNetFromDarknet(CONFIG_PATH, MODEL_PATH);
		net.setPreferableBackend(DNN_BACKEND_OPENCV);
		net.setPreferableTarget(DNN_TARGET_CPU);
		cout << "YOLOv4 загружен. Классов: " << classes.size() << endl;
	}

	vector<BBox> detect(const Mat& frame) {
		vector<BBox> result;
		Mat blob;
		blobFromImage(frame, blob, 1 / 255.0, Size(416, 416), Scalar(), true, false);
		net.setInput(blob);

		vector<String> outNames;
		vector<int> outLayers = net.getUnconnectedOutLayers();
		vector<String> layerNames = net.getLayerNames();
		for (int idx : outLayers) outNames.push_back(layerNames[idx - 1]);

		vector<Mat> outputs;
		net.forward(outputs, outNames);

		vector<Rect> boxes;
		vector<float> confidences;
		vector<int> classIds;

		for (const auto& output : outputs) {
			for (int i = 0; i < output.rows; i++) {
				float* data = (float*)output.data + i * output.cols;
				if (data[4] < confThresh) continue;

				Mat scores = output.row(i).colRange(5, output.cols);
				Point maxPt;
				double maxScore;
				minMaxLoc(scores, 0, &maxScore, 0, &maxPt);

				float confidence = data[4] * (float)maxScore;
				if (confidence < confThresh) continue;

				int cx = (int)(data[0] * frame.cols);
				int cy = (int)(data[1] * frame.rows);
				int w = (int)(data[2] * frame.cols);
				int h = (int)(data[3] * frame.rows);

				int x = max(0, cx - w / 2);
				int y = max(0, cy - h / 2);
				w = min(w, frame.cols - x);
				h = min(h, frame.rows - y);

				classIds.push_back(maxPt.x);
				confidences.push_back(confidence);
				boxes.push_back(Rect(x, y, w, h));
			}
		}

		vector<int> indices;
		NMSBoxes(boxes, confidences, confThresh, nmsThresh, indices);

		for (int idx : indices) {
			result.push_back({ boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height, classIds[idx], confidences[idx] });
		}
		return result;
	}

	void draw(Mat& frame, const vector<BBox>& boxes) {
		for (const auto& b : boxes) {
			Rect r(b.x, b.y, b.w, b.h);
			rectangle(frame, r, Scalar(0, 255, 0), 2);

			ostringstream oss;
			if (b.classId >= 0 && b.classId < (int)classes.size())
				oss << classes[b.classId] << " " << fixed << setprecision(2) << b.conf;
			else
				oss << "obj " << b.conf;
			string label = oss.str();

			int baseline;
			Size ts = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

			rectangle(frame, Point(b.x, b.y - ts.height - 5),
				Point(b.x + ts.width, b.y), Scalar(0, 255, 0), FILLED);
			putText(frame, label, Point(b.x, b.y - 5),
				FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
		}
	}
};


void buildDatabase() {
	fs::create_directories(DB_PATH + "/metadata");
	fs::create_directories(DB_PATH + "/frames");
	fs::create_directories(DB_PATH + "/annotated");

	VideoCapture cap(INPUT_VIDEO);
	if (!cap.isOpened()) {
		cerr << "Ошибка открытия: " << INPUT_VIDEO << endl;
		return;
	}

	int totalFrames = (int)cap.get(CAP_PROP_FRAME_COUNT);
	cout << "Создание базы из: " << INPUT_VIDEO << endl;
	cout << "Всего кадров: " << totalFrames << ", шаг: " << FRAME_STEP << endl;

	Detector detector(CONFIDENCE);
	Mat frame;
	int frameCount = 0, savedCount = 0;

	while (true) {
		cap >> frame;
		if (frame.empty()) break;
		frameCount++;
		if (frameCount % FRAME_STEP != 0) continue;

		vector<BBox> boxes = detector.detect(frame);
		saveFrameData(frameCount, boxes);

		if (!boxes.empty()) {
			Mat annotated = frame.clone();
			detector.draw(annotated, boxes);
			imwrite(frameFilename(frameCount, "annotated") + ".jpg", annotated);
			savedCount++;
		}

		if (frameCount % (FRAME_STEP * 10) == 0)
			imwrite(frameFilename(frameCount, "frames") + ".jpg", frame);

		if (frameCount % 30 == 0)
			cout << "\rОбработано: " << frameCount << "/" << totalFrames << " | С объектами: " << savedCount << flush;
	}

	cap.release();
	cout << "\nБаза создана! Кадров с объектами: " << savedCount << endl;
}


void renderVideo() {
	VideoCapture cap(INPUT_VIDEO);
	if (!cap.isOpened()) {
		cerr << "Ошибка открытия: " << INPUT_VIDEO << endl;
		return;
	}

	int totalFrames = (int)cap.get(CAP_PROP_FRAME_COUNT);
	double fps = cap.get(CAP_PROP_FPS);
	int w = (int)cap.get(CAP_PROP_FRAME_WIDTH);
	int h = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
	if (fps <= 0) fps = 30;

	VideoWriter writer(OUTPUT_VIDEO, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(w, h));
	if (!writer.isOpened()) {
		cerr << "Ошибка создания: " << OUTPUT_VIDEO << endl;
		return;
	}

	Detector detector(CONFIDENCE);
	Mat frame;
	int frameCount = 0, fromDB = 0, fromDetector = 0;

	cout << "Рендеринг: " << INPUT_VIDEO << " -> " << OUTPUT_VIDEO << endl;

	while (true) {
		cap >> frame;
		if (frame.empty()) break;
		frameCount++;

		vector<BBox> boxes;
		if (frameInDB(frameCount)) {
			boxes = loadFrameData(frameCount);
			fromDB++;
		}
		else {
			boxes = detector.detect(frame);
			fromDetector++;
		}

		detector.draw(frame, boxes);

		ostringstream info;
		info << frameCount << "/" << totalFrames << " [" << (frameInDB(frameCount) ? "DB" : "LIVE") << "]";
		putText(frame, info.str(), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);

		writer.write(frame);

		if (frameCount % 30 == 0)
			cout << "\r" << frameCount << "/" << totalFrames << " | DB: " << fromDB << " LIVE: " << fromDetector << flush;
	}

	cap.release();
	writer.release();
	cout << "\nГотово: " << OUTPUT_VIDEO << endl;
}


void showStats() {
	if (!fs::exists(DB_PATH + "/metadata")) {
		cout << "База пуста." << endl;
		return;
	}

	vector<string> classes;
	ifstream cf(CLASSES_PATH);
	if (cf.is_open()) {
		string line;
		while (getline(cf, line)) if (!line.empty()) classes.push_back(line);
		cf.close();
	}

	int totalFrames = 0, framesWithObjects = 0, totalObjects = 0;
	map<string, int> classCount;

	for (const auto& entry : fs::directory_iterator(DB_PATH + "/metadata")) {
		if (entry.path().extension() != ".txt") continue;
		totalFrames++;

		ifstream f(entry.path());
		if (!f.is_open()) continue;

		string line;
		while (getline(f, line)) {
			if (!line.empty()) {
				BBox b = strToBox(line);
				totalObjects++;
				if (b.classId >= 0 && b.classId < (int)classes.size())
					classCount[classes[b.classId]]++;
			}
		}
		f.close();
		if (totalObjects > 0) framesWithObjects++;
	}

	cout << "\n=== СТАТИСТИКА ===" << endl;
	cout << "Кадров: " << totalFrames << " | С объектами: " << framesWithObjects << " | Объектов: " << totalObjects << endl;
	if (totalFrames > 0) cout << "Среднее на кадр: " << (float)totalObjects / totalFrames << endl;
	for (const auto& p : classCount) cout << "  " << p.first << ": " << p.second << endl;
}


int main() {
	setlocale(LC_ALL, "Russian");

	cout << "=== РАЗМЕТКА ВИДЕО С БАЗОЙ ДАННЫХ ===" << endl;
	cout << "Видео: " << INPUT_VIDEO << endl;
	cout << "Результат: " << OUTPUT_VIDEO << endl;
	cout << "База: " << DB_PATH << endl;
	cout << "-------------------------------------" << endl;
	cout << "1. Создать базу" << endl;
	cout << "2. Отрисовать видео" << endl;
	cout << "3. Всё сразу" << endl;
	cout << "4. Статистика" << endl;
	cout << "Выбор: ";

	int choice;
	cin >> choice;

	switch (choice) {
	case 1: buildDatabase(); showStats(); break;
	case 2: renderVideo(); break;
	case 3: buildDatabase(); renderVideo(); break;
	case 4: showStats(); break;
	}

	return 0;
}