#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <sstream>

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
const string INPUT_VIDEO = "C:/vid.avi";
const string OUTPUT_VIDEO = "C:/Дурачок/result.avi";
const string DATABASE_PATH = "C:/Дурачок/frame_database";

const int LINE_X1 = 0;
const int LINE_Y1 = 400;
const int LINE_X2 = 900;
const int LINE_Y2 = 400;

const int WORK_MODE = 2;    
const int SAVE_STEP = 5;    
const float CONFIDENCE = 0.5f;

struct BBox { int x, y, w, h; float conf; };

struct Person {
	int id;
	Rect bbox;
	Point center;
	bool counted;
	deque<Point> trail;
};

void saveFrameData(int frameNum, const vector<BBox>& boxes) {
	char path[200];
	sprintf_s(path, "%s/frame_%06d.txt", DATABASE_PATH.c_str(), frameNum);
	ofstream f(path);
	for (const auto& b : boxes)
		f << b.x << " " << b.y << " " << b.w << " " << b.h << " " << b.conf << endl;
	f.close();
}

vector<BBox> loadFrameData(int frameNum) {
	vector<BBox> boxes;
	char path[200];
	sprintf_s(path, "%s/frame_%06d.txt", DATABASE_PATH.c_str(), frameNum);
	ifstream f(path);
	if (!f.is_open()) return boxes;
	string line;
	while (getline(f, line)) {
		if (line.empty()) continue;
		BBox b;
		istringstream iss(line);
		iss >> b.x >> b.y >> b.w >> b.h >> b.conf;
		boxes.push_back(b);
	}
	return boxes;
}

bool checkIntersection(Point a1, Point a2, Point b1, Point b2) {
	int d = (a1.x - a2.x) * (b1.y - b2.y) - (a1.y - a2.y) * (b1.x - b2.x);
	if (d == 0) return false;
	double t = (double)((a1.x - b1.x) * (b1.y - b2.y) - (a1.y - b1.y) * (b1.x - b2.x)) / d;
	double u = (double)((a1.x - b1.x) * (a1.y - a2.y) - (a1.y - b1.y) * (a1.x - a2.x)) / d;
	return (t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0);
}

vector<BBox> detectPeople(Net& net, const Mat& frame) {
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

	for (const auto& output : outputs) {
		for (int i = 0; i < output.rows; i++) {
			float* data = (float*)output.data + i * output.cols;
			if (data[4] < CONFIDENCE) continue;
			Mat scores = output.row(i).colRange(5, output.cols);
			Point maxPt;
			double maxScore;
			minMaxLoc(scores, 0, &maxScore, 0, &maxPt);
			if (maxPt.x != 0) continue;
			if (data[4] * maxScore < CONFIDENCE) continue;

			int cx = (int)(data[0] * frame.cols);
			int cy = (int)(data[1] * frame.rows);
			int w = (int)(data[2] * frame.cols);
			int h = (int)(data[3] * frame.rows);
			int x = max(0, cx - w / 2);
			int y = max(0, cy - h / 2);
			w = min(w, frame.cols - x);
			h = min(h, frame.rows - y);

			boxes.push_back(Rect(x, y, w, h));
			confidences.push_back((float)(data[4] * maxScore));
		}
	}

	vector<int> indices;
	NMSBoxes(boxes, confidences, CONFIDENCE, 0.4f, indices);
	for (int idx : indices)
		result.push_back({ boxes[idx].x, boxes[idx].y, boxes[idx].width,
						 boxes[idx].height, confidences[idx] });
	return result;
}

int updateTracking(vector<Person>& people, int& nextId,
	const vector<BBox>& detections,
	Point lineStart, Point lineEnd) {
	int newCrossings = 0;
	vector<Person> newPeople;

	for (const auto& det : detections) {
		Point center(det.x + det.w / 2, det.y + det.h / 2);
		bool found = false;

		for (auto& p : people) {
			int dx = p.center.x - center.x;
			int dy = p.center.y - center.y;
			if (sqrt((double)(dx * dx + dy * dy)) < 80) {
				p.bbox = Rect(det.x, det.y, det.w, det.h);
				p.center = center;
				p.trail.push_back(center);
				if (p.trail.size() > 30) p.trail.pop_front();

				if (!p.counted && p.trail.size() >= 2) {
					if (checkIntersection(p.trail[p.trail.size() - 2], center, lineStart, lineEnd)) {
						p.counted = true;
						newCrossings++;
					}
				}
				newPeople.push_back(p);
				found = true;
				break;
			}
		}
		if (!found) {
			Person p;
			p.id = nextId++;
			p.bbox = Rect(det.x, det.y, det.w, det.h);
			p.center = center;
			p.counted = false;
			p.trail.push_back(center);
			newPeople.push_back(p);
		}
	}
	people = newPeople;
	return newCrossings;
}

void drawAll(Mat& frame, const vector<Person>& people,
	Point ls, Point le, int totalCrossings) {

	for (const auto& p : people) {
		rectangle(frame, p.bbox, Scalar(0, 255, 0), 2);
		putText(frame, "ID:" + to_string(p.id), Point(p.bbox.x, p.bbox.y - 8),
			FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 255, 0), 1);
		for (size_t i = 1; i < p.trail.size(); i++)
			cv::line(frame, p.trail[i - 1], p.trail[i], Scalar(255, 0, 0), 1);
	}

	cv::line(frame, ls, le, Scalar(0, 0, 255), 3);
	circle(frame, ls, 5, Scalar(0, 0, 255), -1);
	circle(frame, le, 5, Scalar(0, 0, 255), -1);

	rectangle(frame, Point(8, 8), Point(280, 65), Scalar(0, 0, 0), FILLED);
	putText(frame, "CROSSED: " + to_string(totalCrossings),
		Point(18, 48), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 255), 2);
}


int main() {
	setlocale(LC_ALL, "Russian");
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

	cout << (WORK_MODE == 1 ? "РЕЖИМ 1: СБОР БАЗЫ" : "РЕЖИМ 2: ВИДЕО") << endl;

	if (WORK_MODE == 1) fs::create_directories(DATABASE_PATH);

	Net net = readNetFromDarknet(CONFIG_PATH, MODEL_PATH);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	VideoCapture cap;
	cap.open(INPUT_VIDEO, CAP_MSMF);
	if (!cap.isOpened()) { cerr << "ОШИБКА видео!" << endl; return -1; }

	int fw = (int)cap.get(CAP_PROP_FRAME_WIDTH);
	int fh = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
	double fps = cap.get(CAP_PROP_FPS);
	int total = (int)cap.get(CAP_PROP_FRAME_COUNT);
	if (fps <= 0) fps = 30;

	Point ls(LINE_X1, LINE_Y1);
	Point le(min(LINE_X2, fw - 1), LINE_Y2);

	VideoWriter writer;
	if (WORK_MODE == 2) {
		writer.open(OUTPUT_VIDEO, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(fw, fh));
		cout << "Запись: " << OUTPUT_VIDEO << endl;
	}

	vector<Person> people;
	int nextId = 0, totalCrossings = 0, savedFrames = 0, frameCount = 0;
	Mat frame;

	// ====== ОБРАБОТКА (БЕЗ ОКОН) ======
	while (true) {
		cap >> frame;
		if (frame.empty()) break;
		frameCount++;

		vector<BBox> detections;
		if (WORK_MODE == 1) {
			detections = detectPeople(net, frame);
			// Сохраняем все кадры (даже пустые)
			saveFrameData(frameCount, detections);
			savedFrames++;
		}
		else {
			detections = loadFrameData(frameCount);
		}

		totalCrossings += updateTracking(people, nextId, detections, ls, le);
		drawAll(frame, people, ls, le, totalCrossings);
		if (WORK_MODE == 2) writer.write(frame);

		if (frameCount % 30 == 0)
			cout << "\rКадр " << frameCount << "/" << total
			<< " | Пересечений: " << totalCrossings << flush;
	}

	cap.release();
	if (writer.isOpened()) writer.release();

	cout << "\n\nГОТОВО! Пересечений: " << totalCrossings << endl;
	if (WORK_MODE == 1)
		cout << "База: " << savedFrames << " кадров. Меняй WORK_MODE=2 и запускай снова." << endl;
	else
		cout << "Видео: " << OUTPUT_VIDEO << endl;

	return 0;
}