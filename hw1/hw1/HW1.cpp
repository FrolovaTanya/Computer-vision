#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

vector<Point> points;
//Определяем координаты квадратов по клику ЛКМ
void onMouse(int event, int x, int y, int flags, void* userdata) {
	if (event == EVENT_LBUTTONDOWN) {
		points.push_back(Point(x, y));
		cout << "Add point: (" << x << ", " << y << ")" << endl;
	}
}

int main(int argc, char** argv) {
	
	// Определяем источник видео
	string source;
	bool useDefaultSource = false;

	if (argc < 2) {
		source = "0";  // Камера по умолчанию
		useDefaultSource = true;
	}
	else {
		source = argv[1];
	}

	// Открытие видео источника
	VideoCapture cap;

	bool isNumber = true;
	for (char c : source) {
		if (!isdigit(c)) {
			isNumber = false;
			break;
		}
	}

	if (isNumber) {
		int cameraIndex = stoi(source);
			cap.open(cameraIndex);	
	}
	else {
			cap.open(source);
	}

	// Проверка успешности открытия
	if (!cap.isOpened()) {
		cin.get();
		return -1;
	}

	// Создание окна
	namedWindow("Video for hw", WINDOW_NORMAL);
	setMouseCallback("Video for hw", onMouse, nullptr);

	Mat frame;
	bool paused = false;
	
		while (true) {
		if (!paused) {
			cap >> frame;
			if (frame.empty()) {
				break;
			}
		}

		Mat display = frame.clone();

		// Отрисовка прямоугольников
		for (const auto& pt : points) {
			rectangle(display,
				Point(pt.x - 10, pt.y - 10),
				Point(pt.x + 10, pt.y + 10),
				Scalar(255, 204, 153), 2, LINE_AA);
		}

		imshow("Video for hw", display);

		char key = waitKey(30) & 0xFF;

		if (key == 'q' || key == 'Q') {
			cout << "Quit the programm" << endl;
			break;
		}
		else if (key == 'c' || key == 'C') {
			points.clear();
			cout << "Clear all points" << endl;
		}
	}

	cap.release();
	destroyAllWindows();

	return 0;
}