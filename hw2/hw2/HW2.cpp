#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
	setlocale(LC_ALL, "Russian");
	
	//Открываем видео 
	VideoCapture cap;
	cap.open("video.mov");
	if (!cap.isOpened()) {
			return -1;
	}

	//Загружаем фото (можно еще выбрать image.jpg)
	Mat photo = imread("slide.png");
	if (photo.empty()) {
		return 1;
	}

	namedWindow("Window", WINDOW_NORMAL);

	Mat frame, gray, edges;
	vector<Point2f> screenCorners;
	bool screenDetected = false;

	while (true) {
		cap >> frame;
		if (frame.empty()) {
			cap.set(CAP_PROP_POS_FRAMES, 0);
			continue;
		}

		//Ищем прямоугольники
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		GaussianBlur(gray, gray, Size(5, 5), 0);
		Canny(gray, edges, 50, 150);

		vector<vector<Point>> contours;
		findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		// Выбираем самый большой прямоугольник
		double maxArea = 0;
		vector<Point> screenContour;

		for (const auto& contour : contours) {
			double area = contourArea(contour);
			if (area < 1000) continue;

			vector<Point> approx;
			approxPolyDP(contour, approx, arcLength(contour, true) * 0.02, true);

			if (approx.size() == 4 && area > maxArea) {
				maxArea = area;
				screenContour = approx;
			}
		}

		Mat result = frame.clone();
		screenDetected = !screenContour.empty();

		if (screenDetected) {
			// Преобразуем точки в нужный формат
			screenCorners.clear();
			for (const auto& pt : screenContour) {
				screenCorners.push_back(Point2f(pt.x, pt.y));
			}

			// Сортируем углы
			Point2f center(0, 0);
			for (const auto& pt : screenCorners) center += pt;
			center *= (1.0 / 4);

			vector<Point2f> sorted(4);
			for (const auto& pt : screenCorners) {
				if (pt.x < center.x && pt.y < center.y) sorted[0] = pt;
				else if (pt.x > center.x && pt.y < center.y) sorted[1] = pt;
				else if (pt.x > center.x && pt.y > center.y) sorted[2] = pt;
				else sorted[3] = pt;
			}

			//Вставляем фото
			vector<Point2f> photoCorners = {
				Point2f(0, 0),
				Point2f(photo.cols - 1, 0),
				Point2f(photo.cols - 1, photo.rows - 1),
				Point2f(0, photo.rows - 1)
			};

			Mat H = findHomography(photoCorners, sorted);
			Mat warpedPhoto;
			warpPerspective(photo, warpedPhoto, H, frame.size());

			Mat mask = Mat::zeros(frame.size(), CV_8UC1);
			vector<Point> maskPoints;
			for (const auto& pt : sorted) {
				maskPoints.push_back(Point(pt.x, pt.y));
			}
			fillConvexPoly(mask, maskPoints, Scalar(255));

			warpedPhoto.copyTo(result, mask);

			// Рисуем контур прямоугольника
			polylines(result, maskPoints, true, Scalar(0, 255, 0), 2);
		}
		else {
			return 2;
		}

		imshow("Window", result);

		char key = waitKey(30);
		if (key == 27) break; // выход через esc
	}

	cap.release();
	destroyAllWindows();
	return 0;
}