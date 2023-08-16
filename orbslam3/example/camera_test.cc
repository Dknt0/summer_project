#include <iostream>

#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip> // IO 流控制文件

#include <ctime>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include "System.h"


using namespace std;

int main(int argc, char **argv) {

    if (argc < 3) {
        cerr << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    cv::VideoCapture cap;
    cap.open(0, cv::CAP_ANY);

    if (!cap.isOpened()) {
        cerr << "Failed to open camera" << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR,true);
    float imageScale = SLAM.GetImageScale(); // 比例


    cv::namedWindow("test", cv::WINDOW_GUI_NORMAL);
    char key;

    cv::Mat img;

    while (!SLAM.isShutDown()) {
        cap.read(img);
        cv::imshow("test", img);

        int width = img.cols * imageScale;
        int height = img.rows * imageScale;
        cv::resize(img, img, cv::Size(width, height));
        chrono::seconds sec = chrono::duration_cast<chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch());
        // cout<<"current seconds: " << sec.count() << endl;
        SLAM.TrackMonocular(img, sec.count());

        key = cv::waitKey(1);
        if (key == 27) {
            break;
        }
    }

    cout << img.size << endl;

    cv::destroyAllWindows();

    return 0;
}
