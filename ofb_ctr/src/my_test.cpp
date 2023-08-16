#include <rclcpp/rclcpp.hpp>
#include "OffboardCtr.h"

const std::string default_port = "udp://:14540";

using namespace std::chrono_literals;

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    std::string port;
    if (argc >= 2) {
        port = argv[1];
    }
    else {
        port = default_port;
    }

    OffboardCtr ctr;
    if (!ctr.init(port)) {
        return 1;
    }

    // ctr.telemetry_test();
    
    ctr.gimbalInit();
    ctr.objectTracking();
    
    // std::cout << "Start trans" << std::endl;
    while (rclcpp::ok()) {
        // double ang;
        // std::cin >> ang;
        // // z y x
        // Eigen::AngleAxisd angelAxis(ang, Eigen::Vector3d(1, 0, 0));
        // Eigen::Quaterniond q(angelAxis);
        // q.normalize();
        // ctr.gimbalControl(q);

    }

    // ctr.arm();
    // ctr.takeoff(3);

    // std::cout << "Press Enter to continue:";
    // std::cin.get();

    // ctr.keyboardControl();

    // ctr.land();
    // // ctr.disarm();

    return 0;
}
