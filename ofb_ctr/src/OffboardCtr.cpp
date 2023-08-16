/**
 * Source file of Offboard control class
 * 
 * This is a part of code for my summer internship project 2023 in BMSTU.
 * 
 * Dknt 2023.7
*/

#include <OffboardCtr.h>

using namespace std::chrono_literals;


/**
 * 初始化函数
 * 实例化数传、动作、外部控制等插件，建立状态对应的回调函数
*/
bool OffboardCtr::init(std::string &port) {
    this->_port = port;

    // 仅在 cout 中输出错误信息
    mavsdk::log::subscribe([](mavsdk::log::Level level,
                              const std::string &message,
                              const std::string &file,
                              int line) {
        (void) message;
        (void) file;
        (void) line;
        return (level != mavsdk::log::Level::Err) && (level != mavsdk::log::Level::Info);
    });
    
    // 监听端口
    this->_mavsdk = std::make_shared<mavsdk::Mavsdk>();
    auto add_con_res = _mavsdk->add_any_connection(_port);
    if (add_con_res != mavsdk::ConnectionResult::Success) {
        RCLCPP_ERROR(rclcpp::get_logger("OffboardCtr"), "Connection failed: %d", static_cast<int>(add_con_res));
        return false;
    }

    // 等待连接
    RCLCPP_INFO(rclcpp::get_logger("OffboardCtr"), "Waiting for connection...");
    while (_mavsdk->systems().size() == 0) {
        std::this_thread::sleep_for(500ms);
    }
    
    // 获取系统，初始化插件
    _system = _mavsdk->systems()[0];
    _action = std::make_shared<mavsdk::Action>(_system);
    _telemetry = std::make_shared<mavsdk::Telemetry>(_system);
    _offboard = std::make_shared<mavsdk::Offboard>(_system);
    RCLCPP_INFO(rclcpp::get_logger("OffboardCtr"), "Connected.");

    // 数传配置 配置位置回调函数  NED
    auto tele_res_set_r_pos = _telemetry->set_rate_position(DATA_UPDATE_RATE);
    if (tele_res_set_r_pos != mavsdk::Telemetry::Result::Success) {
        RCLCPP_ERROR(rclcpp::get_logger("OffboardCtr"), "Setting telemetry rate position failed: %d", static_cast<int>(tele_res_set_r_pos));
        return false;
    }
    _telemetry->subscribe_position([this](mavsdk::Telemetry::Position pos) {
            this->_position = pos;
        });
    
    // 数传配置 配置速度位置回调函数  NED
    auto tele_res_set_r_v = _telemetry->set_rate_velocity_ned(DATA_UPDATE_RATE);
    if (tele_res_set_r_v != mavsdk::Telemetry::Result::Success) {
        RCLCPP_ERROR(rclcpp::get_logger("OffboardCtr"), "Setting telemetry rate velocity_ned failed: %d", static_cast<int>(tele_res_set_r_v));
        return false;
    }
    _telemetry->subscribe_velocity_ned([this](mavsdk::Telemetry::VelocityNed vel) {
            this->_velocityNed = vel;
        });
    
    // 数传配置 配置里程计回调函数
    auto tele_res_set_o = _telemetry->set_rate_odometry(DATA_UPDATE_RATE);
    if (tele_res_set_o != mavsdk::Telemetry::Result::Success) {
        RCLCPP_ERROR(rclcpp::get_logger("OffboardCtr"), "Setting telemetry rate rate_odometry failed: %d", static_cast<int>(tele_res_set_r_v));
        return false;
    }
    _telemetry->subscribe_odometry([this](mavsdk::Telemetry::Odometry odo) {
            this->_odometry = odo;
        });

    // ...

    RCLCPP_INFO(rclcpp::get_logger("OffboardCtr"), "Initialization completed.");
    return true;
}

/**
 * 数传测试函数
*/
void OffboardCtr::telemetry_test() {
    while (rclcpp::ok()) {
        // std::cout << "Altitude: " << _position.relative_altitude_m << " m" << std::endl
        //         << "Latitude: " << _position.latitude_deg << std::endl
        //         << "Longitude: " << _position.longitude_deg << '\n';
        // std::cout << "Odometry info " << " w: " << _odometry.q.w
        //                               << " x: " << _odometry.q.x
        //                               << " y: " << _odometry.q.y
        //                               << " z: " << _odometry.q.z << std::endl;
        // Eigen::Quaterniond q(_odometry.q.w, _odometry.q.x, _odometry.q.y, _odometry.q.z);
        // auto eular = q.toRotationMatrix().eulerAngles(2, 1, 0);

        // std::cout << "roll: " << eular[0] << " pitch: " << eular[1] << " yaw: " << eular[2] << std::endl;

        std::this_thread::sleep_for(1s);
    }
}

/**
 * 解锁
*/
bool OffboardCtr::arm() {
    // 等待自检
    RCLCPP_INFO(rclcpp::get_logger("OffboardCtr"), "Waiting for vehicle is RTF...");
    while (_telemetry->health_all_ok()) {
        std::this_thread::sleep_for(500ms);
    }

    // 解锁
    auto arm_res = _action->arm();
    if (arm_res != mavsdk::Action::Result::Success) {
        RCLCPP_ERROR(rclcpp::get_logger("OffboardCtr"), "Armming failed: %d", static_cast<int>(arm_res));
        return false;
    }

    RCLCPP_INFO(rclcpp::get_logger("OffboardCtr"), "Armed.");
    return true;
}

/**
 * 上锁
*/
bool OffboardCtr::disarm() {
    // 上锁
    auto disarm_res = _action->disarm();
    if (disarm_res != mavsdk::Action::Result::Success) {
        RCLCPP_WARN(rclcpp::get_logger("OffboardCtr"), "Disarmming failed: %d. Killing", static_cast<int>(disarm_res));
        auto kill_res = _action->kill();
        if (kill_res != mavsdk::Action::Result::Success) {
            RCLCPP_ERROR(rclcpp::get_logger("OffboardCtr"), "Killing failed: %d", static_cast<int>(kill_res));
            return false;
        }
        RCLCPP_INFO(rclcpp::get_logger("OffboardCtr"), "Killed.");
        return true;
    }

    RCLCPP_INFO(rclcpp::get_logger("OffboardCtr"), "Disarmed.");
    return true;
}

/**
 * 起飞至指定高度
*/
bool OffboardCtr::takeoff(float altitude) {
    // 设置起飞高度
    auto set_alt_res = _action->set_takeoff_altitude(altitude);
    if (set_alt_res != mavsdk::Action::Result::Success) {
        RCLCPP_ERROR(rclcpp::get_logger("OffboardCtr"), "Setting altitude failed: %d", static_cast<int>(set_alt_res));
        return false;
    }

    // 发送起飞指令
    auto takeoff_res = _action->takeoff();
    if (takeoff_res != mavsdk::Action::Result::Success) {
        RCLCPP_ERROR(rclcpp::get_logger("OffboardCtr"), "Taking off failed: %d", static_cast<int>(takeoff_res));
        return false;
    }

    // 等待起飞完成
    std::cout.precision(2);
    while (true) {
        auto cur_altitude = this->_position.relative_altitude_m;
        // 这里的高度总会差 1m 左右，可能是坐标系选取问题
        if (cur_altitude > altitude - 1) {
            break;
        }
        std::cout << "\rTaking off. Target alt: " << altitude << "m. Current alt: "
                  << cur_altitude << "m       " << std::flush;
    }
    std::cout << std::endl;

    RCLCPP_INFO(rclcpp::get_logger("OffboardCtr"), "Taken off.");
    return true;
}

/**
 * 降落 降落后自动上锁
*/
bool OffboardCtr::land() {
    // 发送降落指令
    auto land_res = _action->land();
    if (land_res != mavsdk::Action::Result::Success) {
        RCLCPP_ERROR(rclcpp::get_logger("OffboardCtr"), "Landing failed: %d", static_cast<int>(land_res));
        return false;
    }

    // 等待降落完成
    std::cout.precision(2);
    while (_telemetry->in_air()) {
        auto cur_altitude = this->_position.relative_altitude_m;
        if (cur_altitude < 0.2) {
            break;
        }
        
        std::cout << "\rLanding. Current alt: " << cur_altitude << "m           " << std::flush;
    }
    std::cout << std::endl;

    RCLCPP_INFO(rclcpp::get_logger("OffboardCtr"), "Landed.");
    return true;
}

/**
 * 键盘控制  双线程
*/
void OffboardCtr::keyboardControl() {
    std::shared_ptr<mavsdk::Offboard::VelocityBodyYawspeed> velSetpoint = 
        std::make_shared<mavsdk::Offboard::VelocityBodyYawspeed>();
    std::shared_ptr<std::mutex> m_velSetpoint = std::make_shared<std::mutex>();

    // reset velocity
    velSetpoint->forward_m_s = 0;
    velSetpoint->right_m_s = 0;
    velSetpoint->down_m_s = 0;
    velSetpoint->yawspeed_deg_s = 0;

    // 切换模式前必须发送 setpoint
    auto setpoint_v_res = _offboard->set_velocity_body(*velSetpoint);
    if (setpoint_v_res != mavsdk::Offboard::Result::Success) {
        RCLCPP_WARN(rclcpp::get_logger("OffboardCtr"), "Seting initial velocity failed: %d", static_cast<int>(setpoint_v_res));
    }

    // 切换外部模式
    RCLCPP_INFO(rclcpp::get_logger("OffboardCtr"), "Changing to offboard mode...");
    auto ofb_start_res = _offboard->start();
    if (ofb_start_res != mavsdk::Offboard::Result::Success) {
        RCLCPP_ERROR(rclcpp::get_logger("OffboardCtr"), "Changing to offboard mode failed: %d", static_cast<int>(ofb_start_res));
    }

    // 开线程
    std::shared_ptr<std::promise<bool>> exit_prom = std::make_shared<std::promise<bool>>();
    auto exit_fut = exit_prom->get_future();

    std::thread getKey(std::bind(&OffboardCtr::p_getKey, this, velSetpoint, m_velSetpoint, exit_prom));

    do {
        m_velSetpoint->lock();
        setpoint_v_res = _offboard->set_velocity_body(*velSetpoint);
        m_velSetpoint->unlock();
    } while (exit_fut.wait_for(0.2s) == std::future_status::timeout);
    
    getKey.join();
}

/**
 * 线程函数  获取按键
 * FRD
*/
void OffboardCtr::p_getKey(std::shared_ptr<mavsdk::Offboard::VelocityBodyYawspeed> velSetpoint,
                           std::shared_ptr<std::mutex> m_velSetpoint,
                           std::shared_ptr<std::promise<bool>> exit_prom) {
    char key;
    bool exit_flag = false;
    while (!exit_flag) {
        // get char
        key = getch();
        std::cout << "\rget: " << key << std::flush;

        m_velSetpoint->lock();

        // reset velocity
        velSetpoint->forward_m_s = 0;
        velSetpoint->right_m_s = 0;
        velSetpoint->down_m_s = 0;
        velSetpoint->yawspeed_deg_s = 0;

        // process velocity
        switch (key) {
            // left hand
            case 'w':
                velSetpoint->down_m_s = -VELOCITY_DOWN;
                break;
            
            case 'x':
                velSetpoint->down_m_s = VELOCITY_DOWN;
                break;
            
            case 'a':
                velSetpoint->yawspeed_deg_s = -VELOCITY_YAW;
                break;
            
            case 'd':
                velSetpoint->yawspeed_deg_s = VELOCITY_YAW;
                break;
                
            // right hand
            case 'i':
                velSetpoint->forward_m_s = VELOCITY_FORWARD;
                break;

            case ',':
                velSetpoint->forward_m_s = -VELOCITY_FORWARD;
                break;

            case 'j':
                velSetpoint->right_m_s = -VELOCITY_RIGHT;
                break;

            case 'l':
                velSetpoint->right_m_s = VELOCITY_RIGHT;
                break;

            case 'u':
                velSetpoint->forward_m_s = VELOCITY_FORWARD;
                velSetpoint->right_m_s = -VELOCITY_RIGHT;
                break;

            case 'o':
                velSetpoint->forward_m_s = VELOCITY_FORWARD;
                velSetpoint->right_m_s = VELOCITY_RIGHT;
                break;

            case 'm':
                velSetpoint->forward_m_s = -VELOCITY_FORWARD;
                velSetpoint->right_m_s = -VELOCITY_RIGHT;
                break;

            case '.':
                velSetpoint->forward_m_s = -VELOCITY_FORWARD;
                velSetpoint->right_m_s = VELOCITY_RIGHT;
                break;
            
            // exit
            case 27:
                exit_flag = true;
                exit_prom->set_value(true);
                break;
            
            default:
                break;
        }
        m_velSetpoint->unlock();
    }
}

//// TODO
// void OffboardCtr::keyboardAccControl() {
//
// }

/**
 * 初始化云台控制节点
*/
void OffboardCtr::gimbalInit() {
    this->_controlNode = std::make_shared<ControlNode>();
}

/**
 * 平衡无人机姿态改变的云台姿态控制
*/
void OffboardCtr::gimbalControl(const double yaw_tar, const double pitch_tar) {
    Eigen::Quaterniond q_wb(_odometry.q.w, _odometry.q.x, _odometry.q.y, _odometry.q.z);
    Eigen::Matrix3d R_wb = q_wb.toRotationMatrix();
    double yaw_b = R_wb.eulerAngles(2, 1, 0)[0];
    double pitch_b = R_wb.eulerAngles(2, 1, 0)[1];
    double roll_b = R_wb.eulerAngles(2, 1, 0)[2];

    // std::cout << "yaw: " << yaw_b << " pitch: " << pitch_b << " roll: " << roll_b << std::endl;

    Eigen::AngleAxisd yawAngle(yaw_b, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd pitchAngle(pitch_b, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rowAngle(-roll_b, Eigen::Vector3d::UnitX());

    Eigen::Quaterniond q_bc = pitchAngle * rowAngle;

    // Eigen::Quaterniond q_bc = q_wb.inverse() * q_wc;
    // std::cout << q_bc << std::endl;
    // gimbalControl(q_bc);
    auto eular = q_bc.toRotationMatrix().eulerAngles(2, 1, 0);
    // std::cout << eular.transpose() << std::endl;

    // 俯仰角修正
    {
        auto &i = eular[1];
        if (i > 1.57 && i < 3.2) {
            i -= M_PI;
            i = -i;
        }
        else if (i < -1.57 && i > -3.2) {
            i += M_PI;
            i = -i;
        }
        else if (i > 3.2 && i < -3.2) {
            i = 0.;
        }
    }

    // 滚转角修正
    {
        auto &i = eular[2];
        if (i > 1.57 && i < 3.2) {
            i -= M_PI;
        }
        else if (i < -1.57 && i > -3.2) {
            i += M_PI;
        }
        else if (i > 3.2 && i < -3.2) {
            i = 0.;
        }
    }

    // 偏航角修正
    {
        auto &i = eular[0];
        if (i > 1.57 && i < 3.2) {
            i -= M_PI;
        }
        else if (i < -1.57 && i > -3.2) {
            i += M_PI;
        }
        else if (i > 3.2 && i < -3.2) {
            i = 0.;
        }
    }

    std::cout.precision(3);
    // std::cout << "After Init: " << eular.transpose() << "         \r" << std::flush;
    this->_controlNode->pubEuler(eular[0] + yaw_tar, eular[1] + pitch_tar, eular[2]);
    // std::this_thread::sleep_for(1ms);
}

/**
 * 云台自稳
 * 
 * 最好新开一个线程，并设置一个 future 管理线程运行
*/
void OffboardCtr::gimbalBalance(const double pitch_tar) {
    RCLCPP_INFO(rclcpp::get_logger("OffboardCtr"), "Camera balance mode started.");

    while (rclcpp::ok()) {
        this->gimbalControl(0, pitch_tar);
        std::this_thread::sleep_for(1ms);
    }
}

/**
 * 目标跟踪 + 云台自稳
*/
void OffboardCtr::objectTracking() {
    sensor_msgs::msg::Image img_msg;

    // 图像话题接收
    auto img_subscriber = _controlNode->create_subscription<sensor_msgs::msg::Image>("camera", 10,
        [&](const sensor_msgs::msg::Image &msg) {
            img_msg = msg;
        });

    // 目标检测服务客户端
    auto detection_client = _controlNode->create_client<ofb_ctr::srv::TargetDetection>("yolo_detection");

    RCLCPP_INFO(rclcpp::get_logger("OffboardCtr"), "Waiting for image...");
    while (img_msg.data.data() == nullptr) {
        rclcpp::spin_some(static_cast<rclcpp::Node::SharedPtr>(_controlNode));
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(rclcpp::get_logger("OffboardCtr"), "Interrupted while waiting for image. Exiting.");
            exit(1); // 退出
        }
        std::this_thread::sleep_for(0.5s);
    }
    
    RCLCPP_INFO(rclcpp::get_logger("OffboardCtr"), "Waiting for detection server...");
    while (!detection_client->wait_for_service(0.5s)) {
        if (!rclcpp::ok()) {
            RCLCPP_ERROR(rclcpp::get_logger("OffboardCtr"), "Interrupted while waiting for the service. Exiting.");
            exit(1); // 退出
        }
    }

    // one thread is not stable, we use two threads here
    std::promise<bool> pro;
    auto fut = pro.get_future();

    int u_tar = 0;
    int v_tar = 0;

    double yaw_tar = 0, pitch_tar = 0;
    double yaw_diff = 0, pitch_diff = 0;
    double yaw_sum = 0, pitch_sum = 0;
    std::mutex angle_mutex;

    double yaw_p = 10;
    double yaw_i = 2;

    double pitch_p = 10;
    double pitch_i = 2;


    std::thread control([&]() {
        while (rclcpp::ok() && fut.wait_for(1ms) == std::future_status::timeout) {
            // PID control  T = 0.001s
            angle_mutex.lock();
            yaw_sum += yaw_diff * 0.001;
            pitch_sum += pitch_diff  * 0.001;
            yaw_tar += (yaw_p * yaw_diff + yaw_i * yaw_sum) * 0.001;
            pitch_tar += (pitch_p * pitch_diff + pitch_i * pitch_sum) * 0.001;
            this->gimbalControl(-yaw_tar, pitch_tar);
            angle_mutex.unlock();

        }
    });

    while (rclcpp::ok()) {
        rclcpp::spin_some(static_cast<rclcpp::Node::SharedPtr>(_controlNode));
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(img_msg, "bgr8");

        // cv::Mat img = cv_ptr->image;

        ofb_ctr::srv::TargetDetection::Request request;
        request.src = img_msg;

        auto request_ptr = std::make_shared<ofb_ctr::srv::TargetDetection::Request>(request);
        auto res_future = detection_client->async_send_request(request_ptr);

        rclcpp::spin_until_future_complete(_controlNode, res_future);
        
        auto res = res_future.get();
        u_tar = res->u;
        v_tar = res->v;
        
        if (u_tar == 0 && v_tar == 0) {
            angle_mutex.lock();
            // yaw_tar = 0;
            // pitch_tar = 0;
            yaw_diff = 0;
            pitch_diff = 0;
            angle_mutex.unlock();
        }
        else {
            angle_mutex.lock();
            yaw_diff = atan((u_tar - CAMERA_cx) / CAMERA_fx);
            pitch_diff = atan((v_tar - CAMERA_cy) / CAMERA_fy);
            angle_mutex.unlock();

            // pitch_tar += 0.1 * pitch_diff;
            
            // std::cout << "yaw: " << yaw_tar
            //         << " pitch: " << pitch_tar << std::endl;
            // this->gimbalControl(-yaw_tar, 0);
        }

        cv_bridge::CvImagePtr cv_ptr_res;
        cv_ptr_res = cv_bridge::toCvCopy(res->res);
        cv::imshow("detection result", cv_ptr_res->image);
        char key = cv::waitKey(1);
        if (key =='q') {
            exit(0);
        }
    }

    
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * 云台控制节点构造函数
*/
ControlNode::ControlNode() : Node("ControlNode") {
    this->_rotor1_publisher = this->create_publisher<std_msgs::msg::Float64>("rotor1_cmd", 10);
    this->_rotor2_publisher = this->create_publisher<std_msgs::msg::Float64>("rotor2_cmd", 10);
    this->_rotor3_publisher = this->create_publisher<std_msgs::msg::Float64>("rotor3_cmd", 10);
    RCLCPP_INFO(this->get_logger(), "Initialization completed");
}

/**
 * 云台控制指令发布  弧度制
*/
void ControlNode::pubEuler(double yaw, double pitch, double roll) {
    std_msgs::msg::Float64 roll_m, pitch_m, yaw_m;
    yaw_m.data = yaw;
    pitch_m.data = pitch;
    roll_m.data = roll;
    this->_rotor1_publisher->publish(yaw_m);
    this->_rotor2_publisher->publish(pitch_m);
    this->_rotor3_publisher->publish(roll_m);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * 不按 Enter 获取字符
*/
char getch() {
    char buf = 0;
    struct termios old;
    if (tcgetattr(0, &old) < 0)
        perror("tcsetattr()");
    old.c_lflag &= ~ICANON;
    old.c_lflag &= ~ECHO;
    old.c_cc[VMIN] = 1;
    old.c_cc[VTIME] = 0;
    if (tcsetattr(0, TCSANOW, &old) < 0)
        perror("tcsetattr ICANON");
    if (read(0, &buf, 1) < 0)
        perror ("read()");
    old.c_lflag |= ICANON;
    old.c_lflag |= ECHO;
    if (tcsetattr(0, TCSADRAIN, &old) < 0)
        perror ("tcsetattr ~ICANON");
    return (buf);
}

