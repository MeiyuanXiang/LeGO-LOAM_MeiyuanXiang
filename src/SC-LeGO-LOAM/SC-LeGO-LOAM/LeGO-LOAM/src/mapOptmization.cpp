// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ISAM2.h>

#include "Scancontext.h"
#include "utility.h"

using namespace gtsam;

// 地图优化类
class mapOptimization
{
private:
    NonlinearFactorGraph gtSAMgraph; // 非线性因子库
    Values initialEstimate;          // 保存优化前的关键帧位姿
    Values optimizedEstimate;
    ISAM2 *isam;                // 优化器
    Values isamCurrentEstimate; // 优化后的关键帧位姿

    noiseModel::Diagonal::shared_ptr priorNoise;      // 第一帧的噪声
    noiseModel::Diagonal::shared_ptr odometryNoise;   // 上下两个关键帧之间的噪声
    noiseModel::Diagonal::shared_ptr constraintNoise; // 闭环噪声
    noiseModel::Base::shared_ptr robustNoiseModel;

    ros::NodeHandle nh; // 句柄

    ros::Publisher pubLaserCloudSurround; // 发布局部的点云，可视化(rviz里面基本就是全局地图)
    ros::Publisher pubOdomAftMapped;      // 发布map_odometry，2hz左右
    ros::Publisher pubKeyPoses;           // 发布轨迹点

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRegisteredCloud;

	// 订阅原始点，次边缘点，次平面点，外点，odometry，imu
    ros::Subscriber subLaserCloudRaw;
    ros::Subscriber subLaserCloudCornerLast;
    ros::Subscriber subLaserCloudSurfLast;
    ros::Subscriber subOutlierCloudLast;
    ros::Subscriber subLaserOdometry;
    ros::Subscriber subImu;

    nav_msgs::Odometry odomAftMapped;
    tf::StampedTransform aftMappedTrans;
    tf::TransformBroadcaster tfBroadcaster;

    // 所有关键帧的次边缘点，次平面点，外点
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> outlierCloudKeyFrames;

    // 保存当前帧最近的50个关键帧
    deque<pcl::PointCloud<PointType>::Ptr> recentCornerCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> recentSurfCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> recentOutlierCloudKeyFrames;
    int latestFrameID;

    // 不用闭环检测时找到的当前帧最近的50个关键帧
    vector<int> surroundingExistingKeyPosesID;
    deque<pcl::PointCloud<PointType>::Ptr> surroundingCornerCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> surroundingSurfCloudKeyFrames;
    deque<pcl::PointCloud<PointType>::Ptr> surroundingOutlierCloudKeyFrames;

    PointType previousRobotPosPoint; // 上一个关键帧的位姿
    PointType currentRobotPosPoint;  // 当前关键帧的位姿

    // PointType(pcl::PointXYZI)的XYZI分别保存3个方向上的平移和一个索引(cloudKeyPoses3D->points.size())
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D; // 所有关键帧的位置
    // PointTypePose的XYZI保存和cloudKeyPoses3D一样的内容，另外还保存RPY角度以及一个时间值timeLaserOdometry
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D; // 所有关键帧的位姿

    // 结尾有DS代表是downsize，进行过下采样
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses;
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS;

	// 当前帧的原始点，次边缘点，次平面点，外点
    pcl::PointCloud<PointType>::Ptr laserCloudRaw;
    pcl::PointCloud<PointType>::Ptr laserCloudRawDS;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;   // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;     // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS;   // downsampled surf featuer set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOutlierLast;   // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudOutlierLastDS; // corner feature set from odoOptimization

    // 平面点
    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLast;   // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLastDS; // downsampled corner featuer set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    // 用于地图匹配的局部小地图
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    // kdtree用来找地图最近点
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses; // 用于非闭环检测状态下找当前帧最近关键帧
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;     // 用于查找最近闭环检测点

    pcl::PointCloud<PointType>::Ptr RSlatestSurfKeyFrameCloud; // giseop, RS: radius search
    pcl::PointCloud<PointType>::Ptr RSnearHistorySurfKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr RSnearHistorySurfKeyFrameCloudDS;

    pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistoryCornerKeyFrameCloudDS;
    pcl::PointCloud<PointType>::Ptr SCnearHistorySurfKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr SCnearHistorySurfKeyFrameCloudDS;

    pcl::PointCloud<PointType>::Ptr latestCornerKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr SClatestSurfKeyFrameCloud; // giseop, SC: scan context
    pcl::PointCloud<PointType>::Ptr latestSurfKeyFrameCloudDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS;
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames;
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS;

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

	// 采样器
    pcl::VoxelGrid<PointType> downSizeFilterScancontext;
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterOutlier;
    pcl::VoxelGrid<PointType> downSizeFilterHistoryKeyFrames;    // for histor key frames of loop closure
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;   // for global map visualization
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;  // for global map visualization

    // 时间戳
    double timeLaserCloudCornerLast;
    double timeLaserCloudSurfLast;
    double timeLaserOdometry;
    double timeLaserCloudOutlierLast;
    double timeLastGloalMapPublish;

    // 消息更新标志
    bool newLaserCloudCornerLast;
    bool newLaserCloudSurfLast;
    bool newLaserOdometry;
    bool newLaserCloudOutlierLast;

    float transformLast[6];

    /*************高频转换量**************/
    // 当前帧odometry计算得到的到世界坐标系下的转移矩阵
    float transformSum[6];
    // 前后两帧之间的位姿关系，只使用了后三个平移增量
    float transformIncre[6];

    /*************低频转换量*************/
    // 以起始位置为原点的世界坐标系下的转换矩阵（猜测与调整的对象）
    float transformTobeMapped[6];
    // 存放mapping之前的Odometry计算的世界坐标系的转换矩阵（注：低频量，不一定与transformSum一样）
    float transformBefMapped[6];
    // 存放mapping之后的经过mapping微调之后的转换矩阵
    float transformAftMapped[6];

    int imuPointerFront;
    int imuPointerLast;

    double imuTime[imuQueLength];
    float imuRoll[imuQueLength];
    float imuPitch[imuQueLength];

    std::mutex mtx;

    double timeLastProcessing; // 0.3s 表示每隔0.3s进行一次地图匹配

    PointType pointOri, pointSel, pointProj, coeff;

    cv::Mat matA0;
    cv::Mat matB0;
    cv::Mat matX0;

    cv::Mat matA1;
    cv::Mat matD1;
    cv::Mat matV1;

    // 场景退化标志
    bool isDegenerate;
    cv::Mat matP;

    int laserCloudCornerFromMapDSNum;
    int laserCloudSurfFromMapDSNum;
    int laserCloudCornerLastDSNum;
    int laserCloudSurfLastDSNum;
    int laserCloudOutlierLastDSNum;
    int laserCloudSurfTotalLastDSNum;

    bool potentialLoopFlag;                        // 闭环检测成功标志位
    double timeSaveFirstCurrentScanForLoopClosure; // 闭环帧时间戳
    int RSclosestHistoryFrameID;
    int SCclosestHistoryFrameID; // giseop
    int latestFrameIDLoopCloure; // 当前关键帧下标
    float yawDiffRad;

    bool aLoopIsClosed; // 是否打开闭环检测线程标志位

    float cRoll, sRoll, cPitch, sPitch, cYaw, sYaw, tX, tY, tZ;
    float ctRoll, stRoll, ctPitch, stPitch, ctYaw, stYaw, tInX, tInY, tInZ;

    // // loop detector
    SCManager scManager;

public:
    mapOptimization() : nh("~")
    {
        // 用于闭环图优化的参数设置，使用gtsam库
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        pubKeyPoses = nh.advertise<sensor_msgs::PointCloud2>("/key_pose_origin", 2);
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 2);
        pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 5);

        // 以下几个回调函数，接收原始点云、面点、角点、离群点、以及里程计位姿保存在成员变量中
        // 这里的原始的点云，是没有经过分割和提取的，雷达直接发出来的
        subLaserCloudRaw = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 2, &mapOptimization::laserCloudRawHandler, this);
        subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2, &mapOptimization::laserCloudCornerLastHandler, this);
        subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2, &mapOptimization::laserCloudSurfLastHandler, this);
        subOutlierCloudLast = nh.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2, &mapOptimization::laserCloudOutlierLastHandler, this);
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, &mapOptimization::laserOdometryHandler, this);
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 50, &mapOptimization::imuHandler, this);

        pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/history_cloud", 2);
        pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/corrected_cloud", 2);
        pubRecentKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("/recent_cloud", 2);
        pubRegisteredCloud = nh.advertise<sensor_msgs::PointCloud2>("/registered_cloud", 2);

        float filter_size;
		// 设置滤波时创建的体素大小为0.2m/0.4m立方体，下面的单位为m
        downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);
        filter_size = 0.5;
        downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);
        filter_size = 0.3;
        downSizeFilterSurf.setLeafSize(filter_size, filter_size, filter_size); // default 0.4;
        downSizeFilterOutlier.setLeafSize(0.4, 0.4, 0.4);

        filter_size = 0.3;
        downSizeFilterHistoryKeyFrames.setLeafSize(filter_size, filter_size, filter_size); // default 0.4; for histor key frames of loop closure
        filter_size = 1.0;
        downSizeFilterSurroundingKeyPoses.setLeafSize(filter_size, filter_size, filter_size); // default 1; for surrounding key poses of scan-to-map optimization

        downSizeFilterGlobalMapKeyPoses.setLeafSize(1.0, 1.0, 1.0);  // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4); // for global map visualization

        odomAftMapped.header.frame_id = "/camera_init";
        odomAftMapped.child_frame_id = "/aft_mapped";

        aftMappedTrans.frame_id_ = "/camera_init";
        aftMappedTrans.child_frame_id_ = "/aft_mapped";

        allocateMemory(); // 成员变量内存分配
    }

    void allocateMemory()
    {
        // 关键帧位姿
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        surroundingKeyPoses.reset(new pcl::PointCloud<PointType>());
        surroundingKeyPosesDS.reset(new pcl::PointCloud<PointType>());

        laserCloudRaw.reset(new pcl::PointCloud<PointType>());             // corner feature set from odoOptimization
        laserCloudRawDS.reset(new pcl::PointCloud<PointType>());           // corner feature set from odoOptimization
        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());      // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());        // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>());    // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>());      // downsampled surf featuer set from odoOptimization
        laserCloudOutlierLast.reset(new pcl::PointCloud<PointType>());     // corner feature set from odoOptimization
        laserCloudOutlierLastDS.reset(new pcl::PointCloud<PointType>());   // downsampled corner feature set from odoOptimization
        laserCloudSurfTotalLast.reset(new pcl::PointCloud<PointType>());   // surf feature set from odoOptimization
        laserCloudSurfTotalLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        nearHistoryCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistoryCornerKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());
        SCnearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        SCnearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        latestCornerKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        SClatestSurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        latestSurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        RSlatestSurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>()); // giseop
        RSnearHistorySurfKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        RSnearHistorySurfKeyFrameCloudDS.reset(new pcl::PointCloud<PointType>());

        kdtreeGlobalMap.reset(new pcl::KdTreeFLANN<PointType>());
        globalMapKeyPoses.reset(new pcl::PointCloud<PointType>());
        globalMapKeyPosesDS.reset(new pcl::PointCloud<PointType>());
        globalMapKeyFrames.reset(new pcl::PointCloud<PointType>());
        globalMapKeyFramesDS.reset(new pcl::PointCloud<PointType>());

        timeLaserCloudCornerLast = 0;
        timeLaserCloudSurfLast = 0;
        timeLaserOdometry = 0;
        timeLaserCloudOutlierLast = 0;
        timeLastGloalMapPublish = 0;

        timeLastProcessing = -1;

        newLaserCloudCornerLast = false;
        newLaserCloudSurfLast = false;

        newLaserOdometry = false;
        newLaserCloudOutlierLast = false;

        for (int i = 0; i < 6; ++i)
        {
            transformLast[i] = 0;
            transformSum[i] = 0;
            transformIncre[i] = 0;
            transformTobeMapped[i] = 0;
            transformBefMapped[i] = 0;
            transformAftMapped[i] = 0;
        }

        imuPointerFront = 0;
        imuPointerLast = -1;

        for (int i = 0; i < imuQueLength; ++i)
        {
            imuTime[i] = 0;
            imuRoll[i] = 0;
            imuPitch[i] = 0;
        }

        gtsam::Vector Vector6(6);
        Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;
        priorNoise = noiseModel::Diagonal::Variances(Vector6);
        odometryNoise = noiseModel::Diagonal::Variances(Vector6);

        matA0 = cv::Mat(5, 3, CV_32F, cv::Scalar::all(0));
        matB0 = cv::Mat(5, 1, CV_32F, cv::Scalar::all(-1));
        matX0 = cv::Mat(3, 1, CV_32F, cv::Scalar::all(0));

        // matA1为边缘特征的协方差矩阵
        matA1 = cv::Mat(3, 3, CV_32F, cv::Scalar::all(0));
        // matA1的特征值
        matD1 = cv::Mat(1, 3, CV_32F, cv::Scalar::all(0));
        // matA1的特征向量，对应于matD1存储
        matV1 = cv::Mat(3, 3, CV_32F, cv::Scalar::all(0));

        isDegenerate = false;
        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));

        laserCloudCornerFromMapDSNum = 0;
        laserCloudSurfFromMapDSNum = 0;
        laserCloudCornerLastDSNum = 0;
        laserCloudSurfLastDSNum = 0;
        laserCloudOutlierLastDSNum = 0;
        laserCloudSurfTotalLastDSNum = 0;

        potentialLoopFlag = false;
        aLoopIsClosed = false;

        latestFrameID = 0;
    }

    // 将坐标转移到世界坐标系下,得到可用于建图的Lidar坐标，即修改了transformTobeMapped的值
    void transformAssociateToMap()
    {
        /*
            R = R_yaw * R_pitch * R_roll
              = R_ry * R_rx * R_rz
              = |cry 0 sry|    |1 0 0|        |crz -srz 0|
                |0 1 0|      * |0 crx -srx| * |srz crz 0|
                |-sry 0 cry|   |0 srx crx|    |0 0 1|
              = |crycrz+srxsrysrz  srxsrycrz-crysrz  crxsry|
                |crxsrz  crxcrz  -srx|
                |srxcrysrz-srycrz  srxcrycrz+srysrz  crxcry|

            R_sum = R_bef * R_incre^-1
            T_sum = T_bef - R_sum * T_incre

            所以  T_incre = R_sum^-1 * (T_bef - T_sum)
                  R_incre^-1 = R_bef^-1 * R_sum

            R_tobe = R_aft * R_incre^-1 = R_aft * R_bef^-1 * R_sum
            T_tobe = T_aft - R_tobe * T_incre
        */

        // yaw逆时针旋转
        float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) - sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
        float y1 = transformBefMapped[4] - transformSum[4];
        float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3]) + cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

        // pitch逆时针旋转
        float x2 = x1;
        float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
        float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;

        // roll逆时针旋转
        transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;
        transformIncre[4] = -sin(transformSum[2]) * x2 + cos(transformSum[2]) * y2;
        transformIncre[5] = z2;

        float sbcx = sin(transformSum[0]);
        float cbcx = cos(transformSum[0]);
        float sbcy = sin(transformSum[1]);
        float cbcy = cos(transformSum[1]);
        float sbcz = sin(transformSum[2]);
        float cbcz = cos(transformSum[2]);

        float sblx = sin(transformBefMapped[0]);
        float cblx = cos(transformBefMapped[0]);
        float sbly = sin(transformBefMapped[1]);
        float cbly = cos(transformBefMapped[1]);
        float sblz = sin(transformBefMapped[2]);
        float cblz = cos(transformBefMapped[2]);

        float salx = sin(transformAftMapped[0]);
        float calx = cos(transformAftMapped[0]);
        float saly = sin(transformAftMapped[1]);
        float caly = cos(transformAftMapped[1]);
        float salz = sin(transformAftMapped[2]);
        float calz = cos(transformAftMapped[2]);

        //R_transformTobe = R_transformAft * R_transformBef^-1 * R_transformSum
        /*
            R = |crycrz+srxsrysrz  srxsrycrz-crysrz  crxsry|
                |crxsrz  crxcrz  -srx|
                |srxcrysrz-srycrz  srxcrycrz+srysrz  crxcry|
                
            R_transformAft = |calycalz+salxsalysalz  salxsalycalz-calysalz  calxsaly|
                             |calxsalz  calxcalz  -salx|
                             |salycalysalz-salycalz  salxcalycalz+salysalz  calxcaly|
            R_transformBef^-1 = |cblycblz+sblxsblysblz  cblxsblz  sblxcblysblz-sblycblz|
                                |sblxsblycblz-cblysblz  cblxcblz  sblxcblycblz+sblysblz|
                                |cblxsbly  -sblx  cblxcbly|
            R_transformSum = |cbcycbcz+sbcxsbcysbcz  sbcxsbcysbcz-cbcysbcz  cbcxsbcy|
                             |cbcxsbcz  cbcxcbcz  -sbcx|
                             |sbcxcbcysbcz-sbcycbcz  sbcxcbcycbcz+sbcysbcz  cbcxcbcy|
        */

        /*
            srx = (calxsalz*(cblycblz+sblxsblysblz) + calxcalz*(sblxsblycblz-cblysblz) + (-salx)*(cblxsbly))*cbcxsbcy
                + (calxsalz*(cblxsblz) + (calxcalz)*(cblxcblz) + (-salx)*(-sblx))*(-sbcx)
                + (calxsalz*(sblxcblysblz-sblycblz) + (calxcalz)*(sblxcblycblz+sblysblz) + (-salx)*(cblxcbly))*cbcxcbcy

                = -sbcx*(calxsalz*cblxsblz + calxcalz*cblxcblz + salxsblx)
                + cbcxsbcy*(calxcalz*(sblxsblycblz-cblysblz) + calxsalz*(cblycblz+sblxsblysblz) + (-salx)*(cblxsbly))
                + cbcxcbcy*(calxsalz*(sblxcblysblz-sblycblz) + calxcalz*(sblxcblycblz+sblysblz) + (-salx)*(cblxcbly))

                = -sbcx*(salxsblx + calxcblxsalzsblz + calxcalzcblxcblz)
                - cbcxsbcy*(calxsalz*(cblysblz-sblxsblycblz) - calxsalz*(cblycblz+sblxsblysblz) + salxcblxsbly)
                - cbcxcbcy*(calxsalz*(sbly-cblz-sblxcblysblz) - calxcalz*(sblxcblycblz+sblysblz) + salxcblxcbly)
        */
        float srx = -sbcx * (salx * sblx + calx * cblx * salz * sblz + calx * calz * cblx * cblz) - cbcx * sbcy * (calx * calz * (cbly * sblz - cblz * sblx * sbly) - calx * salz * (cbly * cblz + sblx * sbly * sblz) + cblx * salx * sbly) - cbcx * cbcy * (calx * salz * (cblz * sbly - cbly * sblx * sblz) - calx * calz * (sbly * sblz + cbly * cblz * sblx) + cblx * cbly * salx);
        transformTobeMapped[0] = -asin(srx);

        float srycrx = sbcx * (cblx * cblz * (caly * salz - calz * salx * saly) - cblx * sblz * (caly * calz + salx * saly * salz) + calx * saly * sblx) - cbcx * cbcy * ((caly * calz + salx * saly * salz) * (cblz * sbly - cbly * sblx * sblz) + (caly * salz - calz * salx * saly) * (sbly * sblz + cbly * cblz * sblx) - calx * cblx * cbly * saly) + cbcx * sbcy * ((caly * calz + salx * saly * salz) * (cbly * cblz + sblx * sbly * sblz) + (caly * salz - calz * salx * saly) * (cbly * sblz - cblz * sblx * sbly) + calx * cblx * saly * sbly);
        float crycrx = sbcx * (cblx * sblz * (calz * saly - caly * salx * salz) - cblx * cblz * (saly * salz + caly * calz * salx) + calx * caly * sblx) + cbcx * cbcy * ((saly * salz + caly * calz * salx) * (sbly * sblz + cbly * cblz * sblx) + (calz * saly - caly * salx * salz) * (cblz * sbly - cbly * sblx * sblz) + calx * caly * cblx * cbly) - cbcx * sbcy * ((saly * salz + caly * calz * salx) * (cbly * sblz - cblz * sblx * sbly) + (calz * saly - caly * salx * salz) * (cbly * cblz + sblx * sbly * sblz) - calx * caly * cblx * sbly);
        transformTobeMapped[1] = atan2(srycrx / cos(transformTobeMapped[0]),
                                       crycrx / cos(transformTobeMapped[0]));

        float srzcrx = (cbcz * sbcy - cbcy * sbcx * sbcz) * (calx * salz * (cblz * sbly - cbly * sblx * sblz) - calx * calz * (sbly * sblz + cbly * cblz * sblx) + cblx * cbly * salx) - (cbcy * cbcz + sbcx * sbcy * sbcz) * (calx * calz * (cbly * sblz - cblz * sblx * sbly) - calx * salz * (cbly * cblz + sblx * sbly * sblz) + cblx * salx * sbly) + cbcx * sbcz * (salx * sblx + calx * cblx * salz * sblz + calx * calz * cblx * cblz);
        float crzcrx = (cbcy * sbcz - cbcz * sbcx * sbcy) * (calx * calz * (cbly * sblz - cblz * sblx * sbly) - calx * salz * (cbly * cblz + sblx * sbly * sblz) + cblx * salx * sbly) - (sbcy * sbcz + cbcy * cbcz * sbcx) * (calx * salz * (cblz * sbly - cbly * sblx * sblz) - calx * calz * (sbly * sblz + cbly * cblz * sblx) + cblx * cbly * salx) + cbcx * cbcz * (salx * sblx + calx * cblx * salz * sblz + calx * calz * cblx * cblz);
        transformTobeMapped[2] = atan2(srzcrx / cos(transformTobeMapped[0]),
                                       crzcrx / cos(transformTobeMapped[0]));

        /*
            T_tobe = T_aft - R_tobe * T_incre
        */
        // roll旋转
        x1 = cos(transformTobeMapped[2]) * transformIncre[3] - sin(transformTobeMapped[2]) * transformIncre[4];
        y1 = sin(transformTobeMapped[2]) * transformIncre[3] + cos(transformTobeMapped[2]) * transformIncre[4];
        z1 = transformIncre[5];

        // pitch旋转
        x2 = x1;
        y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
        z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

        // yaw旋转
        transformTobeMapped[3] = transformAftMapped[3] - (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
        transformTobeMapped[4] = transformAftMapped[4] - y2;
        transformTobeMapped[5] = transformAftMapped[5] - (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);
    }

    // 记录odometry发送的转换矩阵与mapping之后的转换矩阵，下一帧点云会使用(有IMU的话会使用IMU进行补偿)
    void transformUpdate()
    {
        // 此时transformTobeMapped已经经过LM优化过.
        if (imuPointerLast >= 0)
        {
            float imuRollLast = 0, imuPitchLast = 0;
            // 寻找是否有点云的时间戳小于IMU的时间戳的IMU位置:imuPointerFront
            while (imuPointerFront != imuPointerLast)
            {
                if (timeLaserOdometry + scanPeriod < imuTime[imuPointerFront])
                {
                    break;
                }

                imuPointerFront = (imuPointerFront + 1) % imuQueLength;
            }

            // 没找到,此时imuPointerFront==imtPointerLast,只能以当前收到的最新的IMU的欧拉角使用
            if (timeLaserOdometry + scanPeriod > imuTime[imuPointerFront])
            {
                imuRollLast = imuRoll[imuPointerFront];
                imuPitchLast = imuPitch[imuPointerFront];
            }
            else
            {
                // 在imu数据充足的情况下可以进行插补
                int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
                float ratioFront = (timeLaserOdometry + scanPeriod - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                float ratioBack = (imuTime[imuPointerFront] - timeLaserOdometry - scanPeriod) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

                imuRollLast = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
                imuPitchLast = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
            }

            // imu稍微补偿俯仰角和翻滚角
            transformTobeMapped[0] = 0.998 * transformTobeMapped[0] + 0.002 * imuPitchLast;
            transformTobeMapped[2] = 0.998 * transformTobeMapped[2] + 0.002 * imuRollLast;
        }

        for (int i = 0; i < 6; i++)
        {
            transformBefMapped[i] = transformSum[i];        // 将当前保存的Lidar_odometry(存在sum中),赋值给BefMapped
            transformAftMapped[i] = transformTobeMapped[i]; // 将优化后并且经过IMU微调后的transform保存到Aft中.用于下次进行预测
        }
    }

    void updatePointAssociateToMapSinCos()
    {
        // 先提前求好roll,pitch,yaw的sin和cos值
        cRoll = cos(transformTobeMapped[0]);
        sRoll = sin(transformTobeMapped[0]);

        cPitch = cos(transformTobeMapped[1]);
        sPitch = sin(transformTobeMapped[1]);

        cYaw = cos(transformTobeMapped[2]);
        sYaw = sin(transformTobeMapped[2]);

        tX = transformTobeMapped[3];
        tY = transformTobeMapped[4];
        tZ = transformTobeMapped[5];
    }

    // 第i帧的点转换到第一帧坐标系下
    void pointAssociateToMap(PointType const *const pi, PointType *const po)
    {
        // 进行6自由度的变换，先进行旋转，然后再平移
        // 主要进行坐标变换，将局部坐标转换到全局坐标中去

        /*
            X0 = R_yaw * R_pitch * R_roll * Xi + T
               = R_ry * R_rx * R_rz * Xi + T
        */

        // 先绕z轴旋转
        //     |cosrz  -sinrz  0|
        //  Rz=|sinrz  cosrz   0|
        //     |0       0      1|
        // [x1,y1,z1]^T=Rz*[pi->x,pi->y,pi->z]
        float x1 = cYaw * pi->x - sYaw * pi->y;
        float y1 = sYaw * pi->x + cYaw * pi->y;
        float z1 = pi->z;

        // [x2,y2,z2]^T=Rx*[x1,y1,z1]
        //    |1     0        0|
        // Rx=|0   cosrx -sinrx|
        //    |0   sinrx  cosrx|
        float x2 = x1;
        float y2 = cRoll * y1 - sRoll * z1;
        float z2 = sRoll * y1 + cRoll * z1;

        // 最后再绕Y轴旋转，然后加上平移
        //    |cosry   0   sinry|
        // Ry=|0       1       0|
        //    |-sinry  0   cosry|
        po->x = cPitch * x2 + sPitch * z2 + tX;
        po->y = y2 + tY;
        po->z = -sPitch * x2 + cPitch * z2 + tZ;
        po->intensity = pi->intensity;
    }

    // 将点云转换到世界坐标系下的Transform
    void updateTransformPointCloudSinCos(PointTypePose *tIn)
    {
        ctRoll = cos(tIn->roll);
        stRoll = sin(tIn->roll);

        ctPitch = cos(tIn->pitch);
        stPitch = sin(tIn->pitch);

        ctYaw = cos(tIn->yaw);
        stYaw = sin(tIn->yaw);

        tInX = tIn->x;
        tInY = tIn->y;
        tInZ = tIn->z;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn)
    {
        // !!! DO NOT use pcl for point cloud transformation, results are not accurate
        // Reason: unkown
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        PointType pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);

        for (int i = 0; i < cloudSize; ++i)
        {
            pointFrom = &cloudIn->points[i];
            float x1 = ctYaw * pointFrom->x - stYaw * pointFrom->y;
            float y1 = stYaw * pointFrom->x + ctYaw * pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = ctRoll * y1 - stRoll * z1;
            float z2 = stRoll * y1 + ctRoll * z1;

            pointTo.x = ctPitch * x2 + stPitch * z2 + tInX;
            pointTo.y = y2 + tInY;
            pointTo.z = -stPitch * x2 + ctPitch * z2 + tInZ;
            pointTo.intensity = pointFrom->intensity;

            cloudOut->points[i] = pointTo;
        }

        return cloudOut;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose *transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;
        PointType pointTo;

        int cloudSize = cloudIn->points.size();
        cloudOut->resize(cloudSize);

        // 坐标系变换，旋转rpy角
        for (int i = 0; i < cloudSize; ++i)
        {
            pointFrom = &cloudIn->points[i];
            float x1 = cos(transformIn->yaw) * pointFrom->x - sin(transformIn->yaw) * pointFrom->y;
            float y1 = sin(transformIn->yaw) * pointFrom->x + cos(transformIn->yaw) * pointFrom->y;
            float z1 = pointFrom->z;

            float x2 = x1;
            float y2 = cos(transformIn->roll) * y1 - sin(transformIn->roll) * z1;
            float z2 = sin(transformIn->roll) * y1 + cos(transformIn->roll) * z1;

            pointTo.x = cos(transformIn->pitch) * x2 + sin(transformIn->pitch) * z2 + transformIn->x;
            pointTo.y = y2 + transformIn->y;
            pointTo.z = -sin(transformIn->pitch) * x2 + cos(transformIn->pitch) * z2 + transformIn->z;
            pointTo.intensity = pointFrom->intensity;

            cloudOut->points[i] = pointTo;
        }

        return cloudOut;
    }

    // 外点回调函数
    void laserCloudOutlierLastHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        timeLaserCloudOutlierLast = msg->header.stamp.toSec(); // 记录接收外点点云时间戳
        laserCloudOutlierLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudOutlierLast);
        newLaserCloudOutlierLast = true; // 设置接收标志位
    }

    void laserCloudRawHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        laserCloudRaw->clear();
        pcl::fromROSMsg(*msg, *laserCloudRaw);
    }
	
    // 次边缘点回调函数
    void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        timeLaserCloudCornerLast = msg->header.stamp.toSec();
        laserCloudCornerLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudCornerLast);
        newLaserCloudCornerLast = true;
    }

    // 次平面点回调函数
    void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        timeLaserCloudSurfLast = msg->header.stamp.toSec();
        laserCloudSurfLast->clear();
        pcl::fromROSMsg(*msg, *laserCloudSurfLast);
        newLaserCloudSurfLast = true;
    }

    // 里程计回调函数
    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
    {
        timeLaserOdometry = laserOdometry->header.stamp.toSec(); // 记录时间戳
        double roll, pitch, yaw;
        geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation; // 接收到的数据z朝前,x朝左,y朝上坐标系
        tf::Matrix3x3(tf::Quaternion(geoQuat.z, -geoQuat.x, -geoQuat.y, geoQuat.w)).getRPY(roll, pitch, yaw);
        transformSum[0] = -pitch;
        transformSum[1] = -yaw;
        transformSum[2] = roll;
        transformSum[3] = laserOdometry->pose.pose.position.x;
        transformSum[4] = laserOdometry->pose.pose.position.y;
        transformSum[5] = laserOdometry->pose.pose.position.z;
        newLaserOdometry = true;
    }

    // imu
    void imuHandler(const sensor_msgs::Imu::ConstPtr &imuIn)
    {
        double roll, pitch, yaw;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(imuIn->orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        imuPointerLast = (imuPointerLast + 1) % imuQueLength; // 移动到下一个位置
        imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
        imuRoll[imuPointerLast] = roll;
        imuPitch[imuPointerLast] = pitch;
    }

    /* rz,rx,ry分别对应着标准右手坐标系中的roll,pitch,yaw角,通过查看createQuaternionMsgFromRollPitchYaw()的函数定义可以发现.
     * 当pitch和yaw角给负值后,四元数中的y和z会变成负值,x和w不受影响.由四元数定义可以知道,x,y,z是指旋转轴在三个轴上的投影,w影响
     * 旋转角度,所以由createQuaternionMsgFromRollPitchYaw()计算得到四元数后,其在一般右手坐标系中的x,y,z分量对应到该应用场景下
     * 的坐标系中,geoQuat.x对应实际坐标系下的z轴分量,geoQuat.y对应x轴分量,geoQuat.z对应实际的y轴分量,而由于rx和ry在计算四元数
     * 时给的是负值,所以geoQuat.y和geoQuat.z取负值,这样就等于没变
    */
    void publishTF()
    {
        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);

        odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        odomAftMapped.pose.pose.orientation.x = -geoQuat.y;
        odomAftMapped.pose.pose.orientation.y = -geoQuat.z;
        odomAftMapped.pose.pose.orientation.z = geoQuat.x;
        odomAftMapped.pose.pose.orientation.w = geoQuat.w;
        odomAftMapped.pose.pose.position.x = transformAftMapped[3];
        odomAftMapped.pose.pose.position.y = transformAftMapped[4];
        odomAftMapped.pose.pose.position.z = transformAftMapped[5];
        odomAftMapped.twist.twist.angular.x = transformBefMapped[0];
        odomAftMapped.twist.twist.angular.y = transformBefMapped[1];
        odomAftMapped.twist.twist.angular.z = transformBefMapped[2];
        odomAftMapped.twist.twist.linear.x = transformBefMapped[3];
        odomAftMapped.twist.twist.linear.y = transformBefMapped[4];
        odomAftMapped.twist.twist.linear.z = transformBefMapped[5];
        pubOdomAftMapped.publish(odomAftMapped);

        aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
        aftMappedTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        aftMappedTrans.setOrigin(tf::Vector3(transformAftMapped[3], transformAftMapped[4], transformAftMapped[5]));
        tfBroadcaster.sendTransform(aftMappedTrans);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw = transformIn[2];

        return thisPose6D;
    }

    void publishKeyPosesAndFrames()
    {
        if (pubKeyPoses.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudKeyPoses3D, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubKeyPoses.publish(cloudMsgTemp);
        }

        if (pubRecentKeyFrames.getNumSubscribers() != 0)
        {
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*laserCloudSurfFromMapDS, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubRecentKeyFrames.publish(cloudMsgTemp);
        }

        if (pubRegisteredCloud.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS, &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfTotalLast, &thisPose6D);

            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudOut, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/camera_init";
            pubRegisteredCloud.publish(cloudMsgTemp);
        }
    }

    // 用于显示
    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2); // 0.2HZ
        while (ros::ok())
        {
            rate.sleep();
            publishGlobalMap();
        }

        // save final point cloud
        pcl::io::savePCDFileASCII(fileDirectory + "finalCloud.pcd", *globalMapKeyFramesDS);

        string cornerMapString = "/tmp/cornerMap.pcd";
        string surfaceMapString = "/tmp/surfaceMap.pcd";
        string trajectoryString = "/tmp/trajectory.pcd";

        pcl::PointCloud<PointType>::Ptr cornerMapCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr cornerMapCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceMapCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceMapCloudDS(new pcl::PointCloud<PointType>());

        for (int i = 0; i < cornerCloudKeyFrames.size(); i++)
        {
            *cornerMapCloud += *transformPointCloud(cornerCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
            *surfaceMapCloud += *transformPointCloud(surfCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
            *surfaceMapCloud += *transformPointCloud(outlierCloudKeyFrames[i], &cloudKeyPoses6D->points[i]);
        }

        downSizeFilterCorner.setInputCloud(cornerMapCloud);
        downSizeFilterCorner.filter(*cornerMapCloudDS);
        downSizeFilterSurf.setInputCloud(surfaceMapCloud);
        downSizeFilterSurf.filter(*surfaceMapCloudDS);

        pcl::io::savePCDFileASCII(fileDirectory + "cornerMap.pcd", *cornerMapCloudDS);
        pcl::io::savePCDFileASCII(fileDirectory + "surfaceMap.pcd", *surfaceMapCloudDS);
        pcl::io::savePCDFileASCII(fileDirectory + "trajectory.pcd", *cloudKeyPoses3D);
    }

    // 显示地图
    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true) // 轨迹点
            return;

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        // 通过KDTree进行最近邻搜索
        kdtreeGlobalMap->radiusSearch(currentRobotPosPoint, globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);

        // downsample near selected key frames
        // 对globalMapKeyPoses进行下采样
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        // extract visualized and downsampled key frames
        for (int i = 0; i < globalMapKeyPosesDS->points.size(); ++i)
        {
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(outlierCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        }

        // downsample visualized points
        // 对globalMapKeyFrames进行下采样
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);

        sensor_msgs::PointCloud2 cloudMsgTemp;
        pcl::toROSMsg(*globalMapKeyFramesDS, cloudMsgTemp);
        cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        cloudMsgTemp.header.frame_id = "/camera_init";
        pubLaserCloudSurround.publish(cloudMsgTemp);

        globalMapKeyPoses->clear();
        globalMapKeyPosesDS->clear();
        globalMapKeyFrames->clear();
        // globalMapKeyFramesDS->clear();
    }

    // 回环检测线程
    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false) // 检测是否开启回环
            return;

        ros::Rate rate(1); // 1HZ
        while (ros::ok())
        {
            rate.sleep();
            performLoopClosure();
        }
    }

    // 回环检测
    // 检测与最新帧可以形成回环的关键帧,两种方法:
    // 1/ 根据几何距离,在一定半径范围内,30s以上,最早的那个帧
    // 2/ ScanContext,确定相似度最高的关键帧
    bool detectLoopClosure()
    {

        // 资源分配时初始化
        // 在互斥量被析构前不解锁
        std::lock_guard<std::mutex> lock(mtx);

        /*
         * 邻域搜索闭环
         * 1. xyz distance-based radius search (contained in the original LeGO LOAM code)
         * - for fine-stichting trajectories (for not-recognized nodes within scan context search) 
         */
        // 基于目前位姿,在一定范围内(20m)内搜索最近邻,若最早的那个超过了30s,则选中为回环目标
        // 选取前后25帧组成点云,并保存当前最近一帧点云
        RSlatestSurfKeyFrameCloud->clear(); // 当前关键帧
        RSnearHistorySurfKeyFrameCloud->clear(); // 尝试进行回环的关键帧前后一定范围帧组成的点云
        RSnearHistorySurfKeyFrameCloudDS->clear(); // 上面的降采样

        // find the closest history key frame
        // kdtree查找当前帧25m之内的关键帧
        std::vector<int> pointSearchIndLoop; // 搜索完的邻域点对应的索引
        std::vector<float> pointSearchSqDisLoop; // 搜索完的邻域点与当前点的欧氏距离
        kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D); // 用当前的所有关键帧生成树
        // 进行半径historyKeyframeSearchRadius内的邻域搜索，
        // currentRobotPosPoint：需要查询的点，
        // pointSearchIndLoop：搜索完的邻域点对应的索引
        // pointSearchSqDisLoop：搜索完的每个邻域点与当前点之间的欧式距离
        // 0：返回的邻域个数，为0表示返回全部的邻域点
        kdtreeHistoryKeyPoses->radiusSearch(currentRobotPosPoint, historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

        // 选取最近邻中,时间距离30s以上,最早的那帧
        RSclosestHistoryFrameID = -1;
        int curMinID = 1000000;
        // policy: take Oldest one (to fix error of the whole trajectory)
        for (int i = 0; i < pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            // 时间差值大于30s, 认为是闭环
            if (abs(cloudKeyPoses6D->points[id].time - timeLaserOdometry) > 30.0)
            {
                // RSclosestHistoryFrameID = id;
                // break;
                if (id < curMinID)
                {
                    curMinID = id;
                    RSclosestHistoryFrameID = curMinID;
                }
            }
        }

        if (RSclosestHistoryFrameID == -1)
        {
            // Do nothing here
            // then, do the next check: Scan context-based search
            // not return false here;
        }
        else
        {
            // save latest key frames
        	// 回环检测的进程是单独进行的,因此这里需要确定最新帧
            // 检测到回环了会保存四种点云
            latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
        	// 根据当前的位姿,对点云进行旋转和平移
            // 点云的xyz坐标进行坐标系变换(分别绕xyz轴旋转)
            *RSlatestSurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
            *RSlatestSurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[latestFrameIDLoopCloure]);
            pcl::PointCloud<PointType>::Ptr RShahaCloud(new pcl::PointCloud<PointType>());
            int cloudSize = RSlatestSurfKeyFrameCloud->points.size();
            for (int i = 0; i < cloudSize; ++i)
            {
            	// intensity不小于0的点放进hahaCloud队列
            	// 初始化时intensity是-1，滤掉那些点
                if ((int)RSlatestSurfKeyFrameCloud->points[i].intensity >= 0)
                {
                    RShahaCloud->push_back(RSlatestSurfKeyFrameCloud->points[i]);
                }
            }
            RSlatestSurfKeyFrameCloud->clear();
            *RSlatestSurfKeyFrameCloud = *RShahaCloud;

            // 保存一定范围内最早的那帧前后25帧的点,并在对应位姿处投影后进行合并
            // save history near key frames
        	// historyKeyframeSearchNum在utility.h中定义为25，前后25个点进行变换
            for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j)
            {
                if (RSclosestHistoryFrameID + j < 0 || RSclosestHistoryFrameID + j > latestFrameIDLoopCloure)
                    continue;

                *RSnearHistorySurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[RSclosestHistoryFrameID + j], &cloudKeyPoses6D->points[RSclosestHistoryFrameID + j]);
                *RSnearHistorySurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[RSclosestHistoryFrameID + j], &cloudKeyPoses6D->points[RSclosestHistoryFrameID + j]);
            }

        	// 下采样滤波减少数据量
            downSizeFilterHistoryKeyFrames.setInputCloud(RSnearHistorySurfKeyFrameCloud);
            downSizeFilterHistoryKeyFrames.filter(*RSnearHistorySurfKeyFrameCloudDS);
        }

        /* 
         * 2. Scan context-based global localization 
         */
        SClatestSurfKeyFrameCloud->clear();
        SCnearHistorySurfKeyFrameCloud->clear();
        SCnearHistorySurfKeyFrameCloudDS->clear();

        // std::lock_guard<std::mutex> lock(mtx);
        latestFrameIDLoopCloure = cloudKeyPoses3D->points.size() - 1;
        SCclosestHistoryFrameID = -1;                        // init with -1
        
        // 这是最重要的部分，根据ScanContext确定回环的关键帧,返回的是关键帧的ID,和yaw角的偏移量
        auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff
        SCclosestHistoryFrameID = detectResult.first;
        yawDiffRad = detectResult.second; // not use for v1 (because pcl icp withi initial somthing wrong...)

        // if all close, reject
        if (SCclosestHistoryFrameID == -1)
        {
            return false;
        }

        // 以下，如果SC检测到了回环，保存回环上的帧前后25帧和当前帧，过程与上面完全一样
        // save latest key frames: query ptcloud (corner points + surface points)
        // NOTE: using "closestHistoryFrameID" to make same root of submap points to get a direct relative between the query point cloud (latestSurfKeyFrameCloud) and the map (nearHistorySurfKeyFrameCloud). by giseop
        // i.e., set the query point cloud within mapside's local coordinate
        *SClatestSurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[SCclosestHistoryFrameID]);
        *SClatestSurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[latestFrameIDLoopCloure], &cloudKeyPoses6D->points[SCclosestHistoryFrameID]);

        pcl::PointCloud<PointType>::Ptr SChahaCloud(new pcl::PointCloud<PointType>());
        int cloudSize = SClatestSurfKeyFrameCloud->points.size();
        for (int i = 0; i < cloudSize; ++i)
        {
            if ((int)SClatestSurfKeyFrameCloud->points[i].intensity >= 0)
            {
                SChahaCloud->push_back(SClatestSurfKeyFrameCloud->points[i]);
            }
        }
        SClatestSurfKeyFrameCloud->clear();
        *SClatestSurfKeyFrameCloud = *SChahaCloud;

        // ScanContext确定的回环关键帧,前后一段范围内的点组成点云地图
        // save history near key frames: map ptcloud (icp to query ptcloud)
        for (int j = -historyKeyframeSearchNum; j <= historyKeyframeSearchNum; ++j)
        {
            if (SCclosestHistoryFrameID + j < 0 || SCclosestHistoryFrameID + j > latestFrameIDLoopCloure)
                continue;
            *SCnearHistorySurfKeyFrameCloud += *transformPointCloud(cornerCloudKeyFrames[SCclosestHistoryFrameID + j], &cloudKeyPoses6D->points[SCclosestHistoryFrameID + j]);
            *SCnearHistorySurfKeyFrameCloud += *transformPointCloud(surfCloudKeyFrames[SCclosestHistoryFrameID + j], &cloudKeyPoses6D->points[SCclosestHistoryFrameID + j]);
        }
        downSizeFilterHistoryKeyFrames.setInputCloud(SCnearHistorySurfKeyFrameCloud);
        downSizeFilterHistoryKeyFrames.filter(*SCnearHistorySurfKeyFrameCloudDS);

        // // optional: publish history near key frames
        // if (pubHistoryKeyFrames.getNumSubscribers() != 0){
        //     sensor_msgs::PointCloud2 cloudMsgTemp;
        //     pcl::toROSMsg(*nearHistorySurfKeyFrameCloudDS, cloudMsgTemp);
        //     cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        //     cloudMsgTemp.header.frame_id = "/camera_init";
        //     pubHistoryKeyFrames.publish(cloudMsgTemp);
        // }

        return true;
    } // detectLoopClosure

    // 回环优化
    void performLoopClosure()
    {
        // 没有关键帧，不用做回环优化
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        // try to find close key frame if there are any
        if (potentialLoopFlag == false)
        {
            // 回环检测，分别根据距离和ScanContext信息查找回环帧，回环信息保存在成员变量中，包括回环帧的ID、点云、偏航角等
            if (detectLoopClosure() == true)
            {
                std::cout << std::endl;
                potentialLoopFlag = true; // find some key frames that is old enough or close enough for loop closure
                timeSaveFirstCurrentScanForLoopClosure = timeLaserOdometry;
            }

            if (potentialLoopFlag == false) // ScanContext未能找到可以形成回环的关键帧
            {
                return;
            }
        }

        // reset the flag first no matter icp successes or not
        potentialLoopFlag = false;

        // 如果当前关键帧与历史关键帧确实形成了回环，开始进行优化

        // *****
        // Main
        // *****
        // make common variables at forward
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionCameraFrame;
        float noiseScore = 0.5; // constant is ok...
        gtsam::Vector Vector6(6);
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        constraintNoise = noiseModel::Diagonal::Variances(Vector6);
        robustNoiseModel = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure
            gtsam::noiseModel::Diagonal::Variances(Vector6)); // - checked it works. but with robust kernel, map modification may be delayed (i.e,. requires more true-positive loop factors)

        bool isValidRSloopFactor = false;
        bool isValidSCloopFactor = false;

        /*
         * 1. RS loop factor (radius search)
         * 将RS检测到的历史帧和当前帧匹配，求transform, 作为约束边
         */
        if (RSclosestHistoryFrameID != -1)
        {
            pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setMaxCorrespondenceDistance(100);
        	icp.setMaximumIterations(100);        // 最大迭代次数
        	icp.setTransformationEpsilon(1e-6);   // 设置收敛判断条件，越小精度越大，收敛也越慢 (前一个变换矩阵和当前变换矩阵的差异小于阈值时，就认为已经收敛)
        	icp.setEuclideanFitnessEpsilon(1e-6); // 还有一条收敛条件是均方误差和小于阈值，停止迭代。
        	// 设置RANSAC运行次数
            icp.setRANSACIterations(0);

            // Align clouds
            icp.setInputSource(RSlatestSurfKeyFrameCloud);
            icp.setInputTarget(RSnearHistorySurfKeyFrameCloudDS);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        	// 进行icp点云对齐
            icp.align(*unused_result);
            // 上面比较的两个点云都已经被投影到了世界坐标系下，所以匹配的结果应该是这段时间内，原点所发生的漂移

            // 通过score阈值判定icp是否匹配成功
            std::cout << "[RS] ICP fit score: " << icp.getFitnessScore() << std::endl;
            if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            {
                std::cout << "[RS] Reject this loop (bad icp fit score, > " << historyKeyframeFitnessScore << ")" << std::endl;
                isValidRSloopFactor = false;
            }
            else
            {
                std::cout << "[RS] The detected loop factor is added between Current [ " << latestFrameIDLoopCloure << " ] and RS nearest [ " << RSclosestHistoryFrameID << " ]" << std::endl;
                isValidRSloopFactor = true;
            }

            // 最新帧与回环帧前后一定时间范围内的点组成的地图进行匹配,得到的坐标变换为最新帧与回环帧之间的约束
            // 因为作为地图的帧在回环帧前后很小的范围内,位姿变化很小,可以认为他们之间的相对位姿很准,地图也很准
            // 这里RS检测成功，加入约束边
            if (isValidRSloopFactor == true)
            {
                correctionCameraFrame = icp.getFinalTransformation(); // 匹配结果，get transformation in camera frame (because points are in camera frame)
                pcl::getTranslationAndEulerAngles(correctionCameraFrame, x, y, z, roll, pitch, yaw); // 旋转矩阵转6个自由度
        		/*
            		x_map = y_lidar y_map = z_lidar z_map = x_lidar

            		X0 = R_correct * (R_累积 * Xi + T_累积) + T_correct
               		= R_correct * R_累积 * Xi + R_corrent * T_累积 + T_correct
               		= (correctionLidarFrame * tWorng) * Xi
        
            		R = R_yaw *  R_pitch * R_roll
              		= R_ry * R_rx * R_rz
        		*/
                Eigen::Affine3f correctionLidarFrame = pcl::getTransformation(z, x, y, yaw, roll, pitch); // 6个自由度转旋转矩阵
                // transform from world origin to wrong pose
                // 最新关键帧在地图坐标系中的坐标，在过程中会存在误差的积累，否则匹配的结果必然是E
                // 这种误差可以被解释为地图原点发生了漂移
                Eigen::Affine3f tWrong = pclPointToAffine3fCameraToLidar(cloudKeyPoses6D->points[latestFrameIDLoopCloure]); // 原始计算得到的位姿
                // transform from world origin to corrected pose
                // 地图原点的漂移×在漂移后的地图中的坐标=没有漂移的坐标，即在回环上的关键帧时刻其应该所处的位姿
                // 这样就把当前帧的位姿转移到了回环关键帧所在时刻，没有漂移的情况下的位姿，两者再求解相对位姿
                // 感觉以上很复杂，一开始完全没有把点云往世界坐标系投影啊！直接匹配不就是相对位姿么？
                Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong; // 真实的位姿，pre-multiplying -> successive rotation about a fixed frame
                pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw); // 旋转矩阵转6个自由度
                gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
                gtsam::Pose3 poseTo = pclPointTogtsamPose3(cloudKeyPoses6D->points[RSclosestHistoryFrameID]);
                gtsam::Vector Vector6(6);

                std::lock_guard<std::mutex> lock(mtx);
                gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, RSclosestHistoryFrameID, poseFrom.between(poseTo), robustNoiseModel));
                isam->update(gtSAMgraph);
                isam->update();
                gtSAMgraph.resize(0);
            }
        }

        /*
         * 2. SC loop factor (scan context)
         * SC检测成功，进行icp匹配
         */
        if (SCclosestHistoryFrameID != -1)
        {
            pcl::IterativeClosestPoint<PointType, PointType> icp;
            icp.setMaxCorrespondenceDistance(100);
            icp.setMaximumIterations(100);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
            icp.setRANSACIterations(0);

            // Align clouds
            // Eigen::Affine3f icpInitialMatFoo = pcl::getTransformation(0, 0, 0, yawDiffRad, 0, 0); // because within cam coord: (z, x, y, yaw, roll, pitch)
            // Eigen::Matrix4f icpInitialMat = icpInitialMatFoo.matrix();
            icp.setInputSource(SClatestSurfKeyFrameCloud);
            icp.setInputTarget(SCnearHistorySurfKeyFrameCloudDS);
            pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
            icp.align(*unused_result);
            // icp.align(*unused_result, icpInitialMat); // PCL icp non-eye initial is bad ... don't use (LeGO LOAM author also said pcl transform is weird.)

            std::cout << "[SC] ICP fit score: " << icp.getFitnessScore() << std::endl;
            if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            {
                std::cout << "[SC] Reject this loop (bad icp fit score, > " << historyKeyframeFitnessScore << ")" << std::endl;
                isValidSCloopFactor = false;
            }
            else
            {
                std::cout << "[SC] The detected loop factor is added between Current [ " << latestFrameIDLoopCloure << " ] and SC nearest [ " << SCclosestHistoryFrameID << " ]" << std::endl;
                isValidSCloopFactor = true;
            }

            // icp匹配成功也加入约束边
            if (isValidSCloopFactor == true)
            {
                correctionCameraFrame = icp.getFinalTransformation(); // get transformation in camera frame (because points are in camera frame)
                pcl::getTranslationAndEulerAngles(correctionCameraFrame, x, y, z, roll, pitch, yaw);
                gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
                gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

                std::lock_guard<std::mutex> lock(mtx);
                // gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, closestHistoryFrameID, poseFrom.between(poseTo), constraintNoise)); // original
                gtSAMgraph.add(BetweenFactor<Pose3>(latestFrameIDLoopCloure, SCclosestHistoryFrameID, poseFrom.between(poseTo), robustNoiseModel)); // giseop
                isam->update(gtSAMgraph);
                isam->update();
                gtSAMgraph.resize(0);
            }
        }

        // just for visualization
        // // publish corrected cloud
        // if (pubIcpKeyFrames.getNumSubscribers() != 0){
        //     pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
        //     pcl::transformPointCloud (*latestSurfKeyFrameCloud, *closed_cloud, icp.getFinalTransformation());
        //     sensor_msgs::PointCloud2 cloudMsgTemp;
        //     pcl::toROSMsg(*closed_cloud, cloudMsgTemp);
        //     cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        //     cloudMsgTemp.header.frame_id = "/camera_init";
        //     pubIcpKeyFrames.publish(cloudMsgTemp);
        // }

        // flagging
        aLoopIsClosed = true; // 回环结束

    } // performLoopClosure

    Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
		// camera frame to lidar frame
        return Pose3(Rot3::RzRyRx(double(thisPoint.yaw), double(thisPoint.roll), double(thisPoint.pitch)),
                     Point3(double(thisPoint.z), double(thisPoint.x), double(thisPoint.y)));
    }

    Eigen::Affine3f pclPointToAffine3fCameraToLidar(PointTypePose thisPoint)
    {
		// camera frame to lidar frame
        return pcl::getTransformation(thisPoint.z, thisPoint.x, thisPoint.y, thisPoint.yaw, thisPoint.roll, thisPoint.pitch);
    }

    // 构建局部小地图
    void extractSurroundingKeyFrames()
    {
        // 没有关键帧，不用构建小地图
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        /*
            有闭环检测和没有闭环检测局部小地图的构建方式不一样
        */

        // loopClosureEnableFlag 这个变量另外只在loopthread这部分中有用到
        if (loopClosureEnableFlag == true) // 有闭环检测构建方式
        {
            // only use recent key poses for graph building
            if (recentCornerCloudKeyFrames.size() < surroundingKeyframeSearchNum)
            {
                // queue is not full (the beginning of mapping or a loop is just closed)
                // clear recent key frames queue
                // recentCornerCloudKeyFrames保存的点云数量太少，则清空后重新塞入新的点云直至数量够
                // recentCornerCloudKeyFrames不足50帧点云
                recentCornerCloudKeyFrames.clear();
                recentSurfCloudKeyFrames.clear();
                recentOutlierCloudKeyFrames.clear();
                int numPoses = cloudKeyPoses3D->points.size(); // Poses3d为历史的lidar位置,只有位置,没有姿态
                for (int i = numPoses - 1; i >= 0; --i)
                {
                    // cloudKeyPoses3D的intensity中存的是索引值?
                    // 保存的索引值从1开始编号；
                    int thisKeyInd = (int)cloudKeyPoses3D->points[i].intensity;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd]; // Poses6D保存的历史的lidar位姿.这个包含姿态信息
                    updateTransformPointCloudSinCos(&thisTransformation);
                    // extract surrounding map
                    // 依据上面得到的变换thisTransformation，对cornerCloudKeyFrames，surfCloudKeyFrames，surfCloudKeyFrames
                    // 进行坐标变换,后加入到用于回环检测的点云中
                    recentCornerCloudKeyFrames.push_front(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
                    recentSurfCloudKeyFrames.push_front(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                    recentOutlierCloudKeyFrames.push_front(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                    if (recentCornerCloudKeyFrames.size() >= surroundingKeyframeSearchNum) // 保存最近的50帧关键帧
                        break;
                }
            }
            else // cloudKeyPoses3D已经更新，表示已经有新的关键帧添加进来
            {
                // queue is full, pop the oldest key frame and push the latest key frame
                // recentCornerCloudKeyFrames中点云保存的数量较多
                // pop队列最前端的一个，再push后面一个
                if (latestFrameID != cloudKeyPoses3D->points.size() - 1)
                {
                    // if the robot is not moving, no need to update recent frames
                    recentCornerCloudKeyFrames.pop_front(); // 弹出最早的那一帧点云
                    recentSurfCloudKeyFrames.pop_front();
                    recentOutlierCloudKeyFrames.pop_front();
                    // push latest scan to the end of queue
                    latestFrameID = cloudKeyPoses3D->points.size() - 1; // 更新latestFrameID
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[latestFrameID];
                    updateTransformPointCloudSinCos(&thisTransformation);
                    // 添加最新关键帧进去
                    recentCornerCloudKeyFrames.push_back(transformPointCloud(cornerCloudKeyFrames[latestFrameID])); // 将最新的keypose对应的点云加入进去
                    recentSurfCloudKeyFrames.push_back(transformPointCloud(surfCloudKeyFrames[latestFrameID]));
                    recentOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[latestFrameID]));
                }
            }

            // 构建小地图
            for (int i = 0; i < recentCornerCloudKeyFrames.size(); ++i)
            {
                *laserCloudCornerFromMap += *recentCornerCloudKeyFrames[i];
                *laserCloudSurfFromMap += *recentSurfCloudKeyFrames[i];
                *laserCloudSurfFromMap += *recentOutlierCloudKeyFrames[i];
            }
        }
        else // 没有回环检测构建方式
        {
            surroundingKeyPoses->clear();
            surroundingKeyPosesDS->clear();
            // extract all the nearby key poses and downsample them
            // cloudKeyPoses3D虽说是点云，但是是为了保存机器人在建图过程中的轨迹，其中的点就是定周期采样的轨迹点，这一点是在saveKeyFramesAndFactor中计算出的，在第一帧时必然是空的
            // surroundingKeyframeSearchRadius是50米，也就是说是在当前位置进行半径查找，得到附近的轨迹点
            // 距离数据保存在pointSearchSqDis中
            kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // 构建kdtree
            // 找当前帧50m之内的所有关键帧，surroundingKeyframeSearchRadius = 50.0
            // 进行半径surroundingKeyframeSearchRadius内的邻域搜索，
            // currentRobotPosPoint：需要查询的点，
            // pointSearchInd：搜索完的邻域点对应的索引
            // pointSearchSqDis：搜索完的每个领域点点与传讯点之间的欧式距离
            // 0：返回的邻域个数，为0表示返回全部的邻域点
            kdtreeSurroundingKeyPoses->radiusSearch(currentRobotPosPoint, (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis, 0);
            for (int i = 0; i < pointSearchInd.size(); ++i)
                surroundingKeyPoses->points.push_back(cloudKeyPoses3D->points[pointSearchInd[i]]);

            // 对附近轨迹点的点云进行降采样，轨迹具有一定间隔
            // 下采样
            downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
            downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
            // delete key frames that are not in surrounding region
            int numSurroundingPosesDS = surroundingKeyPosesDS->points.size();

            /*
                surroundingExistingKeyPosesID保存的时surroundingCornerCloudKeyFrames中保存的所有关键帧下标
                遍历surroundingExistingKeyPosesID判断距离当前帧50m之内的关键帧是否之前已经保存了
            */
            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i)
            {
                bool existingFlag = false;
                for (int j = 0; j < numSurroundingPosesDS; ++j)
                {
                    // 双重循环，不断对比surroundingExistingKeyPosesID[i]和surroundingKeyPosesDS的点的index
                    // 如果能够找到一样的，说明存在相同的关键点(因为surroundingKeyPosesDS从cloudKeyPoses3D中筛选而来)
                    if (surroundingExistingKeyPosesID[i] == (int)surroundingKeyPosesDS->points[j].intensity)
                    {
                        existingFlag = true;
                        break;
                    }
                }

                // 没找到就删除
                if (existingFlag == false)
                {
                    // 如果surroundingExistingKeyPosesID[i]对比了一轮的已经存在的关键位姿的索引后（intensity保存的就是size()）
                    // 没有找到相同的关键点，那么把这个点从当前队列中删除
                    // 否则的话，existingFlag为true，该关键点就将它留在队列中
                    surroundingExistingKeyPosesID.erase(surroundingExistingKeyPosesID.begin() + i);
                    surroundingCornerCloudKeyFrames.erase(surroundingCornerCloudKeyFrames.begin() + i);
                    surroundingSurfCloudKeyFrames.erase(surroundingSurfCloudKeyFrames.begin() + i);
                    surroundingOutlierCloudKeyFrames.erase(surroundingOutlierCloudKeyFrames.begin() + i);
                    --i;
                }
            }

            // add new key frames that are not in calculated existing key frames
            // 上一个两重for循环主要用于删除数据，此处的两重for循环用来添加数据
            for (int i = 0; i < numSurroundingPosesDS; ++i)
            {
                bool existingFlag = false;
                for (auto iter = surroundingExistingKeyPosesID.begin(); iter != surroundingExistingKeyPosesID.end(); ++iter)
                {
                    // *iter就是不同的cloudKeyPoses3D->points.size(),
                    // 把surroundingExistingKeyPosesID内没有对应的点放进一个队列里
                    // 这个队列专门存放周围存在的关键帧，但是和surroundingExistingKeyPosesID的点没有对应的，也就是新的点
                    if ((*iter) == (int)surroundingKeyPosesDS->points[i].intensity)
                    {
                        existingFlag = true;
                        break;
                    }
                }

                if (existingFlag == true)
                {
                    continue;
                }
                else // 没有就添加
                {
                    int thisKeyInd = (int)surroundingKeyPosesDS->points[i].intensity;
                    PointTypePose thisTransformation = cloudKeyPoses6D->points[thisKeyInd];
                    updateTransformPointCloudSinCos(&thisTransformation);
                    surroundingExistingKeyPosesID.push_back(thisKeyInd);
                    surroundingCornerCloudKeyFrames.push_back(transformPointCloud(cornerCloudKeyFrames[thisKeyInd]));
                    surroundingSurfCloudKeyFrames.push_back(transformPointCloud(surfCloudKeyFrames[thisKeyInd]));
                    surroundingOutlierCloudKeyFrames.push_back(transformPointCloud(outlierCloudKeyFrames[thisKeyInd]));
                }
            }

            // 累加点云
            for (int i = 0; i < surroundingExistingKeyPosesID.size(); ++i)
            {
                *laserCloudCornerFromMap += *surroundingCornerCloudKeyFrames[i];
                *laserCloudSurfFromMap += *surroundingSurfCloudKeyFrames[i];
                *laserCloudSurfFromMap += *surroundingOutlierCloudKeyFrames[i];
            }
        }

        // Downsample the surrounding corner key frames (or map)
        // 进行两次下采样
        // 最后的输出结果是laserCloudCornerFromMapDS和laserCloudSurfFromMapDS
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->points.size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->points.size();
    }

    // 下采样
    void downsampleCurrentScan()
    {

        laserCloudRawDS->clear();
        downSizeFilterScancontext.setInputCloud(laserCloudRaw);
        downSizeFilterScancontext.filter(*laserCloudRawDS);

        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->points.size();
        // std::cout << "laserCloudCornerLastDSNum: " << laserCloudCornerLastDSNum << std::endl;

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->points.size();
        // std::cout << "laserCloudSurfLastDSNum: " << laserCloudSurfLastDSNum << std::endl;

        laserCloudOutlierLastDS->clear();
        downSizeFilterOutlier.setInputCloud(laserCloudOutlierLast);
        downSizeFilterOutlier.filter(*laserCloudOutlierLastDS);
        laserCloudOutlierLastDSNum = laserCloudOutlierLastDS->points.size();

        laserCloudSurfTotalLast->clear();
        laserCloudSurfTotalLastDS->clear();
        *laserCloudSurfTotalLast += *laserCloudSurfLastDS;
        *laserCloudSurfTotalLast += *laserCloudOutlierLastDS;
        downSizeFilterSurf.setInputCloud(laserCloudSurfTotalLast);
        downSizeFilterSurf.filter(*laserCloudSurfTotalLastDS);
        laserCloudSurfTotalLastDSNum = laserCloudSurfTotalLastDS->points.size();
    }

    // 边缘点优化
    void cornerOptimization(int iterCount)
    {
        updatePointAssociateToMapSinCos(); // 先更新预测的T的sin值和cos值，方便使用
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            pointOri = laserCloudCornerLastDS->points[i];

            // 进行坐标变换,转换到全局坐标中去（世界坐标系）
            // pointSel:表示选中的点，point select
            // 输入是pointOri，输出是pointSel
            pointAssociateToMap(&pointOri, &pointSel);

            // 进行5邻域搜索，
            // pointSel为需要搜索的点，
            // pointSearchInd搜索完的邻域对应的索引
            // pointSearchSqDis 邻域点与查询点之间的距离
            // 利用kd树查找最近的5个点，接下来需要计算这五个点的协方差
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            // 只有当最远的那个邻域点的距离pointSearchSqDis[4]小于1m时才进行下面的计算
            // 以下部分的计算是在计算点集的协方差矩阵，Zhang Ji的论文中有提到这部分
            if (pointSearchSqDis[4] < 1.0)
            {
                // 先求5个样本的平均值
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++)
                {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }

                // 计算算术平均值
                cx /= 5;
                cy /= 5;
                cz /= 5;

                // 下面在求矩阵matA1=[ax,ay,az]^t*[ax,ay,az]
                // 更准确地说应该是在求协方差matA1
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                // 计算协方差矩阵
                for (int j = 0; j < 5; j++)
                {
                    // ax代表的是x-cx,表示均值与每个实际值的差值，求取5个之后再次取平均，得到matA1
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax;
                    a12 += ax * ay;
                    a13 += ax * az;
                    a22 += ay * ay;
                    a23 += ay * az;
                    a33 += az * az;
                }

                a11 /= 5;
                a12 /= 5;
                a13 /= 5;
                a22 /= 5;
                a23 /= 5;
                a33 /= 5;

                matA1.at<float>(0, 0) = a11;
                matA1.at<float>(0, 1) = a12;
                matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12;
                matA1.at<float>(1, 1) = a22;
                matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13;
                matA1.at<float>(2, 1) = a23;
                matA1.at<float>(2, 2) = a33;

                // 求正交阵的特征值和特征向量
                // 特征值：matD1，特征向量：matV1中
                cv::eigen(matA1, matD1, matV1);

                // 边缘：与较大特征值相对应的特征向量代表边缘线的方向（一大两小，大方向）
                // 以下这一大块是在计算点到边缘的距离，最后通过系数s来判断是否距离很近
                // 如果距离很近就认为这个点在边缘上，需要放到laserCloudOri中
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1))
                {
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0); // 在5个的中心值处沿着线的方向前后选择0.1米远的点，作为直线上的两个点
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    // 这边是在求[(x0-x1),(y0-y1),(z0-z1)]与[(x0-x2),(y0-y2),(z0-z2)]叉乘得到的向量的模长
                    // 这个模长是由0.2*V1[0]和点[x0,y0,z0]构成的平行四边形的面积
                    // 因为[(x0-x1),(y0-y1),(z0-z1)]x[(x0-x2),(y0-y2),(z0-z2)]=[XXX,YYY,ZZZ],
                    // [XXX,YYY,ZZZ]=[(y0-y1)(z0-z2)-(y0-y2)(z0-z1),-(x0-x1)(z0-z2)+(x0-x2)(z0-z1),(x0-x1)(y0-y2)-(x0-x2)(y0-y1)]
                    float a012 = sqrt(((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1)));

                    // l12表示的是0.2*(||V1[0]||)
                    // 也就是平行四边形一条底的长度
                    float l12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));

                    // 求叉乘结果[la',lb',lc']=[(x1-x2),(y1-y2),(z1-z2)]x[XXX,YYY,ZZZ]
                    // [la,lb,lc]=[la',lb',lc']/a012/l12
                    // LLL=[la,lb,lc]是0.2*V1[0]这条高上的单位法向量。||LLL||=1；
                    float la = ((y1 - y2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) + (z1 - z2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2) * ((x0 - x1) * (y0 - y2) - (x0 - x2) * (y0 - y1)) - (z1 - z2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2) * ((x0 - x1) * (z0 - z2) - (x0 - x2) * (z0 - z1)) + (y1 - y2) * ((y0 - y1) * (z0 - z2) - (y0 - y2) * (z0 - z1))) / a012 / l12;

                    // 计算点pointSel到直线的距离
                    // 这里需要特别说明的是ld2代表的是点pointSel到过点[cx,cy,cz]的方向向量直线的距离
                    float ld2 = a012 / l12;

                    // 如果在最理想的状态的话，ld2应该为0，表示点在直线上
                    // 最理想状态s=1；
                    float s = 1 - 0.9 * fabs(ld2);

                    // coeff代表系数的意思
                    // coff用于保存距离的方向向量
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;

                    // intensity本质上构成了一个核函数，ld2越接近于1，增长越慢
                    // intensity=(1-0.9*ld2)*ld2=ld2-0.9*ld2*ld2
                    coeff.intensity = s * ld2;

                    // 所以就应该认为这个点是边缘点
                    // s>0.1 也就是要求点到直线的距离ld2要小于1m
                    // s越大说明ld2越小(离边缘线越近)，这样就说明点pointOri在直线上
                    if (s > 0.1)
                    {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    // 平面点优化
    void surfOptimization(int iterCount)
    {
        updatePointAssociateToMapSinCos();
        for (int i = 0; i < laserCloudSurfTotalLastDSNum; i++)
        {
            pointOri = laserCloudSurfTotalLastDS->points[i];
            // 转换到地图坐标系下
            pointAssociateToMap(&pointOri, &pointSel);
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            if (pointSearchSqDis[4] < 1.0)
            {
                for (int j = 0; j < 5; j++)
                {
                    matA0.at<float>(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0.at<float>(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0.at<float>(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                // matB0是一个5x1的矩阵
                // matB0 = cv::Mat (5, 1, CV_32F, cv::Scalar::all(-1));
                // matX0是3x1的矩阵
                // 求解方程matA0*matX0=matB0
                // 公式其实是在求由matA0中的点构成的平面的法向量matX0
                cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);

                // [pa,pb,pc,pd]=[matX0,pd]
                // 正常情况下（见后面planeValid判断条件），应该是
                // pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                // pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                // pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z = -1
                // 所以pd设置为1
                float pa = matX0.at<float>(0, 0);
                float pb = matX0.at<float>(1, 0);
                float pc = matX0.at<float>(2, 0);
                float pd = 1;

                // 对[pa,pb,pc,pd]进行单位化
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                // 求解后再次检查平面是否是有效平面
                bool planeValid = true;
                for (int j = 0; j < 5; j++)
                {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2)
                    {
                        planeValid = false;
                        break;
                    }
                }

                if (planeValid)
                {
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    // 后面部分相除求的是[pa,pb,pc,pd]与pointSel的夹角余弦值(两个sqrt，其实并不是余弦值)
                    // 这个夹角余弦值越小越好，越小证明所求的[pa,pb,pc,pd]与平面越垂直
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    // 判断是否是合格平面，是就加入laserCloudOri
                    if (s > 0.1)
                    {
                        laserCloudOri->push_back(pointOri);
                        coeffSel->push_back(coeff);
                    }
                }
            }
        }
    }

    // 这部分的代码是基于高斯牛顿法的优化，不是zhang ji论文中提到的基于L-M的优化方法
    // 这部分的代码使用旋转矩阵对欧拉角求导，优化欧拉角，不是zhang ji论文中提到的使用angle-axis的优化
    bool LMOptimization(int iterCount)
    {
        float srx = sin(transformTobeMapped[0]);
        float crx = cos(transformTobeMapped[0]);
        float sry = sin(transformTobeMapped[1]);
        float cry = cos(transformTobeMapped[1]);
        float srz = sin(transformTobeMapped[2]);
        float crz = cos(transformTobeMapped[2]);

        int laserCloudSelNum = laserCloudOri->points.size();
        // 如果进行配准的点的个数小于50个，则不优化
        if (laserCloudSelNum < 50)
        {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        for (int i = 0; i < laserCloudSelNum; i++)
        {
            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            /*
                X0 = R * Xi + T
                R = R_yaw * R_pitch * R_roll
                  = R_ry * R_rx * R_rz
                  = |cry 0 sry|    |1 0 0|        |crz -srz 0|
                    |0 1 0|      * |0 crx -srx| * |srz crz 0|
                    |-sry 0 cry|   |0 srx crx|    |0 0 1|
                  = |crycrz+srxsrysrz  srxsrycrz-crysrz  crxsry|
                    |crxsrz  crxcrz  -srx|
                    |srxcrysrz-srycrz  srysrz+srxcrycrz  crxcry|

                error = (R * point + T) * coeff

                derror/drx = |crxsrysrz  crxsrycrz  -srxsry|
                             |-srxsrz  -srxcrz  -crx|
                             |crxcrysrz  crxcrycrz  -srxcry|

                derror/dry = |srxcrysrz-srycrz  srxcrycrz+srysrz  crxcry|
                             |0 0 0|
                             |-srxsrysrz-crycrz  crysrz-srxsrycrz  -crxsry|

                derror/drz = |srxsrycrz-crysrz  -srxsrysrz-crycrz  0|
                             |crxcrz  -crxsrz  0|
                             |srxcrycrz+srysrz  srycrz-srxcrysrz  0|
            
                derror/dtx = coeff.x
                
                derror/dty = coeff.y

                derror/dtz = coeff.z
            */

            // 求雅克比矩阵中的元素，距离d对roll角度的偏导量即d(d)/d(roll)
            // 更详细的数学推导参看wykxwyc.github.io
            float arx = (crx * sry * srz * pointOri.x + crx * crz * sry * pointOri.y - srx * sry * pointOri.z) * coeff.x + (-srx * srz * pointOri.x - crz * srx * pointOri.y - crx * pointOri.z) * coeff.y + (crx * cry * srz * pointOri.x + crx * cry * crz * pointOri.y - cry * srx * pointOri.z) * coeff.z;

            // 同上，求解的是对pitch的偏导量
            float ary = ((cry * srx * srz - crz * sry) * pointOri.x + (sry * srz + cry * crz * srx) * pointOri.y + crx * cry * pointOri.z) * coeff.x + ((-cry * crz - srx * sry * srz) * pointOri.x + (cry * srz - crz * srx * sry) * pointOri.y - crx * sry * pointOri.z) * coeff.z;

            float arz = ((crz * srx * sry - cry * srz) * pointOri.x + (-cry * crz - srx * sry * srz) * pointOri.y) * coeff.x + (crx * crz * pointOri.x - crx * srz * pointOri.y) * coeff.y + ((sry * srz + cry * crz * srx) * pointOri.x + (crz * sry - cry * srx * srz) * pointOri.y) * coeff.z;

            /*
                在求点到直线的距离时，coeff表示的是如下内容
                [la,lb,lc]表示的是点到直线的垂直连线方向，s是长度
                coeff.x = s * la;
                coeff.y = s * lb;
                coeff.z = s * lc;
                coeff.intensity = s * ld2;

                在求点到平面的距离时，coeff表示的是
                [pa,pb,pc]表示过外点的平面的法向量，s是线的长度
                coeff.x = s * pa;
                coeff.y = s * pb;
                coeff.z = s * pc;
                coeff.intensity = s * pd2;
            */
            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;

            // 这部分是雅克比矩阵中距离对平移的偏导
            matA.at<float>(i, 3) = coeff.x;
            matA.at<float>(i, 4) = coeff.y;
            matA.at<float>(i, 5) = coeff.z;

            // 残差项
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        // 将矩阵由matA转置生成matAt
        // 先进行计算，以便于后边调用 cv::solve求解
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;

        // 利用高斯牛顿法进行求解，
        // 高斯牛顿法的原型是J^(T)*J * delta(x) = -J*f(x)
        // J是雅克比矩阵，这里是A，f(x)是优化目标，这里是-B(符号在给B赋值时候就放进去了)
        // 通过QR分解的方式，求解matAtA*matX=matAtB，得到解matX
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        // iterCount==0 说明是第一次迭代，需要初始化
        if (iterCount == 0)
        {
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0)); // 特征值1*6矩阵
            // 特征向量6*6矩阵
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            // 对近似的Hessian矩阵求特征值和特征向量，
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            // 特征值取值门槛
            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) // 从小到大查找
            {
                if (matE.at<float>(0, i) < eignThre[i]) // 特征值太小，则认为处在兼并环境中，发生了退化
                {
                    for (int j = 0; j < 6; j++) // 对应的特征向量置为0
                    {
                        matV2.at<float>(i, j) = 0;
                    }

                    isDegenerate = true;
                }
                else
                {
                    break;
                }
            }

            // 计算P矩阵
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) // 如果发生退化，只使用预测矩阵P计算
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        // 积累每次的调整量
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
            pow(matX.at<float>(3, 0) * 100, 2) +
            pow(matX.at<float>(4, 0) * 100, 2) +
            pow(matX.at<float>(5, 0) * 100, 2));

        // 旋转或者平移量足够小就停止这次迭代过程
        if (deltaR < 0.05 && deltaT < 0.05)
        {
            return true;
        }

        return false;
    }

    // 地图匹配
    void scan2MapOptimization()
    {
        // laserCloudCornerFromMapDSNum是extractSurroundingKeyFrames()函数最后降采样得到的coner点云数
        // laserCloudSurfFromMapDSNum是extractSurroundingKeyFrames()函数降采样得到的surface点云数
        if (laserCloudCornerFromMapDSNum > 10 && laserCloudSurfFromMapDSNum > 100)
        {
            // laserCloudCornerFromMapDS和laserCloudSurfFromMapDS的来源有2个：
            // 当有闭环时，来源是：recentCornerCloudKeyFrames，没有闭环时，来源surroundingCornerCloudKeyFrames
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 10; iterCount++)
            {
                // 用for循环控制迭代次数，最多迭代10次
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization(iterCount); // 次边缘点匹配
                surfOptimization(iterCount);   // 次平面点匹配

                if (LMOptimization(iterCount) == true) // 梯度下降求解
                    break;
            }

            // 迭代结束更新相关的转移矩阵
            transformUpdate();
        }
    }

    // 选取关键帧
    void saveKeyFramesAndFactor()
    {
        // 当前帧的全局位置
        currentRobotPosPoint.x = transformAftMapped[3];
        currentRobotPosPoint.y = transformAftMapped[4];
        currentRobotPosPoint.z = transformAftMapped[5];

        // 距离上一个关键帧已经过去0.3m,那么当前帧可以作为关键帧被保存
        bool saveThisKeyFrame = true;
        if (sqrt((previousRobotPosPoint.x - currentRobotPosPoint.x) * (previousRobotPosPoint.x - currentRobotPosPoint.x) + (previousRobotPosPoint.y - currentRobotPosPoint.y) * (previousRobotPosPoint.y - currentRobotPosPoint.y) + (previousRobotPosPoint.z - currentRobotPosPoint.z) * (previousRobotPosPoint.z - currentRobotPosPoint.z)) < 0.3)
        {
            saveThisKeyFrame = false;
        }

        if (saveThisKeyFrame == false && !cloudKeyPoses3D->points.empty())
            return;

        previousRobotPosPoint = currentRobotPosPoint;

        /**
         * update grsam graph
         */
        // 第一个关键帧
        if (cloudKeyPoses3D->points.empty())
        {
            // static Rot3 	RzRyRx (double x, double y, double z),Rotations around Z, Y, then X axes
            // RzRyRx依次按照z(transformTobeMapped[2])，y(transformTobeMapped[0])，x(transformTobeMapped[1])坐标轴旋转
            // Point3 (double x, double y, double z)  Construct from x(transformTobeMapped[5]), y(transformTobeMapped[3]), and z(transformTobeMapped[4]) coordinates.
            // Pose3 (const Rot3 &R, const Point3 &t) Construct from R,t. 从旋转和平移构造姿态
            // NonlinearFactorGraph增加一个PriorFactor因子
            gtSAMgraph.add(PriorFactor<Pose3>(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]), Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])), priorNoise));
            // initialEstimate的数据类型是Values,其实就是一个map，这里在0对应的值下面保存了一个Pose3
            initialEstimate.insert(0, Pose3(Rot3::RzRyRx(transformTobeMapped[2], transformTobeMapped[0], transformTobeMapped[1]),
                                            Point3(transformTobeMapped[5], transformTobeMapped[3], transformTobeMapped[4])));
            for (int i = 0; i < 6; ++i)
                transformLast[i] = transformTobeMapped[i]; // 保存第一针位姿
        }
        else
        {
            /*
                transform: 2->yaw   0->pitch   1->roll
            */
            gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(transformLast[2], transformLast[0], transformLast[1]),
                                          Point3(transformLast[5], transformLast[3], transformLast[4]));
            gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                        Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4]));
            // 构造函数原型:BetweenFactor (Key key1, Key key2, const VALUE &measured, const SharedNoiseModel &model)
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->points.size() - 1, cloudKeyPoses3D->points.size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->points.size(), Pose3(Rot3::RzRyRx(transformAftMapped[2], transformAftMapped[0], transformAftMapped[1]),
                                                                         Point3(transformAftMapped[5], transformAftMapped[3], transformAftMapped[4])));
        }

        /*
         * update iSAM
         */
        // gtsam::ISAM2::update函数原型:
        // ISAM2Result gtsam::ISAM2::update	(	const NonlinearFactorGraph & 	newFactors = NonlinearFactorGraph(),
        // const Values & 	newTheta = Values(),
        // const std::vector< size_t > & 	removeFactorIndices = std::vector<size_t>(),
        // const boost::optional< FastMap< Key, int > > & 	constrainedKeys = boost::none,
        // const boost::optional< FastList< Key > > & 	noRelinKeys = boost::none,
        // const boost::optional< FastList< Key > > & 	extraReelimKeys = boost::none,
        // bool 	force_relinearize = false )
        // gtSAMgraph是新加到系统中的因子
        // initialEstimate是加到系统中的新变量的初始点
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        gtSAMgraph.resize(0);
        initialEstimate.clear();

        /*
         * save key poses
         */
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size() - 1);

        thisPose3D.x = latestEstimate.translation().y();
        thisPose3D.y = latestEstimate.translation().z();
        thisPose3D.z = latestEstimate.translation().x();
        thisPose3D.intensity = cloudKeyPoses3D->points.size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity; // this can be used as index
        thisPose6D.roll = latestEstimate.rotation().pitch();
        thisPose6D.pitch = latestEstimate.rotation().yaw();
        thisPose6D.yaw = latestEstimate.rotation().roll(); // in camera frame
        thisPose6D.time = timeLaserOdometry;
        cloudKeyPoses6D->push_back(thisPose6D);

        /*
         * save updated transform
         */
        if (cloudKeyPoses3D->points.size() > 1)
        {
            transformAftMapped[0] = latestEstimate.rotation().pitch();
            transformAftMapped[1] = latestEstimate.rotation().yaw();
            transformAftMapped[2] = latestEstimate.rotation().roll();
            transformAftMapped[3] = latestEstimate.translation().y();
            transformAftMapped[4] = latestEstimate.translation().z();
            transformAftMapped[5] = latestEstimate.translation().x();

            for (int i = 0; i < 6; ++i) // 此时transformAftMapped经过gtsam优化后,为最终的transform.保存到transformLast,用于下次的gtsam因子图边构建.
            {
                transformLast[i] = transformAftMapped[i];
                transformTobeMapped[i] = transformAftMapped[i]; // 将gtsam调整后的transformAftMaped赋值给tobeMapped,在调用该函数前,AftMapped刚由tobeMap赋值.所以此时将调整后的值在赋值回去.
            }
        }

        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisOutlierKeyFrame(new pcl::PointCloud<PointType>());

        pcl::copyPointCloud(*laserCloudCornerLastDS, *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS, *thisSurfKeyFrame);
        pcl::copyPointCloud(*laserCloudOutlierLastDS, *thisOutlierKeyFrame);

        /* 
            Scan Context loop detector 
            - ver 1: using surface feature as an input point cloud for scan context (2020.04.01: checked it works.)
            - ver 2: using downsampled original point cloud (/full_cloud_projected + downsampling)
            */
        bool usingRawCloud = true;
        if (usingRawCloud)
        { // v2 uses downsampled raw point cloud, more fruitful height information than using feature points (v1)
            pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame(new pcl::PointCloud<PointType>());
            pcl::copyPointCloud(*laserCloudRawDS, *thisRawCloudKeyFrame);
            scManager.makeAndSaveScancontextAndKeys(*thisRawCloudKeyFrame);
        }
        else
        { // v1 uses thisSurfKeyFrame, it also works. (empirically checked at Mulran dataset sequences)
            scManager.makeAndSaveScancontextAndKeys(*thisSurfKeyFrame);
        }

        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);
        outlierCloudKeyFrames.push_back(thisOutlierKeyFrame);
    }

    // 修正关键帧位姿
    void correctPoses()
    {
        // 闭环检测并优化结束，那么更新所有关键帧位姿
        if (aLoopIsClosed == true)
        {
            recentCornerCloudKeyFrames.clear();
            recentSurfCloudKeyFrames.clear();
            recentOutlierCloudKeyFrames.clear();
            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().z();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().x();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();
                cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
            }

            aLoopIsClosed = false;
        }
    }

    // 清空变量
    void clearCloud()
    {
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear();
        laserCloudCornerFromMapDS->clear();
        laserCloudSurfFromMapDS->clear();
    }

    void run()
    {
        // 有新数据进来，才执行后续
        // 如果成员变量里接收到了新的数据
        if (newLaserCloudCornerLast && std::abs(timeLaserCloudCornerLast - timeLaserOdometry) < 0.005 &&
            newLaserCloudSurfLast && std::abs(timeLaserCloudSurfLast - timeLaserOdometry) < 0.005 &&
            newLaserCloudOutlierLast && std::abs(timeLaserCloudOutlierLast - timeLaserOdometry) < 0.005 &&
            newLaserOdometry)
        {
            newLaserCloudCornerLast = false;
            newLaserCloudSurfLast = false;
            newLaserCloudOutlierLast = false;
            newLaserOdometry = false;

            std::lock_guard<std::mutex> lock(mtx); // 和回环检测不同时进行

            // 距离上一次进行scan-to-map优化足够久了
            // mappingProcessInterval = 0.3 每隔3帧和地图匹配一次
            if (timeLaserOdometry - timeLastProcessing >= mappingProcessInterval)
            {
                timeLastProcessing = timeLaserOdometry; // 更新时间戳

                transformAssociateToMap();     // 根据当前的odo pose,以及上一次进行map_optimation前后的pose(即漂移),计算目前最优的位姿估计
                extractSurroundingKeyFrames(); // 确定周围的关键帧的索引,点云保存到recentCorner等，地图拼接保存到laserCloudCornerFromMap等
                downsampleCurrentScan();       // 对当前帧原始点云,角点,面点,离群点进行降采样
                // 最优位姿保存在和transformAftMapped中，同时transformBfeMapped中保存了优化前的位姿，两者的差距就是激光odo和最优位姿之间偏移量的估计
                scan2MapOptimization();        // 进行scan-to-map位姿优化,并为下一次做准备
                // 如果距离上一次保存的关键帧欧式距离最够大，需要保存当前关键帧
                // 计算与上一关键帧之间的约束，这种约束可以理解为局部的小回环，加入后端进行优化，
                // 将优化的结果保存作为关键帧位姿，同步到scan-to-map优化环节
                // 为了检测全局的大回环,还需要生成当前关键帧的ScanContext
                saveKeyFramesAndFactor();      // 保存关键帧
                correctPoses();                // 如果另一个线程中isam完成了一次全局位姿优化,那么对关键帧中cloudKeyPoses3D/6D的位姿进行修正
                publishTF();                   // 发布优化后的位姿,及tf变换
                publishKeyPosesAndFrames();    // 发布所有关键帧位姿,当前的局部面点地图及当前帧中的面点/角点
                clearCloud();
            }
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Map Optimization Started.");

    mapOptimization MO;

    // std::thread 构造函数，将MO作为参数传入构造的线程中使用
    // 进行回环检测与回环的功能
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    // 该线程中进行的工作是publishGlobalMap()，将数据发布到ros中，可视化
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::Rate rate(200);
    while (ros::ok())
    {
        ros::spinOnce();

        MO.run();

        rate.sleep();
    }

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
