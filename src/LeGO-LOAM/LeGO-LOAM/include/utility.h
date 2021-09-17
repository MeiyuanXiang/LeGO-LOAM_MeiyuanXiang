#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <ros/ros.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

#include "cloud_msgs/cloud_info.h"

#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#define PI 3.14159265

using namespace std;

typedef pcl::PointXYZI PointType; // 带intensity的PointXYZ

extern const string pointCloudTopic = "/velodyne_points";
extern const string imuTopic = "/imu/data";

// Save pcd
extern const string fileDirectory = "/tmp/";

// Using velodyne cloud "ring" channel for image projection (other lidar may have different name for this channel, change "PointXYZIR" below)
extern const bool useCloudRing = true; // 使用Ouster数据时需要改为false，if true, ang_res_y and ang_bottom are not used

// VLP-16
extern const int N_SCAN = 16;               // 扫描线数
extern const int Horizon_SCAN = 1800;       // 旋转一周采样次数，每线1800个点
extern const float ang_res_x = 0.2;         // 水平分辨率，水平方向每条线间隔0.2°
extern const float ang_res_y = 2.0;         // 垂直分辨率，垂直方向每条线间隔2°
extern const float ang_bottom = 15.0 + 0.1; // 底部线束的角度偏移，垂直方向上起始角度是负角度，与水平方向相差15.1°
extern const int groundScanInd = 7;         // 判断为地面的线数，以多少个扫描圈来表示地面

// HDL-32E
// extern const int N_SCAN = 32;
// extern const int Horizon_SCAN = 1800;
// extern const float ang_res_x = 360.0/float(Horizon_SCAN);
// extern const float ang_res_y = 41.33/float(N_SCAN-1);
// extern const float ang_bottom = 30.67;
// extern const int groundScanInd = 20;

// VLS-128
// extern const int N_SCAN = 128;
// extern const int Horizon_SCAN = 1800;
// extern const float ang_res_x = 0.2;
// extern const float ang_res_y = 0.3;
// extern const float ang_bottom = 25.0;
// extern const int groundScanInd = 10;

// Ouster users may need to uncomment line 159 in imageProjection.cpp
// Usage of Ouster imu data is not fully supported yet (LeGO-LOAM needs 9-DOF IMU), please just publish point cloud data
// Ouster OS1-16
// extern const int N_SCAN = 16;
// extern const int Horizon_SCAN = 1024;
// extern const float ang_res_x = 360.0/float(Horizon_SCAN);
// extern const float ang_res_y = 33.2/float(N_SCAN-1);
// extern const float ang_bottom = 16.6+0.1;
// extern const int groundScanInd = 7;

// Ouster OS1-64
// extern const int N_SCAN = 64;
// extern const int Horizon_SCAN = 1024;
// extern const float ang_res_x = 360.0/float(Horizon_SCAN);
// extern const float ang_res_y = 33.2/float(N_SCAN-1);
// extern const float ang_bottom = 16.6+0.1;
// extern const int groundScanInd = 15;

extern const bool loopClosureEnableFlag = false;  // 是否需要回环检测
extern const double mappingProcessInterval = 0.3; // 建图时间间隔（周期）

extern const float scanPeriod = 0.1; // imu扫描频率
extern const int systemDelay = 0;    // 系统延迟
extern const int imuQueLength = 200; // imu缓存大小

extern const float sensorMinimumRange = 1.0;                 // 过滤距离小于1m的点云
extern const float sensorMountAngle = 0.0;                   // 判断是否为地面点的阈值
extern const float segmentTheta = 60.0 / 180.0 * M_PI;       // 点云分割每个方向的角度大小，用于判断平面，decrese this value may improve accuracy
extern const int segmentValidPointNum = 5;                   // 检查上下左右连续5个点做为分割的特征依据
extern const int segmentValidLineNum = 3;                    // 检查上下左右连续3条线做为分割的特征依据
extern const float segmentAlphaX = ang_res_x / 180.0 * M_PI; // 水平方向的最小分辨率弧度
extern const float segmentAlphaY = ang_res_y / 180.0 * M_PI; // 垂直方向的最小分辨率弧度

extern const int edgeFeatureNum = 2;                // 边缘特征数量
extern const int surfFeatureNum = 4;                // 平面特征数量
extern const int sectionsTotal = 6;                 // 没用到
extern const float edgeThreshold = 0.1;             // 边缘曲率阈值
extern const float surfThreshold = 0.1;             // 平面曲率阈值
extern const float nearestFeatureSearchSqDist = 25; // 特征查找半径

// Mapping Params
extern const float surroundingKeyframeSearchRadius = 50.0; // 点云地图搜索半径，key frame that is within n meters from current pose will be considerd for scan-to-map optimization (when loop closure disabled)
extern const int surroundingKeyframeSearchNum = 50;        // 点云地图搜索大小，submap size (when loop closure enabled)
// history key frames (history submap for loop closure)
extern const float historyKeyframeSearchRadius = 7.0; // 回环检测首尾距离，key frame that is within n meters from current pose will be considerd for loop closure
extern const int historyKeyframeSearchNum = 25;       // 回环检测点云查找范围[-25，25]，2n+1 number of hostory key frames will be fused into a submap for loop closure
extern const float historyKeyframeFitnessScore = 0.3; // icp匹配分数，平均距离小于0.3，the smaller the better alignment

extern const float globalMapVisualizationSearchRadius = 500.0; // 可视化点云范围，key frames with in n meters will be visualized

struct smoothness_t
{
    float value;
    size_t ind;
};

struct by_value
{
    bool operator()(smoothness_t const &left, smoothness_t const &right)
    {
        return left.value < right.value;
    }
};

/*
  A point cloud type that has "ring" channel
*/
struct PointXYZIR
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIR,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(uint16_t, ring, ring))

/*
  A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
*/
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D // 该点类型有4个元素
        PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 确保new操作符对齐操作
} EIGEN_ALIGN16;                    // 强制SSE对齐

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))

typedef PointXYZIRPYT PointTypePose;

#endif
