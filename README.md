# LeGO-LOAM_MeiyuanXiang
LeGO-LOAM相关论文、代码中文注释以及代码改动

# 参考
https://github.com/RobustFieldAutonomyLab/LeGO-LOAM  
https://github.com/irapkaist/SC-LeGO-LOAM  

# 环境
1. Ubuntu（测试了Ubuntu16.04.5、Ubuntu18.04）  
2. ROS (测试了kinetic、melodic)  
3. gtsam（测试了4.0.0-alpha2）  

# 编译
1. 下载源码 git clone https://github.com/MeiyuanXiang/LeGO-LOAM_MeiyuanXiang.git  
2. 将LeGO-LOAM_MeiyuanXiang\src下的LeGO-LOAM或SC-LeGO-LOAM拷贝到ros工程空间src文件夹内，例如~/catkin_ws/src/  
3. cd ~/catkin_ws  
4. catkin_make -j1  
5. source ~/catkin_ws/devel/setup.bash  

# Bag数据
Stevens-VLP16-Dataset：https://drive.google.com/drive/folders/16p5UPUCZ1uK0U4XE-hJKjazTsRghEMJa  

# MulRan数据
https://sites.google.com/view/mulran-pr/download  

# 运行
1. 运行bag包数据  
roslaunch lego_loam run.launch  
rosbag play *.bag --clock --topic /velodyne_points /imu/data  
2. SC-LeGO-LOAM对MulRan数据集  
roslaunch lego_loam run.launch 

对于MulRan数据集的播放，打开新的终端，使用file_player_mulran播放MulRan数据(详情请参考：https://github.com/irapkaist/file_player_mulran)  
