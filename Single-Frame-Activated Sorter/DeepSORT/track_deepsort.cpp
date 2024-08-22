//
// Created by xuyufeng1 on 2021/3/25.
//

#include "track_deepsort.h"
#include <chrono>
#include <iostream>
#include <unistd.h> 
#include <fstream>
#include <map>
#include <vector>
#include <numeric>


struct vote{
	int count = 0;
    float confidence = 0.0;
};
// std::vector <std::vector<float> > dic;
std::map <int, vote> dic;
int classification;

#define MIN_HEIGHT 96 //输出最优帧的最小高度default=96
#define MIN_WIDTH 48 //输出最优帧的最小宽度default=48
#define MIN_HITS 20 //输出最优帧的最小匹配次数default=20

using namespace std;

track_deepsort::track_deepsort() : mytracker(MAX_COSINE_DISTANCE, NN_BUDGET), missed_num(0) {
    


    
   // mytracker = tracker(MAX_COSINE_DISTANCE, NN_BUDGET);
}

void track_deepsort::run(DETECTIONS &detections) {
    std::vector<int> ReleaseCacheFrames;
    //遮挡计算
    for (int i = 0; i < detections.size(); i++) {
        DETECTION_ROW &detectionRow = detections[i];
        if (detectionRow.shelter) continue;
        DETECTBOX tlwh = detectionRow.tlwh;
        int x1 = tlwh.array()[0];
        int y1 = tlwh.array()[1];
        int w = tlwh.array()[2];
        int h = tlwh.array()[3];
        int x2 = x1 + w;
        int y2 = y1 + h;
        int s = w * h;
        for (int j = i + 1; j < detections.size(); j++) {
            DETECTION_ROW _detectionRow = detections[j];
            DETECTBOX _tlwh = _detectionRow.tlwh;
            int _x1 = _tlwh.array()[0];
            int _y1 = _tlwh.array()[1];
            int _w = _tlwh.array()[2];
            int _h = _tlwh.array()[3];
            int _x2 = _x1 + _w;
            int _y2 = _y1 + _h;
            int startx = min(x1, _x1);
            int starty = min(y1, _y1);
            int endx = max(x2, _x2);
            int endy = max(y2, _y2);
            int width = w + _w - (endx - startx);  // 重叠部分宽
            int height = h + _h - (endy - starty);  // 重叠部分高
            if (width > 0 and height > 0) {
                int s_iou = width * height;
                if (s_iou / s >= 0.3) { //0.3
                    detections[j].shelter = true;
                    detections[i].shelter = true;
                    break;
                }
            }
        }
    }
    //处理上一帧输出的最优帧的残余数据
    for (Track &track: nextReleaseTrack) {
        std::map<int, std::set<std::string>>::iterator iter;
        iter = timestamplist.find(track.best_timestamp);
        if (iter != timestamplist.end()) {
            std::set<std::string> track_uuid_list = iter->second;
            if (track_uuid_list.count(track.uuid) != 0) {
                track_uuid_list.erase(track.uuid);
            }
            if (track_uuid_list.size() == 0) {
                timestamplist.erase(iter);
                ReleaseCacheFrames.push_back(track.best_timestamp);
            } else {
                timestamplist.erase(iter);
                timestamplist.insert(std::pair<int, std::set<std::string>>(track.best_timestamp, track_uuid_list));
            }
        }
    }
    nextReleaseTrack.clear();
    std::vector<DETECTION_ROW> filter_tracks;
    std::set<std::string> cached_track_list;


    auto trackStart = std::chrono::system_clock::now();
//        卡尔曼滤波更新
    mytracker.predict();
    std::vector<Track> missed_tracks = mytracker.update(detections);


    //输出消失track的最优帧
    for (Track &track: missed_tracks) {
        missed_num++;
        nextReleaseTrack.push_back(track);
        if (track.best_timestamp == -1) continue;
        DETECTBOX tlwh = track.best_img.tlwh;
        int h = tlwh.array()[3];
        int w = tlwh.array()[2];
        std::cout << "release track uuid:" << track.uuid << std::endl;
        if (track.hits > MIN_HITS and h > MIN_HEIGHT and w > MIN_WIDTH) {
            filter_tracks.push_back(track.best_img);
        }
        //对于每一个该帧输出的track，在下个时间戳应该释放掉包含该track的缓存帧

    }
    std::cout<<"missed num:"<<missed_num<<std::endl;
    std::cout<<"tracks num:"<<mytracker.tracks.size()<<std::endl<<std::endl;

}

int track_deepsort::serial(cv::Mat frame){
     cv::Mat image = frame;
     int *id_array;
     for (int i=0;i <mytracker.tracks.size();i++)
     {
         //printf("size %d\n",i);
         float t = mytracker.tracks[i].latest_detection.tlwh[0];//center point t,l
         float l = mytracker.tracks[i].latest_detection.tlwh[1];//latest_detection type: DETECTION_ROW in track.h
         float w = mytracker.tracks[i].latest_detection.tlwh[2];//the parameters of ith tracker
         float h = mytracker.tracks[i].latest_detection.tlwh[3];
         int class_id = mytracker.tracks[i].latest_detection.type;
         //id_array[i] = class_id;
         //printf("classid: %d:\n",class_id);
         int track_id = mytracker.tracks[i].track_id;
         float conf = mytracker.tracks[i].latest_detection.confidence;
        //  cv::putText(frame, frame_id, point, cv::FONT_HERSHEY_PLAIN, 1.0,
        //              cv::Scalar(0xFF, 0xFF, 0xFF), 2);
         
         //printf("tlwh %.1f, %.1f, %.1f, %.1f:\n",t,l,w,h);
         w = 80; // static w and h for bbox
         h = 60;
         int x1 = t - w/2; //(t,l) -> centroid t - w/2
         int y1 = l - h/2;  //l - h/2
         int x2 = x1 + w; //x1 + w (default)
         int y2 = y1 + h; //y1 + h (default)

         if (x1>0 && x2<800 && y1>0 && y2<400 && x2-x1==150 && y2-y1 == 150)
         {
            

            if (dic.count(track_id)){
                if (class_id == '1'){
                    dic[track_id].confidence = dic[track_id].confidence+conf;
                    dic[track_id].count = dic[track_id].count+1;
                }else{
                    dic[track_id].confidence = dic[track_id].confidence-conf;
                    dic[track_id].count = dic[track_id].count+1;
                }

            }else{
                // vote vote_a;
                dic[track_id] = vote{0,0.0};
                if (class_id == '1'){
                    dic[track_id].confidence = dic[track_id].confidence+conf;
                    dic[track_id].count = dic[track_id].count+1;
                }else{
                    dic[track_id].confidence = dic[track_id].confidence-conf;
                    dic[track_id].count = dic[track_id].count+1;
                }
            }
            
            // string output_dir = "../fluorescence/demo/001/crop/cell" + track_id; 
            // int mkdirretval;
            // //mkdirretval=light::mkpath("foo2/bar",0755);
            // //mkdirretval=light::mkpath("./lsl/foo2/bar");
            // mkdirretval=light::mkpath(output_dir);
            // std::cout << mkdirretval << '\n';
            // cv::imwrite("../fluorescence/demo/001/crop/cell" + track_id + "/imagex" + n + ".jpg", crop); // Crop image
            // ofstream outfile("../fluorescence/demo/001/id/id_" + track_id + ".txt", std::ios::app); //track_id <-> class_id 
            // outfile << (class_id+conf) << std::endl;
            // outfile.close();
            // ofstream outfiles("../fluorescence/demo/001/count/count_" + track_id + ".txt", std::ios::app); //track_id <-> class_id 
            // outfiles << (count) << std::endl;
            // outfiles.close();
         }

         if (x1>700){
            float mean = dic[track_id].confidence/dic[track_id].count;
            if(mean>0){
                return 1;
            }else{
                return 0;
            }
            
         }else{
            return -1;
         }
     }
     return -1;
}
            





        //  if (x1>0 && x2<800 && y1>0 && y2<400)
        //  {
        //     //printf("xywh %d, %d, %d, %d:\n",x1,y1,x2,y2);
        //     //sleep(2);
        //     cv::Rect rect(x1,y1,w,h); //(x1,y1,w,h)
        //     cv::Mat crop = image(rect);
        //     cv::imwrite("../cellball_demo/crop/cell" + id + "image" + n + ".jpg", crop); // Crop image
        //     ofstream outfile("../cellball_demo/id/cell" + id + "Class.txt", std::ios::app); //track_id <-> class_id 
        //     outfile << class_id << std::endl;
        //     outfile.close();
        //     // ofstream outfile("../fluorescence/demo/001/id/id_" + track_id + ".txt", std::ios::app); //track_id <-> class_id 
        //     // outfile << (class_id+conf) << std::endl;
        //  }

         


//  cv::Mat track_deepsort::display(cv::Mat frame)
//  {
//      cv::Mat image = frame;
//      int *id_array;
//      for (int i=0;i <mytracker.tracks.size();i++)
//      {
//          //printf("size %d\n",i);
//          float t = mytracker.tracks[i].latest_detection.tlwh[0];//center point t,l
//          float l = mytracker.tracks[i].latest_detection.tlwh[1];//latest_detection type: DETECTION_ROW in track.h
//          float w = mytracker.tracks[i].latest_detection.tlwh[2];//the parameters of ith tracker
//          float h = mytracker.tracks[i].latest_detection.tlwh[3];
//          int class_id = mytracker.tracks[i].latest_detection.type;
//          //id_array[i] = class_id;
//          //printf("classid: %d:\n",class_id);
//          cv::Point point(t+w/4,l-h/2);
//          int track_id = mytracker.tracks[i].track_id;
//          string id =std::to_string(mytracker.tracks[i].track_id);
//          float conf = mytracker.tracks[i].latest_detection.confidence;
//          string frame_id = "id:" + id;
//         //  cv::putText(frame, frame_id, point, cv::FONT_HERSHEY_PLAIN, 1.0,
//         //              cv::Scalar(0xFF, 0xFF, 0xFF), 2);
         
//          string n = std::to_string(mytracker.tracks[i].trail_list.size());
//          //printf("tlwh %.1f, %.1f, %.1f, %.1f:\n",t,l,w,h);
//          w = 80; // static w and h for bbox
//          h = 60;
//          int x1 = t - w/2; //(t,l) -> centroid t - w/2
//          int y1 = l - h/2;  //l - h/2
//          int x2 = x1 + w; //x1 + w (default)
//          int y2 = y1 + h; //y1 + h (default)

//          if (x1>0 && x2<800 && y1>0 && y2<400 && x2-x1==150 && y2-y1 == 150)
//          {
//             //printf("xywh %d, %d, %d, %d:\n",x1,y1,x2,y2);
//             //sleep(2);
//             int count = 0;
//             cv::Rect rect(x1,y1,w,h); //(x1,y1,w,h)
//             cv::Mat crop = image(rect);
//             int rows = crop.rows;
//             int cols = crop.cols;
//             for (int i=0; i<rows ; i++)  
//             {  
//                 for (int j=0; j<cols ; j++)  
//                 {  
//                     if(crop.at<uchar>(i,j) == 255)
//                     {
//                         count += 1;
//                     }  
//                 }  
//             }

//             if (dic.count(track_id)){
//                 if (class_id == '1'){
//                     dic[track_id].push_back(conf);
//                 }else{
//                     dic[track_id].push_back(-1*conf);
//                 }

//             }else{
//                 vector<float> vector_a;
//                 dic[track_id] = vector_a;
//                 if (class_id == '1'){
//                     dic[track_id].push_back(conf);
//                 }else{
//                     dic[track_id].push_back(-1*conf);
//                 }
//             }
            
//             // string output_dir = "../fluorescence/demo/001/crop/cell" + track_id; 
//             // int mkdirretval;
//             // //mkdirretval=light::mkpath("foo2/bar",0755);
//             // //mkdirretval=light::mkpath("./lsl/foo2/bar");
//             // mkdirretval=light::mkpath(output_dir);
//             // std::cout << mkdirretval << '\n';
//             // cv::imwrite("../fluorescence/demo/001/crop/cell" + track_id + "/imagex" + n + ".jpg", crop); // Crop image
//             // ofstream outfile("../fluorescence/demo/001/id/id_" + track_id + ".txt", std::ios::app); //track_id <-> class_id 
//             // outfile << (class_id+conf) << std::endl;
//             // outfile.close();
//             // ofstream outfiles("../fluorescence/demo/001/count/count_" + track_id + ".txt", std::ios::app); //track_id <-> class_id 
//             // outfiles << (count) << std::endl;
//             // outfiles.close();
//          }

//          if (x1>700){
//             vector<float> vector_c = dic[track_id];
//             float sum = std::accumulate(std::begin(vector_c), std::end(vector_c), 0.0);
//             float mean =  sum / vector_c.size();
//             if(mean>0){
//                 classification = 1;
//             }else{
//                 classification = -1;
//             }
//             std::cout << classification << '\n';
            
//          }
            





//         //  if (x1>0 && x2<800 && y1>0 && y2<400)
//         //  {
//         //     //printf("xywh %d, %d, %d, %d:\n",x1,y1,x2,y2);
//         //     //sleep(2);
//         //     cv::Rect rect(x1,y1,w,h); //(x1,y1,w,h)
//         //     cv::Mat crop = image(rect);
//         //     cv::imwrite("../cellball_demo/crop/cell" + id + "image" + n + ".jpg", crop); // Crop image
//         //     ofstream outfile("../cellball_demo/id/cell" + id + "Class.txt", std::ios::app); //track_id <-> class_id 
//         //     outfile << class_id << std::endl;
//         //     outfile.close();
//         //     // ofstream outfile("../fluorescence/demo/001/id/id_" + track_id + ".txt", std::ios::app); //track_id <-> class_id 
//         //     // outfile << (class_id+conf) << std::endl;
//         //  }

//          for (int j=1;j<mytracker.tracks[i].trail_list.size();j++) // Max length set in track.h TRACK_POINT_NUM
//          {
//             //cv::circle(frame, mytracker.tracks[i].trail_list[j], 5, cv::Scalar(0xFF, 0xFF, 0xFF));
//             cv::Point point_s = mytracker.tracks[i].trail_list[j-1];
//             cv::Point point_e = mytracker.tracks[i].trail_list[j];
//             point_s.x = point_s.x - w/2;
//             point_s.y = point_s.y - h/2;
//             point_e.x = point_e.x - w/2;
//             point_e.y = point_e.y - h/2;
//             cv::line(frame, point_s, point_e, cv::Scalar(0xFF, 0xFF, 0xFF), 3);
//          }
//      }
//      return frame;
//  }


