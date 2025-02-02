#ifndef TRACK_H
#define TRACK_H

#include "../DeepAppearanceDescriptor/dataType.h"

#include "kalmanfilter.h"
#include "../DeepAppearanceDescriptor/model.h"
#include "uuid/uuid.h"
#include "opencv2/opencv.hpp"
#include <string>
#define TRACK_POINT_NUM 200

class Track
{
    /*"""
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """*/
    enum TrackState {Tentative = 1, Confirmed, Deleted};

public:
    Track(KAL_MEAN& mean, KAL_COVA& covariance, int track_id,
          int n_init, int max_age, int type, DETECTION_ROW detection, const FEATURE& feature);
    void predit(KalmanFilter *kf);
    void update(KalmanFilter * const kf, const DETECTION_ROW &detection, int type);
    void mark_missed();
    bool is_confirmed();
    bool is_deleted();
    bool is_tentative();
    DETECTBOX to_tlwh();
    int time_since_update;
    int track_id;
    FEATURESS features;
    KAL_MEAN mean;
    KAL_COVA covariance;

    DETECTION_ROW latest_detection;
    std::vector<cv::Point> trail_list;
    int type;
    std::string uuid;
    //当前缓存的最优帧时间戳
    int best_timestamp;
    // 缓存的时间戳
    std::vector<int> cache_timestamp_list;
    //最优帧detection对象
    DETECTION_ROW best_img;

    int hits;
    int age;
    int _n_init;
    int _max_age;
    TrackState state;
private:
    void featuresAppendOne(const FEATURE& f);
};

#endif // TRACK_H
