#include <exception>
#include <boost/program_options.hpp>
#include <type_traits>

#include <metavision/hal/utils/hal_exception.h>
#include <metavision/hal/facilities/i_trigger_in.h>
#include <metavision/hal/facilities/i_trigger_out.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <metavision/hal/facilities/i_monitoring.h>
#include <metavision/hal/facilities/i_event_rate_noise_filter_module.h>
#include <metavision/hal/facilities/i_device_control.h>
#include <metavision/hal/facilities/i_geometry.h>
#include <metavision/hal/facilities/i_decoder.h>
#include <metavision/hal/facilities/i_event_decoder.h>
#include <metavision/hal/facilities/i_plugin_software_info.h>
#include <metavision/hal/facilities/i_hw_identification.h>
#include <metavision/hal/device/device.h>
#include <metavision/hal/device/device_discovery.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/events/event_ext_trigger.h>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "preprocess.h"
#include <condition_variable>
#include "DeepSORT/track_deepsort.h"
#include <unistd.h> 

#include <functional>
#include <metavision/sdk/core/algorithms/polarity_filter_algorithm.h>
#include <metavision/sdk/core/pipeline/pipeline.h>
#include <metavision/sdk/core/pipeline/frame_generation_stage.h>
#include <metavision/sdk/core/pipeline/frame_composition_stage.h>
#include <metavision/sdk/driver/pipeline/camera_stage.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/pipeline/frame_display_stage.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/ui/utils/window.h>
#include <libserial/SerialPort.h>
#include <cstdlib>
#include <fstream>

// constexpr const char* const SERIAL_PORT_2 = "/dev/ttyUSB0" ;
// LibSerial::SerialPort serial_port ;

namespace po = boost::program_options;

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0
#define CONF_THRESH 0.1
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !

#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#define SLEEP_E 800
#define MAT_ROWS 1000
#define MAT_COLS 1280
#define FPN_PATH    "../Samples/config/FPN_2.txt"
int count = 0;
#ifdef _WIN32
#include <windows.h>
#else
#include<unistd.h>
#include <signal.h>
#endif


std::queue<cv::Mat> proDataQueue;
std::mutex rawDataMutex;

std::condition_variable repo_not_full;
std::condition_variable repo_not_empty;

static const int repository_size = 8;
static const int item_total = 20;

cv::Mat item_buffer[repository_size];

static std::size_t read_position = 0;
static std::size_t write_position = 0;

std::mutex rawDataMutex_res;

std::condition_variable repo_not_full_res;
std::condition_variable repo_not_empty_res;

static const int repository_size_res = repository_size;
static const int item_total_res = 20;

cv::Mat item_buffer_res[repository_size_res];

static std::size_t read_position_res = 0;
static std::size_t write_position_res = 0;
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

static std::size_t read_position_yolo = 0;
static std::size_t write_position_yolo = 0;

static int get_width(int x, float gw, int divisor = 8) {
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}

void produce_item(cv::Mat i)
{
    
	std::unique_lock<std::mutex> lck(rawDataMutex);
	while (((write_position + 1) % repository_size) == read_position)
	{
		std::cout << "Producer is waiting for an empty slot..." << std::endl;
		repo_not_full.wait(lck);
	}                           
    
    //std::cout << "Producer is waiting for an empty slot..." << std::endl;
    item_buffer[write_position] = i;
	//item_buffer[(read_position+1)%repository_size] = i;
	write_position++;
    //write_position = (read_position+1)%repository_size;
    //std::cout << "Producor " << write_position << std::endl;
    //std::cout << "Consumer " << read_position << std::endl;
	if (write_position == repository_size)
	{
		write_position = 0;
	}
 
	repo_not_empty.notify_all();
 
	
}

cv::Mat consume_item()
{
	cv::Mat data;
    
	std::unique_lock<std::mutex> lck(rawDataMutex);
	while (write_position == read_position)
	{
		std::cout << "Consumer is waiting for items..." << std::endl;
		repo_not_empty.wait(lck);
	}                           
    
	data = item_buffer[read_position];
	read_position++;
 
	if (read_position >= repository_size)
	{
		read_position = 0;
	}
		
	repo_not_full.notify_all();
	//lck.unlock();
 
	return data;
}


void produce_item_res(cv::Mat i)
{
    
	item_buffer_res[write_position_res] = i;
	write_position_res++;
 
	if (write_position_res == repository_size)
	{
		write_position_res = 0;
	}
 
	repo_not_empty_res.notify_all();
}

cv::Mat consume_item_res()
{
	cv::Mat data;
	std::unique_lock<std::mutex> lck(rawDataMutex_res);
	while (write_position_res == read_position_res)
	{
		std::cout << "Consumer is waiting for res items..." << std::endl;
		repo_not_empty_res.wait(lck);
	}                        

    data = item_buffer_res[write_position_res-1];
 
	if (read_position_res >= repository_size)
	{
		read_position_res = 0;
	}
		
	repo_not_full_res.notify_all();
	//lck.unlock();
 
	return data;
}



namespace po = boost::program_options;

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);
    /* ------ yolov5 backbone------ */
    auto conv0 = convBlock(network, weightMap, *data,  get_width(64, gw), 6, 2, 1,  "model.0");
    assert(conv0);
    auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto bottleneck_csp8 = C3(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto spp9 = SPPF(network, weightMap, *bottleneck_csp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, "model.9");
    /* ------ yolov5 head ------ */
    auto conv10 = convBlock(network, weightMap, *spp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

    ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

    ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

ICudaEngine* build_engine_p6(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);
    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
    assert(data);
    
    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    /* ------ yolov5 backbone------ */
    auto conv0 = convBlock(network, weightMap, *data,  get_width(64, gw), 6, 2, 1,  "model.0");
    auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto c3_2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *c3_2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto c3_4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *c3_4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto c3_6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *c3_6->getOutput(0), get_width(768, gw), 3, 2, 1, "model.7");
    auto c3_8 = C3(network, weightMap, *conv7->getOutput(0), get_width(768, gw), get_width(768, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto conv9 = convBlock(network, weightMap, *c3_8->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.9");
    auto c3_10 = C3(network, weightMap, *conv9->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), true, 1, 0.5, "model.10");
    auto sppf11 = SPPF(network, weightMap, *c3_10->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, "model.11");

    /* ------ yolov5 head ------ */
    auto conv12 = convBlock(network, weightMap, *sppf11->getOutput(0), get_width(768, gw), 1, 1, 1, "model.12");
    auto upsample13 = network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(ResizeMode::kNEAREST);
    upsample13->setOutputDimensions(c3_8->getOutput(0)->getDimensions());
    ITensor* inputTensors14[] = { upsample13->getOutput(0), c3_8->getOutput(0) };
    auto cat14 = network->addConcatenation(inputTensors14, 2);
    auto c3_15 = C3(network, weightMap, *cat14->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.15");

    auto conv16 = convBlock(network, weightMap, *c3_15->getOutput(0), get_width(512, gw), 1, 1, 1, "model.16");
    auto upsample17 = network->addResize(*conv16->getOutput(0));
    assert(upsample17);
    upsample17->setResizeMode(ResizeMode::kNEAREST);
    upsample17->setOutputDimensions(c3_6->getOutput(0)->getDimensions());
    ITensor* inputTensors18[] = { upsample17->getOutput(0), c3_6->getOutput(0) };
    auto cat18 = network->addConcatenation(inputTensors18, 2);
    auto c3_19 = C3(network, weightMap, *cat18->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.19");

    auto conv20 = convBlock(network, weightMap, *c3_19->getOutput(0), get_width(256, gw), 1, 1, 1, "model.20");
    auto upsample21 = network->addResize(*conv20->getOutput(0));
    assert(upsample21);
    upsample21->setResizeMode(ResizeMode::kNEAREST);
    upsample21->setOutputDimensions(c3_4->getOutput(0)->getDimensions());
    ITensor* inputTensors21[] = { upsample21->getOutput(0), c3_4->getOutput(0) };
    auto cat22 = network->addConcatenation(inputTensors21, 2);
    auto c3_23 = C3(network, weightMap, *cat22->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.23");

    auto conv24 = convBlock(network, weightMap, *c3_23->getOutput(0), get_width(256, gw), 3, 2, 1, "model.24");
    ITensor* inputTensors25[] = { conv24->getOutput(0), conv20->getOutput(0) };
    auto cat25 = network->addConcatenation(inputTensors25, 2);
    auto c3_26 = C3(network, weightMap, *cat25->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.26");

    auto conv27 = convBlock(network, weightMap, *c3_26->getOutput(0), get_width(512, gw), 3, 2, 1, "model.27");
    ITensor* inputTensors28[] = { conv27->getOutput(0), conv16->getOutput(0) };
    auto cat28 = network->addConcatenation(inputTensors28, 2);
    auto c3_29 = C3(network, weightMap, *cat28->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.29");

    auto conv30 = convBlock(network, weightMap, *c3_29->getOutput(0), get_width(768, gw), 3, 2, 1, "model.30");
    ITensor* inputTensors31[] = { conv30->getOutput(0), conv12->getOutput(0) };
    auto cat31 = network->addConcatenation(inputTensors31, 2);
    auto c3_32 = C3(network, weightMap, *cat31->getOutput(0), get_width(2048, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.32");

    /* ------ detect ------ */
    IConvolutionLayer* det0 = network->addConvolutionNd(*c3_23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.0.weight"], weightMap["model.33.m.0.bias"]);
    IConvolutionLayer* det1 = network->addConvolutionNd(*c3_26->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.1.weight"], weightMap["model.33.m.1.bias"]);
    IConvolutionLayer* det2 = network->addConvolutionNd(*c3_29->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.2.weight"], weightMap["model.33.m.2.bias"]);
    IConvolutionLayer* det3 = network->addConvolutionNd(*c3_32->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.3.weight"], weightMap["model.33.m.3.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, "model.33", std::vector<IConvolutionLayer*>{det0, det1, det2, det3});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, bool& is_p6, float& gd, float& gw, std::string& wts_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = nullptr;
    if (is_p6) {
        engine = build_engine_p6(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    } else {
        engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    }
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir) {
    if (argc < 4) return false;
    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto net = std::string(argv[4]);
        if (net[0] == 'n') {
            gd = 0.33;
            gw = 0.25;
        } else if (net[0] == 's') {
            gd = 0.33;
            gw = 0.50;
        } else if (net[0] == 'm') {
            gd = 0.67;
            gw = 0.75;
        } else if (net[0] == 'l') {
            gd = 1.0;
            gw = 1.0;
        } else if (net[0] == 'x') {
            gd = 1.33;
            gw = 1.25;
        } else if (net[0] == 'c' && argc == 7) {
            gd = atof(argv[5]);
            gw = atof(argv[6]);
        } else {
            return false;
        }
        if (net.size() == 2 && net[1] == '6') {
            is_p6 = true;
        }
    } else if (std::string(argv[1]) == "-d" && argc == 4) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}

void prophesee_sensor(){
    std::string in_raw_file_path;

    const std::string program_desc("Code sample demonstrating how to use Metavision SDK CV to filter events\n"
                                   "and show a frame combining unfiltered and filtered events.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-raw-file,i", po::value<std::string>(&in_raw_file_path), "Path to input RAW file. If not specified, the camera live stream is used.")
        ;
    // clang-format on

    po::variables_map vm;
    try {
        // po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return ;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return ;
    }

    // A pipeline for which all added stages will automatically be run in their own processing threads (if applicable)
    Metavision::Pipeline p(true);

    // Construct a camera from a file or a live stream
    Metavision::Camera cam;
    if (!in_raw_file_path.empty()) {
        cam = Metavision::Camera::from_file(in_raw_file_path);
    } else {
        cam = Metavision::Camera::from_first_available();
    }

    const auto w = cam.geometry().width();
    const auto h = cam.geometry().height();
    const std::uint32_t acc = 1000;
    double fps = 1000;
    auto frame_gen = Metavision::PeriodicFrameGenerationAlgorithm(w, h, acc, fps);

    //Metavision::Window window("Frames", w, h, Metavision::BaseWindow::RenderMode::BGR);
  
    frame_gen.set_output_callback([&](Metavision::timestamp ts, cv::Mat &frame){
        //window.show(frame);
        // std::cout<<ts<<std::endl;
        
    });
    auto event_begin_t = 10000;

    // cam.cd().

    cv::Mat mat = cv::Mat::zeros(cv::Size(1280, 720), CV_8UC1);
    auto sender_start = std::chrono::system_clock::now();
    cam.cd().add_callback([&](const Metavision::EventCD *begin, const Metavision::EventCD *end){
        // frame_gen.process_events(begin, end);
        
        for(auto it = begin; it !=end; ++it){
            auto event = *it;
            // std::cout<<(*begin).t<<std::endl;
            // mat.at<uchar>(720 - event.y - 1, 1280 - event.x - 1) = 255;
            if (event.p == 1){
                mat.at<uchar>(720 - event.y - 1, 1280 - event.x - 1) = 255;
            }else{
                mat.at<uchar>(720 - event.y - 1, 1280 - event.x - 1) = 125;
            }
            if (event.t - event_begin_t >1000){
                
                cv::Mat roi = mat(cv::Range(150,550),cv::Range(200,1100));

                // send mat
                std::stringstream ss;
                ss << "/home/weihua/Desktop/cell-dataset/" << std::setw(10) << std::setfill('0') << count <<".png";
                cv::imwrite(ss.str(),roi);
                ss.clear();
                count = count+1;
                // if (count >1){
                //     count=0;
                // }

                produce_item(roi);

                auto sender_end = std::chrono::system_clock::now();
                std::cout << "sending time: " << std::chrono::duration_cast<std::chrono::microseconds>(sender_end - sender_start).count() << "us; " << std::endl;
                sender_start = sender_end;
                mat = cv::Mat::zeros(cv::Size(1280, 720), CV_8UC1);
                event_begin_t = event.t;
            }
            // if (event.p == 1){
            //     mat.at<uchar>(720 - event.y - 1, 1280 - event.x - 1) = cv::Vec3b(236, 223, 216);
            // }else{
            //     mat.at<uchar>(720 - event.y - 1, 1280 - event.x - 1) = cv::Vec3b(201, 126, 64);
            // }
        }
        // auto end_pre = std::chrono::system_clock::now();
        // if(std::chrono::duration_cast<std::chrono::microseconds>(end_pre - start_pre).count() >= 1000)
        // {
        //     mat = cv::Mat::zeros(cv::Size(1280, 720), CV_8UC1);
        //     //std::cout << "pc time: " << std::chrono::duration_cast<std::chrono::microseconds>(end_pre - start_pre).count() << "us; " 
        //     //<<"n event " << end - begin <<"; "<<"event time span " << end_t - start_t << "us;" << std::endl;
        //     start_pre = end_pre;
        // }        
    });

    cam.start();
    
    while(cam.is_running()){
        
        // if(std::chrono::duration_cast<std::chrono::microseconds>(sender_end - sender_start).count() >= 1000)
        // {
            
        
        // }
        Metavision::EventLoop::poll_and_dispatch(0);
    }
    cam.stop();
    return;


}

void get_detections(DETECTBOX box, float confidence, int type, DETECTIONS &d, int timestamp) {
    DETECTION_ROW tmpRow;
    tmpRow.tlwh = box;//DETECTBOX(x, y, w, h);
    tmpRow.type = type;
    tmpRow.timestamp = timestamp;
    tmpRow.shelter = false;
    tmpRow.confidence = confidence;
    d.push_back(tmpRow);
}

cv::Mat convertTo3Channels(const cv::Mat& binImg)
{
    cv::Mat three_channel = cv::Mat::zeros(binImg.rows,binImg.cols,CV_8UC3);
    std::vector<cv::Mat> channels;
    for (int i=0;i<3;i++)
    {
        channels.push_back(binImg);
    }
    merge(channels,three_channel);
    return three_channel;
}

int yoloDetector(std::string wts_name, std::string engine_name, std::string img_dir){

    // // Open the Serial Port at the desired hardware port.
    // serial_port.Open(SERIAL_PORT_2) ;

    // // Set the baud rate of the serial port.
    // serial_port.SetBaudRate(LibSerial::BaudRate::BAUD_115200) ;

    // // Set the number of data bits.
    // serial_port.SetCharacterSize(LibSerial::CharacterSize::CHAR_SIZE_8) ;

    // // Turn off hardware flow control.
    // serial_port.SetFlowControl(LibSerial::FlowControl::FLOW_CONTROL_NONE) ;

    // // Disable parity.
    // serial_port.SetParity(LibSerial::Parity::PARITY_NONE) ;

    // // Set the number of stop bits.
    // serial_port.SetStopBits(LibSerial::StopBits::STOP_BITS_1) ;
    // std::cout << "\nOutputaswdefasdfadcfszdfds:\n\n";
    track_deepsort deepsort;
    cudaSetDevice(DEVICE);
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;
    
    // create a model using the API directly and serialize it to a stream
    if (!wts_name.empty()) {
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream, is_p6, gd, gw, wts_name);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    }

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return -1;
    }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    float* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    uint8_t* img_host = nullptr;
    uint8_t* img_device = nullptr;
    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    int fcount = 0;
    std::vector<cv::Mat> imgs_buffer(BATCH_SIZE);


    
    

    int yolo_count = 0;
    // while(true){
    for (int f = 0; f < (int)file_names.size(); f++) {
        // yolo_count++;
        fcount++;
        //if (fcount < BATCH_SIZE && f + 1 != (int)file_names.size()) continue;
        auto start_pre = std::chrono::system_clock::now();
        float* buffer_idx = (float*)buffers[inputIndex];
        // cv::Mat img_temp = consume_item().clone();
        // cv::Mat img = convertTo3Channels(img_temp);
        // cv::Mat img = consume_item().clone();
        //if (img.empty()) continue;
        //if (rawDataQueue.size()<2) continue;
        /*std::cout<<"queue "<< rawDataQueue.size() <<std::endl;
        if (rawDataQueue.size()<2){
            std::cout<<"empty!"<<std::endl;
            usleep(1000);
            continue;
        }else{
            std::cout<<"queue "<< rawDataQueue.size() <<std::endl;
            //cv::Mat img = rawDataQueue.back();
			//rawDataQueue.pop();
			//cv::imshow("Event Gray Pic", img);
		    //cv::waitKey(1);
        }*/
        cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + 0]);
        for (int b = 0; b < fcount; b++) {
            std::cout<<" "<<fcount<<" "<<b<<std::endl;
            
            // cv::Mat img_temp = consume_item().clone();
            // cv::Mat img = convertTo3Channels(img_temp);
            //cv::Mat img = rawDataQueue.front().clone();
			//rawDataQueue.pop();
            
            imgs_buffer[b] = img;
            size_t  size_image = img.cols * img.rows * 3;
            size_t  size_image_dst = INPUT_H * INPUT_W * 3;
            //copy data to pinned memory
            memcpy(img_host,img.data,size_image);
            //copy data to device memory
            CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
            preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);       
            buffer_idx += size_image_dst;
            //std::cout<<"hello"<<buffer_idx<<std::endl;
        }
        // Run inference
        auto end_pre = std::chrono::system_clock::now();
        auto start_yolo = std::chrono::system_clock::now();
        //auto start = std::chrono::system_clock::now();
        doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
        
        std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
        for (int b = 0; b < fcount; b++) {
            auto& res = batch_res[b];
            nms(res, &prob[b * OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);
        }
        auto end_yolo = std::chrono::system_clock::now();
        auto start_sort = std::chrono::system_clock::now();
        
        for (int b = 0; b < fcount; b++) 
        {
            
            auto& res = batch_res[b];
            cv::Mat img = imgs_buffer[b];
            DETECTIONS detections;
            DETECTION_ROW OBJ;
            int timestamp =0;
            //By bbox sequence
            for (size_t j = 0; j < res.size(); j++) 
            {
                std::string id = std::to_string((int)res[j].class_id);
                //id = "Class:" + id;
                
                cv::Rect r = get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                cv::putText(img, id, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0xFF, 0xFF, 0xFF), 2);//class id text
                
                //(x,y,w,h) need to transform from 640x640 to 1280x800
                float x =r.x+round(r.width/2); 
                float y =r.y+round(r.height/2);
                float w =r.width;
                float h =r.height;
                int class_id =res[j].class_id;
                float conf  =res[j].conf;
                //printf("xywh %.1f, %.1f, %.1f, %.1f:\n",x,y,w,h);
                get_detections(DETECTBOX(x, y, w, h), conf, class_id, detections, timestamp);
            }
            timestamp++;
            deepsort.run(detections);

            int classification = 1;
            classification =  deepsort.serial(img);
            if (classification>-1){
                if(classification==1){
                    // serial_port.WriteByte('1') ;
                    std::cout<<"xserial::::::::------------"<<std::endl;
                }else{
                    // serial_port.WriteByte('0') ;
                    std::cout<<"xserial:::::::::::::::::::::::::::::::::::::::"<<std::endl;
                }
                // char data_byte  = classification;
                // serial_port.WriteByte('1') ;
                // std::cout<<"xserial::::::::"<<data_byte<<std::endl;
            }else{
                std::cout<<"xserial:without"<<std::endl;
            }
            
            produce_item_res(img);
            cv::imwrite("../output/" + file_names[f - fcount + 1 + b].substr(0,file_names[f - fcount + 1 + b].rfind(".")) + ".jpg", img);
            //sleep(1);
            //printf("File_names:%s\n\n",file_names[f - fcount + 1 + b].c_str());
        }
        auto end_sort = std::chrono::system_clock::now();
        
        // produce_item_yolo(yolo_result(img, batch_res));
        
        fcount = 0;
    }

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}



void imshower(){
    while(true){
        
        cv::Mat img = consume_item_res().clone();
        cv::imshow("Yolo Result", img);
        cv::waitKey(1);
    }
}

int main(int argc, char** argv) {
    //cudaSetDevice(DEVICE);
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;
    std::string wts_name = "";
    std::string engine_name = "";
    
    std::string img_dir;
    
    if (!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, img_dir)) {

        std::cerr << "arguments not right!" << std::endl;
        std::endl;
        return -1;
    }

    cv::Mat mat = cv::Mat::zeros(cv::Size(1280, 800), CV_8UC1);
    int count = 0;
    while(count<15){
        item_buffer[count%repository_size] = mat;
        count++;
    }
    
	// std::thread t0(prophesee_sensor);
    //usleep(1000000);
    std::thread t1(yoloDetector, wts_name, engine_name, img_dir);
    // std::thread t2(nms_sort);
    std::thread t3(imshower);

    // t0.join();
    t1.join();
    // t2.join();
    t3.join();
    


    // Print histogram of the output distribution
    //std::cout << "\nOutput:\n\n";
    //for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    //{
    //    std::cout << prob[i] << ", ";
    //    if (i % 10 == 0) std::cout << std::endl;
    //}
    //std::cout << std::endl;

    return 0;
}
