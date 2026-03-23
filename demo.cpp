// demo.cpp — Drone detection using TensorFlow Lite
// Loads a .tflite model (raw outputs, no Flex ops), decodes boxes
// from anchors, runs NMS, draws bounding boxes, and saves the result.
//
// Build (example with CMake — see CMakeLists.txt):
//   mkdir build && cd build && cmake .. && make
//
// Dependencies:
//   • TensorFlow Lite C API
//   • OpenCV 4.x

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/c/c_api.h>

// ── Configuration ────────────────────────────────────────────────────
static const char* PATH_TO_MODEL       = "model/drone_detect.tflite";
static const char* PATH_TO_ANCHORS     = "model/anchors.bin";
static const char* PATH_TO_LABELS      = "model/object-detection.pbtxt";
static const float MIN_SCORE_THRESH    = 0.5f;
static const float NMS_IOU_THRESH      = 0.6f;
static const int   MAX_DETECTIONS      = 100;

// Box encoding scales from pipeline.config
static const float Y_SCALE = 10.0f;
static const float X_SCALE = 10.0f;
static const float H_SCALE = 5.0f;
static const float W_SCALE = 5.0f;

static const int NUM_ANCHORS = 1917;
static const int NUM_CLASSES = 2;  // background + 1 class (drone)

// ── Detection result ─────────────────────────────────────────────────
struct Detection {
    float ymin, xmin, ymax, xmax;
    float score;
    int   cls;
};

// ── Parse label map (.pbtxt) ─────────────────────────────────────────
static std::map<int, std::string> parse_labelmap(const char* path)
{
    std::ifstream ifs(path);
    if (!ifs) { std::cerr << "Cannot open label map: " << path << "\n"; std::exit(1); }
    std::string text((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());

    std::map<int, std::string> category_index;
    std::regex item_re(R"(item\s*\{([\s\S]*?)\})");
    std::regex id_re(R"(id\s*:\s*(\d+))");
    std::regex name_re(R"(name\s*:\s*['"](.+?)['"])");

    auto it  = std::sregex_iterator(text.begin(), text.end(), item_re);
    auto end = std::sregex_iterator();
    for (; it != end; ++it) {
        std::string block = (*it)[1].str();
        std::smatch id_m, name_m;
        if (std::regex_search(block, id_m, id_re) &&
            std::regex_search(block, name_m, name_re)) {
            category_index[std::stoi(id_m[1].str())] = name_m[1].str();
        }
    }
    return category_index;
}

// ── Load anchor boxes from binary file ───────────────────────────────
static std::vector<float> load_anchors(const char* path, int num_anchors)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) { std::cerr << "Cannot open anchors: " << path << "\n"; std::exit(1); }
    std::vector<float> anchors(num_anchors * 4);
    ifs.read(reinterpret_cast<char*>(anchors.data()), anchors.size() * sizeof(float));
    return anchors;
}

// ── Sigmoid ──────────────────────────────────────────────────────────
static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

// ── Decode raw boxes using anchors ───────────────────────────────────
// raw_boxes: [num_anchors, 4] encoded as [ty, tx, th, tw]
// anchors:   [num_anchors, 4] as [cy, cx, h, w]
// output:    [num_anchors, 4] as [ymin, xmin, ymax, xmax] normalised
static void decode_boxes(const float* raw_boxes, const float* anchors,
                         int num_anchors, std::vector<float>& decoded)
{
    decoded.resize(num_anchors * 4);
    for (int i = 0; i < num_anchors; ++i) {
        float a_cy = anchors[i * 4 + 0];
        float a_cx = anchors[i * 4 + 1];
        float a_h  = anchors[i * 4 + 2];
        float a_w  = anchors[i * 4 + 3];

        float ty = raw_boxes[i * 4 + 0] / Y_SCALE;
        float tx = raw_boxes[i * 4 + 1] / X_SCALE;
        float th = raw_boxes[i * 4 + 2] / H_SCALE;
        float tw = raw_boxes[i * 4 + 3] / W_SCALE;

        float cy = ty * a_h + a_cy;
        float cx = tx * a_w + a_cx;
        float h  = std::exp(th) * a_h;
        float w  = std::exp(tw) * a_w;

        decoded[i * 4 + 0] = cy - h / 2.0f;  // ymin
        decoded[i * 4 + 1] = cx - w / 2.0f;  // xmin
        decoded[i * 4 + 2] = cy + h / 2.0f;  // ymax
        decoded[i * 4 + 3] = cx + w / 2.0f;  // xmax
    }
}

// ── Non-Maximum Suppression ──────────────────────────────────────────
static float iou(const Detection& a, const Detection& b)
{
    float inter_ymin = std::max(a.ymin, b.ymin);
    float inter_xmin = std::max(a.xmin, b.xmin);
    float inter_ymax = std::min(a.ymax, b.ymax);
    float inter_xmax = std::min(a.xmax, b.xmax);

    float inter_area = std::max(0.0f, inter_ymax - inter_ymin) *
                       std::max(0.0f, inter_xmax - inter_xmin);
    float area_a = (a.ymax - a.ymin) * (a.xmax - a.xmin);
    float area_b = (b.ymax - b.ymin) * (b.xmax - b.xmin);
    float union_area = area_a + area_b - inter_area;

    return (union_area > 0.0f) ? inter_area / union_area : 0.0f;
}

static std::vector<Detection> nms(std::vector<Detection>& dets,
                                   float iou_thresh, int max_dets)
{
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.score > b.score;
              });

    std::vector<Detection> keep;
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t i = 0; i < dets.size() && (int)keep.size() < max_dets; ++i) {
        if (suppressed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && iou(dets[i], dets[j]) > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }
    return keep;
}

// ── Post-process: decode + sigmoid + filter + NMS ────────────────────
static std::vector<Detection> postprocess(const float* raw_boxes,
                                           const float* raw_scores,
                                           const float* anchors,
                                           int num_anchors, int num_classes)
{
    // Decode boxes
    std::vector<float> decoded;
    decode_boxes(raw_boxes, anchors, num_anchors, decoded);

    // Collect detections above threshold (skip class 0 = background)
    std::vector<Detection> candidates;
    for (int i = 0; i < num_anchors; ++i) {
        for (int c = 1; c < num_classes; ++c) {
            float score = sigmoid(raw_scores[i * num_classes + c]);
            if (score < MIN_SCORE_THRESH) continue;

            Detection det;
            det.ymin  = decoded[i * 4 + 0];
            det.xmin  = decoded[i * 4 + 1];
            det.ymax  = decoded[i * 4 + 2];
            det.xmax  = decoded[i * 4 + 3];
            det.score = score;
            det.cls   = c;
            candidates.push_back(det);
        }
    }

    return nms(candidates, NMS_IOU_THRESH, MAX_DETECTIONS);
}

// ── Main ─────────────────────────────────────────────────────────────
// ── Helper: check if string is a non-negative integer ─────────────
static bool is_camera_index(const std::string& s)
{
    return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path | camera_index | gst-pipeline> [output_path]\n"
                  << "  camera_index : 0, 1, ...  (e.g. /dev/video0)\n"
                  << "  gst-pipeline : e.g. 'libcamerasrc ! video/x-raw,width=640,height=480 ! videoconvert ! appsink'\n";
        return 1;
    }
    const std::string input_arg = argv[1];
    const char* output_path = (argc >= 3) ? argv[2] : nullptr;
    bool live_mode = is_camera_index(input_arg) || input_arg.find('!') != std::string::npos;

    // ── 1. Parse labels ──────────────────────────────────────────────
    auto category_index = parse_labelmap(PATH_TO_LABELS);
    std::cout << "Labels:";
    for (auto& kv : category_index) std::cout << " {" << kv.first << ": " << kv.second << "}";
    std::cout << "\n";

    // ── 2. Load anchors ──────────────────────────────────────────────
    auto anchors = load_anchors(PATH_TO_ANCHORS, NUM_ANCHORS);
    std::cout << "Loaded " << NUM_ANCHORS << " anchors.\n";

    // ── 3. Load TFLite model ─────────────────────────────────────────
    std::cout << "Loading model...\n";
    TfLiteModel* model = TfLiteModelCreateFromFile(PATH_TO_MODEL);
    if (!model) {
        std::cerr << "Failed to load model: " << PATH_TO_MODEL << "\n";
        return 1;
    }

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 4);

    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
    if (!interpreter) {
        std::cerr << "Failed to create interpreter\n";
        return 1;
    }

    if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors\n";
        return 1;
    }
    std::cout << "Model loaded.\n";

    // ── Query input tensor dimensions ────────────────────────────────
    const TfLiteTensor* input_tensor_info = TfLiteInterpreterGetInputTensor(interpreter, 0);
    int input_h = TfLiteTensorDim(input_tensor_info, 1);
    int input_w = TfLiteTensorDim(input_tensor_info, 2);
    int input_c = TfLiteTensorDim(input_tensor_info, 3);
    std::cout << "Model input: " << input_w << "x" << input_h << "x" << input_c << "\n";

    // ── 4. Open video / camera ────────────────────────────────────────
    cv::VideoCapture cap;
    if (is_camera_index(input_arg)) {
    int cam_idx = std::stoi(input_arg);
    cap.open(cam_idx, cv::CAP_V4L2);
    } else if (input_arg.find('!') != std::string::npos) {
        cap.open(input_arg, cv::CAP_GSTREAMER);
    } else {
        cap.open(input_arg);
    }

    if (!cap.isOpened()) {
        std::cerr << "Could not open input: " << input_arg << "\n";
        return 1;
    }

    cv::Mat test_frame;
    if (!cap.read(test_frame) || test_frame.empty()) {
        std::cerr << "Opened input, but failed to grab a frame: " << input_arg << "\n";
        return 1;
    }

    int frame_w  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_h  = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps   = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;  // default for cameras that don't report FPS
    int total    = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << (live_mode ? "Camera: " : "Video: ")
              << frame_w << "x" << frame_h << " @ " << fps << " fps";
    if (!live_mode) std::cout << ", " << total << " frames";
    std::cout << "\n";

    cv::VideoWriter writer;
    if (output_path) {
        writer.open(output_path,
                    cv::VideoWriter::fourcc('m','p','4','v'),
                    fps, cv::Size(frame_w, frame_h));
        if (!writer.isOpened()) {
            std::cerr << "Could not open output video for writing: " << output_path << "\n";
            return 1;
        }
    }

    if (live_mode) {
        std::cout << "Live mode — press 'q' to quit.\n";
    }

    // ── 5. Process frames ────────────────────────────────────────────
    cv::Mat frame_bgr, frame_rgb, frame_resized;
    int frame_no = 0;
    double total_ms = 0.0;

    while (cap.read(frame_bgr)) {
        ++frame_no;

        cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);
        cv::resize(frame_rgb, frame_resized, cv::Size(input_w, input_h));

        // Normalize to [-1, 1]:  pixel * (2.0/255.0) - 1.0
        cv::Mat frame_float;
        frame_resized.convertTo(frame_float, CV_32FC3, 2.0 / 255.0, -1.0);

        TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
        TfLiteTensorCopyFromBuffer(input_tensor, frame_float.data,
                                   (size_t)input_h * input_w * input_c * sizeof(float));

        auto t_start = std::chrono::high_resolution_clock::now();

        if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
            std::cerr << "Inference failed on frame " << frame_no << "\n";
            continue;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        total_ms += elapsed_ms;

        // ── Extract raw outputs ──────────────────────────────────────
        // Output 0: raw_boxes  [1, 1917, 4]  float32 (encoded)
        // Output 1: raw_scores [1, 1917, 2]  float32 (logits)
        const TfLiteTensor* t_boxes  = TfLiteInterpreterGetOutputTensor(interpreter, 0);
        const TfLiteTensor* t_scores = TfLiteInterpreterGetOutputTensor(interpreter, 1);

        const float* raw_boxes  = (const float*)TfLiteTensorData(t_boxes);
        const float* raw_scores = (const float*)TfLiteTensorData(t_scores);

        // ── Decode + NMS ─────────────────────────────────────────────
        auto detections = postprocess(raw_boxes, raw_scores,
                                       anchors.data(), NUM_ANCHORS, NUM_CLASSES);

        // ── Draw bounding boxes ──────────────────────────────────────
        int h = frame_bgr.rows;
        int w = frame_bgr.cols;
        int count = (int)detections.size();

        for (auto& det : detections) {
            int left   = (int)(det.xmin * w);
            int top    = (int)(det.ymin * h);
            int right  = (int)(det.xmax * w);
            int bottom = (int)(det.ymax * h);

            std::string label;
            auto it = category_index.find(det.cls);
            if (it != category_index.end()) label = it->second;
            else {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "class %d", det.cls);
                label = buf;
            }

            char text[128];
            std::snprintf(text, sizeof(text), "%s: %.0f%%", label.c_str(), det.score * 100.0f);

            cv::rectangle(frame_bgr, cv::Point(left, top), cv::Point(right, bottom),
                          cv::Scalar(0, 255, 0), 3);
            cv::putText(frame_bgr, text, cv::Point(left, top - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        }

        if (writer.isOpened()) writer.write(frame_bgr);

        if (live_mode) {
            cv::imshow("Drone Detection", frame_bgr);
            if (cv::waitKey(1) == 'q') break;
            std::printf("\rFrame %d  |  %.1f ms  |  %d detection(s)  ",
                        frame_no, elapsed_ms, count);
        } else {
            std::printf("\rFrame %d/%d  |  %.1f ms  |  %d detection(s)",
                        frame_no, total, elapsed_ms, count);
        }
        std::fflush(stdout);
    }

    std::cout << "\n";
    cap.release();
    if (writer.isOpened()) writer.release();
    if (live_mode) cv::destroyAllWindows();

    if (frame_no > 0) {
        std::printf("Done. %d frames processed, avg %.1f ms/frame.\n",
                    frame_no, total_ms / frame_no);
    }
    if (output_path) std::cout << "Saved " << output_path << "\n";

    // ── Cleanup ──────────────────────────────────────────────────────
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    return 0;
}
