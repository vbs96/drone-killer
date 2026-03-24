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
#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <string>
#include <ctime>
#include <vector>

#include <opencv2/opencv.hpp>
#include <tensorflow/lite/c/c_api.h>
#ifdef DRONE_DEMO_HAS_CURL
#include <curl/curl.h>
#endif

// ── Configuration ────────────────────────────────────────────────────
static const char* PATH_TO_MODEL       = "model/drone_detect.tflite";
static const char* PATH_TO_ANCHORS     = "model/anchors.bin";
static const char* PATH_TO_LABELS      = "model/object-detection.pbtxt";
static const float MIN_SCORE_THRESH    = 0.5f;
static const float NMS_IOU_THRESH      = 0.6f;
static const int   MAX_DETECTIONS      = 100;
static const int   DETECTION_LOG_COOLDOWN_MS = 3000;
static const char* DETECTION_FRAME_DIR = "detections";
static const char* DEFAULT_EVENTS_SERVER_URL = "";
static const double GPS_BUCHAREST_LAT = 44.436142;
static const double GPS_BUCHAREST_LON = 26.102684;

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

static std::string json_escape(const std::string& in)
{
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

static std::string iso8601_utc_now()
{
    std::time_t now = std::time(nullptr);
    std::tm tm_utc{};
#ifdef _WIN32
    gmtime_s(&tm_utc, &now);
#else
    gmtime_r(&now, &tm_utc);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    return std::string(buf);
}

#ifdef DRONE_DEMO_HAS_CURL
static size_t curl_write_cb(char* ptr, size_t size, size_t nmemb, void* userdata)
{
    const size_t total = size * nmemb;
    auto* out = static_cast<std::string*>(userdata);
    out->append(ptr, total);
    return total;
}

static bool post_detection_file(const std::string& server_url,
                                const std::string& file_path,
                                int frame_no,
                                int count,
                                float best_score,
                                std::string& response)
{
    response.clear();
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    // Keep metadata aligned with backend events and audio uploader shape.
    const std::string metadata =
        "{"
        "\"timestamp\":\"" + iso8601_utc_now() + "\"," 
        "\"gps\":{"
        "\"lat\":" + std::to_string(GPS_BUCHAREST_LAT) + ","
        "\"lon\":" + std::to_string(GPS_BUCHAREST_LON) +
        "},"
        "\"event\":\"drone detected\"," 
        "\"type\":\"video\"," 
        "\"confidence\":" + std::to_string(best_score) + ","
        "\"frame\":" + std::to_string(frame_no) + ","
        "\"detections\":" + std::to_string(count) + ","
        "\"snippet\":\"" + json_escape(file_path) + "\""
        "}";

    curl_mime* mime = curl_mime_init(curl);
    if (!mime) {
        curl_easy_cleanup(curl);
        return false;
    }

    curl_mimepart* part = curl_mime_addpart(mime);
    curl_mime_name(part, "metadata");
    curl_mime_data(part, metadata.c_str(), CURL_ZERO_TERMINATED);

    part = curl_mime_addpart(mime);
    // Backend accepts both "audio" and "snippet" fields.
    curl_mime_name(part, "snippet");
    curl_mime_filedata(part, file_path.c_str());
    curl_mime_type(part, "image/jpeg");

    curl_easy_setopt(curl, CURLOPT_URL, server_url.c_str());
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    const CURLcode rc = curl_easy_perform(curl);
    long http_code = 0;
    if (rc == CURLE_OK) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    }

    curl_mime_free(mime);
    curl_easy_cleanup(curl);

    return rc == CURLE_OK && http_code >= 200 && http_code < 300;
}
#endif

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path | camera_index | gst-pipeline> [output_path] [--server-url URL]\n"
                  << "  camera_index : 0, 1, ...  (e.g. /dev/video0)\n"
                  << "  gst-pipeline : e.g. 'libcamerasrc ! video/x-raw,width=640,height=480 ! videoconvert ! appsink'\n"
                  << "  --server-url : POST detection snippets to backend /events\n";
        return 1;
    }
    const std::string input_arg = argv[1];

    std::string output_path_arg;
    std::string server_url = DEFAULT_EVENTS_SERVER_URL;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--server-url") {
            if (i + 1 >= argc) {
                std::cerr << "Missing URL after --server-url\n";
                return 1;
            }
            server_url = argv[++i];
            continue;
        }
        if (output_path_arg.empty()) {
            output_path_arg = arg;
            continue;
        }
        std::cerr << "Unknown argument: " << arg << "\n";
        return 1;
    }

    const char* output_path = output_path_arg.empty() ? nullptr : output_path_arg.c_str();
    bool live_mode = is_camera_index(input_arg) || input_arg.find('!') != std::string::npos;

#ifdef DRONE_DEMO_HAS_CURL
    bool upload_enabled = !server_url.empty();
    bool curl_ready = false;
    if (upload_enabled) {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK) {
            curl_ready = true;
            std::cout << "Event upload enabled: " << server_url << "\n";
        } else {
            std::cerr << "Warning: failed to initialize curl; uploads disabled\n";
            upload_enabled = false;
        }
    }
#else
    if (!server_url.empty()) {
        std::cerr << "Warning: --server-url ignored (built without libcurl support)\n";
    }
#endif

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
    bool was_detected_last_frame = false;
    auto last_detection_log_time = std::chrono::steady_clock::now() -
                                   std::chrono::milliseconds(DETECTION_LOG_COOLDOWN_MS);

    std::error_code fs_ec;
    std::filesystem::create_directories(DETECTION_FRAME_DIR, fs_ec);
    if (fs_ec) {
        std::cerr << "Warning: could not create detection frame directory '"
                  << DETECTION_FRAME_DIR << "': " << fs_ec.message() << "\n";
    }

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

        float best_score = 0.0f;
        for (const auto& det : detections) {
            best_score = std::max(best_score, det.score);
        }

        bool detected_now = count > 0;
        if (detected_now) {
            auto now = std::chrono::steady_clock::now();
            auto ms_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    now - last_detection_log_time)
                                    .count();
            if (!was_detected_last_frame || ms_since_last >= DETECTION_LOG_COOLDOWN_MS) {
                std::printf("\n[DETECTED] frame=%d count=%d best_score=%.0f%%\n",
                            frame_no, count, best_score * 100.0f);
                std::fflush(stdout);

                char image_path[256];
                std::snprintf(image_path, sizeof(image_path),
                              "%s/frame_%06d_score_%03d.jpg",
                              DETECTION_FRAME_DIR,
                              frame_no,
                              (int)(best_score * 100.0f));
                if (cv::imwrite(image_path, frame_bgr)) {
                    std::printf("[SAVED] %s\n", image_path);
#ifdef DRONE_DEMO_HAS_CURL
                    if (upload_enabled) {
                        std::string response;
                        if (post_detection_file(server_url, image_path, frame_no, count, best_score, response)) {
                            std::printf("[POST OK] %s\n", server_url.c_str());
                        } else {
                            std::printf("[POST FAILED] %s", server_url.c_str());
                            if (!response.empty()) {
                                std::printf(" response=%s", response.c_str());
                            }
                            std::printf("\n");
                        }
                        std::fflush(stdout);
                    }
#endif
                } else {
                    std::printf("[WARN] failed to save %s\n", image_path);
                }
                std::fflush(stdout);

                last_detection_log_time = now;
            }
        } else if (was_detected_last_frame) {
            std::printf("\n[CLEAR] frame=%d no drone detected\n", frame_no);
            std::fflush(stdout);
        }
        was_detected_last_frame = detected_now;

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

#ifdef DRONE_DEMO_HAS_CURL
    if (curl_ready) {
        curl_global_cleanup();
    }
#endif

    // ── Cleanup ──────────────────────────────────────────────────────
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    return 0;
}
