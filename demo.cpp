// demo.cpp — C++ equivalent of demo.py
// Loads a TensorFlow SavedModel, runs drone detection on an image,
// draws bounding boxes, and saves the result.
//
// Build (example with pkg-config / CMake — see CMakeLists.txt):
//   mkdir build && cd build && cmake .. && make
//
// Dependencies:
//   • TensorFlow C library  (libtensorflow >= 2.x)
//   • OpenCV 4.x            (libopencv)

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>

#include <opencv2/opencv.hpp>
#include <tensorflow/c/c_api.h>

// ── Configuration ────────────────────────────────────────────────────
static const char* PATH_TO_MODEL_DIR   = "model";
static const char* PATH_TO_LABELS      = "model/object-detection.pbtxt";
static const float MIN_SCORE_THRESH    = 0.5f;
static const char* IMAGE_PATH          = "heavy-lift-drone.jpg";
static const char* OUTPUT_PATH         = "output.jpg";

// ── Helper: RAII wrapper for TF_Buffer ───────────────────────────────
struct TFBufferDeleter { void operator()(TF_Buffer* b) { if (b) TF_DeleteBuffer(b); } };

// ── Helper: safely free a TF_Tensor* ─────────────────────────────────
static void FreeTensor(TF_Tensor* t) { if (t) TF_DeleteTensor(t); }

// ── Parse label map (.pbtxt) ─────────────────────────────────────────
static std::map<int, std::string> parse_labelmap(const char* path)
{
    std::ifstream ifs(path);
    if (!ifs) { std::cerr << "Cannot open label map: " << path << "\n"; std::exit(1); }
    std::string text((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());

    std::map<int, std::string> category_index;
    // Match each 'item { ... }' block
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

// ── Create a TF_Tensor from a raw buffer (uint8 image) ───────────────
static TF_Tensor* make_input_tensor(const uint8_t* data,
                                     int height, int width, int channels)
{
    const int64_t dims[4] = {1, height, width, channels};
    const size_t  nbytes  = (size_t)height * width * channels;

    TF_Tensor* tensor = TF_AllocateTensor(TF_UINT8, dims, 4, nbytes);
    if (!tensor) { std::cerr << "Failed to allocate input tensor\n"; std::exit(1); }
    std::memcpy(TF_TensorData(tensor), data, nbytes);
    return tensor;
}

// ── Resolve an output operation by name ──────────────────────────────
static TF_Output get_output(TF_Graph* graph, const char* name)
{
    TF_Output out;
    out.oper  = TF_GraphOperationByName(graph, name);
    out.index = 0;
    if (!out.oper) {
        std::cerr << "Operation not found: " << name << "\n";
        std::exit(1);
    }
    return out;
}

// ── Main ─────────────────────────────────────────────────────────────
int main()
{
    // ── 1. Parse labels ──────────────────────────────────────────────
    auto category_index = parse_labelmap(PATH_TO_LABELS);
    std::cout << "Labels:";
    for (auto& kv : category_index) std::cout << " {" << kv.first << ": " << kv.second << "}";
    std::cout << "\n";

    // ── 2. Load TF SavedModel ────────────────────────────────────────
    std::cout << "Loading model...\n";
    TF_Status*        status  = TF_NewStatus();
    TF_SessionOptions* opts   = TF_NewSessionOptions();
    TF_Graph*          graph  = TF_NewGraph();

    const char* tags[] = {"serve"};
    std::string saved_model_dir = std::string(PATH_TO_MODEL_DIR) + "/saved_model";

    TF_Session* session = TF_LoadSessionFromSavedModel(
        opts, /*run_options=*/nullptr,
        saved_model_dir.c_str(),
        tags, 1,
        graph, /*meta_graph_def=*/nullptr,
        status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Failed to load model: " << TF_Message(status) << "\n";
        return 1;
    }
    TF_DeleteSessionOptions(opts);
    std::cout << "Model loaded.\n";

    // ── 3. Read image ────────────────────────────────────────────────
    cv::Mat image_bgr = cv::imread(IMAGE_PATH);
    if (image_bgr.empty()) {
        std::cerr << "Could not read image: " << IMAGE_PATH << "\n";
        return 1;
    }

    cv::Mat image_rgb;
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);
    int h = image_rgb.rows;
    int w = image_rgb.cols;

    // ── 4. Build input tensor ────────────────────────────────────────
    TF_Tensor* input_tensor = make_input_tensor(image_rgb.data, h, w, 3);

    // Input operation — name found via:
    //   saved_model_cli show --dir .../saved_model --tag_set serve --signature_def serving_default
    // or by inspecting model.signatures['serving_default'].inputs in Python.
    TF_Output input_op = get_output(graph, "image_tensor");

    // ── 5. Prepare output operations ─────────────────────────────────
    // Output operation names (same source as above).
    TF_Output output_ops[] = {
        get_output(graph, "detection_boxes"),    // [1, N, 4]  float32
        get_output(graph, "detection_classes"),   // [1, N]     float32
        get_output(graph, "detection_scores"),    // [1, N]     float32
        get_output(graph, "num_detections"),      // [1]        float32
    };
    constexpr int NUM_OUTPUTS = 4;

    // ── 6. Run session ───────────────────────────────────────────────
    TF_Tensor* output_tensors[NUM_OUTPUTS] = {};

    auto t_start = std::chrono::high_resolution_clock::now();

    TF_SessionRun(
        session,
        /*run_options=*/nullptr,
        /*inputs=*/&input_op,      &input_tensor, 1,
        /*outputs=*/output_ops,    output_tensors, NUM_OUTPUTS,
        /*targets=*/nullptr, 0,
        /*run_metadata=*/nullptr,
        status);

    auto t_end = std::chrono::high_resolution_clock::now();

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Inference failed: " << TF_Message(status) << "\n";
        return 1;
    }
    FreeTensor(input_tensor);

    // ── Inference timing ─────────────────────────────────────────────
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double megapixels = (double)h * w / 1.0e6;
    double ms_per_mp  = elapsed_ms / megapixels;
    std::printf("Inference: %.1f ms  |  Image: %.2f MP  |  %.1f ms/MP\n",
                elapsed_ms, megapixels, ms_per_mp);

    // ── 7. Extract results ───────────────────────────────────────────
    // Outputs are in the same order we requested above:
    //   0 – detection_boxes   [1, N, 4]  float32
    //   1 – detection_classes [1, N]     float32
    //   2 – detection_scores  [1, N]     float32
    //   3 – num_detections    [1]        float32
    float* boxes_data   = (float*)TF_TensorData(output_tensors[0]);
    float* classes_data = (float*)TF_TensorData(output_tensors[1]);
    float* scores_data  = (float*)TF_TensorData(output_tensors[2]);
    int    num_det      = (int)(*(float*)TF_TensorData(output_tensors[3]));

    // ── 8. Draw bounding boxes ───────────────────────────────────────
    int count = 0;
    for (int i = 0; i < num_det; ++i) {
        float score = scores_data[i];
        if (score < MIN_SCORE_THRESH) continue;

        float ymin = boxes_data[i * 4 + 0];
        float xmin = boxes_data[i * 4 + 1];
        float ymax = boxes_data[i * 4 + 2];
        float xmax = boxes_data[i * 4 + 3];

        int left   = (int)(xmin * w);
        int top    = (int)(ymin * h);
        int right  = (int)(xmax * w);
        int bottom = (int)(ymax * h);

        int cls = (int)classes_data[i];
        std::string label;
        auto it = category_index.find(cls);
        if (it != category_index.end()) label = it->second;
        else {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "class %d", cls);
            label = buf;
        }

        char text[128];
        std::snprintf(text, sizeof(text), "%s: %.0f%%", label.c_str(), score * 100.0f);

        cv::rectangle(image_bgr, cv::Point(left, top), cv::Point(right, bottom),
                      cv::Scalar(0, 255, 0), 3);
        cv::putText(image_bgr, text, cv::Point(left, top - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        ++count;
    }

    // ── 9. Save result ───────────────────────────────────────────────
    cv::imwrite(OUTPUT_PATH, image_bgr);
    std::cout << "Saved " << OUTPUT_PATH << " with " << count << " detection(s).\n";

    // ── Cleanup ──────────────────────────────────────────────────────
    for (int i = 0; i < NUM_OUTPUTS; ++i) FreeTensor(output_tensors[i]);
    TF_CloseSession(session, status);
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);

    return 0;
}
