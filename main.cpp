#include <cstdio>
#include "main.h"
#include "version.h"
#include "include/OcrLite.h"
#include "include/OcrUtils.h"

int main(char** argv) {
    std::string pathToVideo = argv[1];
    std::string modelDetPath, modelClsPath, modelRecPath, keysPath;
    int numThread = 8;
    int padding = 50;
    int maxSideLen = 1024;
    float boxScoreThresh = 0.5f;
    float boxThresh = 0.3f;
    float unClipRatio = 1.6f;
    bool doAngle = true;
    int flagDoAngle = 1;
    bool mostAngle = true;
    int flagMostAngle = 1;
    int flagGpu = 0;
    int opt;
    int optionIndex = 0;

    cv::Mat frame;
    cv::VideoCapture cap;

    OcrLite ocrLite;
    ocrLite.setNumThread(numThread);
    ocrLite.initLogger(
        true,//isOutputConsole
        false,//isOutputPartImg
        false);//isOutputResultImg
    ocrLite.setGpuIndex(flagGpu);
    ocrLite.Logger("=====Input Params=====\n");
    ocrLite.Logger(
        "numThread(%d),padding(%d),maxSideLen(%d),boxScoreThresh(%f),boxThresh(%f),unClipRatio(%f),doAngle(%d),mostAngle(%d),GPU(%d)\n",
        numThread, padding, maxSideLen, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle,
        flagGpu);
    modelDetPath = "./models/en_PP-OCRv3_det_infer.onnx";
    modelClsPath = "./models/ch_ppocr_mobile_v2.0_cls_infer.meta.onnx";
    modelRecPath = "./models/en_PP-OCRv3_rec_infer.meta.onnx";
    keysPath = "./models/en_dict.txt";
    ocrLite.initModels(modelDetPath, modelClsPath, modelRecPath, keysPath);
    cap.open(pathToVideo);
    while (true) {
        cap.read(frame);
        if (frame.empty()) {
            break;
        };
        OcrResult result = ocrLite.detect(frame, padding, maxSideLen,
            boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    }
    return 0;
}